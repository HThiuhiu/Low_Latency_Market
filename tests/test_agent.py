"""Trading agent: LLM advice parsing, risk-manager rules, when to ask."""

import pytest

from qforecast.agent.core import AgentConfig, AgentCore, RiskManager, normalise_advice
from qforecast.agent.llm import LLMError, parse_json
from qforecast.sim.broker import PaperBroker, SimConfig


def adv(action="OPEN_LONG", conf=0.8, size=0.5, stop=40, take=80, hz=30):
    return {"action": action, "confidence": conf, "size": size, "stop_bps": stop,
            "take_bps": take, "horizon_bars": hz, "reason": "x"}


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"action": "SKIP"}\n```') == {"action": "SKIP"}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_normalise_maps_synonyms_and_clamps():
    a = normalise_advice({"action": "buy", "confidence": 3, "size": -1, "stop_bps": "abc"})
    assert (a["action"], a["confidence"], a["size"], a["stop_bps"]) == ("OPEN_LONG", 1.0, 0.0, 0.0)
    assert normalise_advice({"action": "moon"})["action"] == "SKIP"


def test_veto_mode_blocks_disagreeing_direction():
    rm = RiskManager(AgentConfig(mode="veto"))
    d = rm.decide(adv("OPEN_SHORT"), "LONG", 15, 0, [])
    assert d.action == "NONE" and any("disagrees" in v for v in d.vetoes)
    free = RiskManager(AgentConfig(mode="free"))
    assert free.decide(adv("OPEN_SHORT"), "LONG", 15, 0, []).action == "OPEN_SHORT"


def test_risk_clamps_and_floors():
    cfg = AgentConfig(max_size=0.3, min_stop_bps=20, max_stop_bps=100, max_horizon=60)
    rm = RiskManager(cfg)
    d = rm.decide(adv(size=0.9, stop=500, take=9999, hz=999), "LONG", 15, 0, [])
    assert (d.size, d.stop_bps, d.take_bps, d.horizon) == (0.3, 100, cfg.max_take_bps, 60)
    d = rm.decide(adv(stop=0, hz=0), "LONG", 15, 0, [])
    assert d.stop_bps == cfg.default_stop_bps and d.horizon == 15  # mandatory stop, model horizon
    assert rm.decide(adv(conf=0.5), "LONG", 15, 0, []).action == "NONE"
    assert rm.decide(adv(), "LONG", 15, 0, ["daily loss limit hit"]).action == "NONE"
    assert rm.decide(None, "LONG", 15, 0, []).action == "NONE"


def test_close_rules():
    rm = RiskManager(AgentConfig())
    assert rm.decide(adv("CLOSE", conf=0.7), "FLAT", 15, 1, []).action == "CLOSE"
    assert rm.decide(adv("CLOSE", conf=0.7), "FLAT", 15, 0, []).action == "NONE"


def _core(**kw):
    return AgentCore(PaperBroker(SimConfig(cash=1000, fee_bps=0, slippage_bps=0)),
                     AgentConfig(**kw), 60_000)


def test_should_ask_budget_and_gap():
    core = _core(max_calls_hour=2, entry_gap_bars=0)
    for i in range(3):
        core.on_bar(i * 60_000, 100, 100, 100, 100)
    assert core.should_ask("FLAT", 0) is None
    assert core.should_ask("LONG", 0) == "entry"
    core.mark_asked(0)
    core.mark_asked(1)
    assert core.should_ask("LONG", 2) is None  # hourly budget used up
    assert core.should_ask("LONG", 3_700_000) == "entry"


def test_execute_with_per_trade_stop_and_cooldown():
    core = _core(cooldown_bars=3, entry_gap_bars=0)
    core.on_bar(0, 100, 100, 100, 100)
    d = core.decide(adv(stop=50, take=0, hz=10), "LONG", 10)
    core.execute(d, 0, 100.0)
    assert core.broker.pos.stop_bps == 50 and core.broker.pos.qty == pytest.approx(5.0)
    core.on_bar(60_000, 100, 100, 99, 99.6)  # low 99 < stop 99.5 -> stopped out at a loss
    assert core.broker.trades[0].reason == "stop"
    assert core.blocked() and core.should_ask("LONG", 60_000) is None


def test_daily_loss_halts_entries():
    core = _core(daily_loss_pct=1.0)
    core.on_bar(0, 100, 100, 100, 100)
    core.execute(core.decide(adv(size=1.0, stop=150), "LONG", 10), 0, 100.0)
    core.on_bar(60_000, 98.7, 98.7, 98.6, 98.6)  # -1.4% with stop at -1.5%: still open
    assert "daily loss limit hit" in core.blocked()
