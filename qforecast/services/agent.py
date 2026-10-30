"""Agent service (live, paper only): ONNX forecast -> LLM advice -> risk manager -> paper broker.

    QF_AGENT=1 python -m qforecast.services.all       # runs next to the plain `paper` service
    python -m qforecast.services.agent                 # standalone (needs ingestor + signal on the bus)

The LLM call (seconds) runs as a background task so the bus consumer never blocks; at
most one call is in flight. Every decision is appended to data/agent/<symbol>.jsonl and
the state is published on ``sim.agent.<symbol>`` (gateway: GET /v1/agent).
"""

from __future__ import annotations

import asyncio
import logging
import time

import msgspec
import orjson

from qforecast.agent.core import SYSTEM_PROMPT, AgentConfig, AgentCore
from qforecast.agent.llm import LLMClient, LLMError
from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.schemas import Forecast, decode_analysis, decode_forecast, decode_md
from qforecast.sim.broker import PaperBroker, SimConfig

log = logging.getLogger("agent")


def agent_config(s: Settings) -> AgentConfig:
    return AgentConfig(mode=s.agent_mode, min_conf=s.agent_min_conf, max_size=s.agent_max_size,
                       daily_loss_pct=s.agent_daily_loss_pct, cooldown_bars=s.agent_cooldown_bars,
                       review_bars=s.agent_review_bars, max_calls_hour=s.agent_max_calls_hour)


def forecast_view(f: Forecast) -> dict:
    return {"signal": f.signal, "best_horizon_bars": f.signal_horizon, "edge_bps": round(f.edge_bps, 1),
            "by_horizon": {str(h): {"exp_bps": round(e * 1e4, 1), "resid_sigma_bps": round(sg * 1e4, 1)}
                           for h, e, sg in zip(f.horizons, f.exp_logret, f.sigma, strict=True)}}


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    broker = PaperBroker(SimConfig(cash=s.paper_cash, fee_bps=s.paper_fee_bps,
                                   slippage_bps=s.paper_slippage_bps, stop_bps=50.0))
    core = AgentCore(broker, agent_config(s), s.interval_ms)
    llm = LLMClient(model=s.agent_model, timeout_s=s.agent_timeout_s)
    if not llm.configured:
        log.error("DEEPSEEK_API_KEY not set: agent will never open trades")
    log_path = s.data_dir / "agent" / f"{s.symbol}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = int(time.time() * 1000)
    analysis: dict | None = None
    busy = False

    async def publish() -> None:
        st = core.state()
        st.update(symbol=s.symbol, ts=int(time.time() * 1000), llm_calls=llm.calls, llm_tokens=llm.tokens,
                  model=llm.model, return_pct=(broker.equity() / broker.cfg.cash - 1) * 100)
        await bus.publish(s.t_agent, orjson.dumps(st))

    async def consult(f: Forecast, reason: str) -> None:
        nonlocal busy
        ctx = core.context(s.symbol, f.bar_time, f.close, forecast_view(f), analysis)
        advice, lat, err = None, 0.0, ""
        try:
            advice, lat = await llm.ask_json(SYSTEM_PROMPT, core.prompt(ctx))
        except LLMError as e:
            err = str(e)
            reg.inc("llm_errors")
        try:
            price = analysis["mid"] if analysis and analysis.get("mid") else broker.last_price or f.close
            d = core.decide(advice, f.signal, f.signal_horizon)
            if time.time() * 1000 - f.bar_time > 3 * s.interval_ms and d.action.startswith("OPEN"):
                d.vetoes.append("advice arrived too late")
                d.action = "NONE"
            core.execute(d, f.bar_time, price)
            rec = core.record(f.bar_time, reason, d, lat, err)
            with open(log_path, "ab") as fh:
                fh.write(orjson.dumps({**rec, "context": ctx}) + b"\n")
            log.info("[%s] model=%s llm=%s(%.2f) -> %s %s %s", reason, f.signal,
                     d.advice.get("action", "-"), d.advice.get("confidence", 0), d.action,
                     "; ".join(d.vetoes), (d.advice.get("reason") or err)[:120])
            reg.inc("agent_decisions")
            await publish()
        finally:
            busy = False

    log.info("agent on %s: model=%s mode=%s (paper only)", s.symbol, llm.model, s.agent_mode)
    tasks: set[asyncio.Task] = set()
    try:
        async for topic, payload in bus.subscribe([s.t_kline, s.t_forecast, s.t_analysis], replay=True):
            if topic == s.t_analysis:
                a = msgspec.to_builtins(decode_analysis(payload))
                analysis = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()
                            if k not in ("symbol", "ts")}
            elif topic == s.t_kline:
                k = decode_md(payload)
                live = k.close_time >= started - s.interval_ms
                core.on_bar(k.open_time, k.open, k.high, k.low, k.close, trade=live)
                if live:
                    await publish()
            else:
                f = decode_forecast(payload)
                if f.bar_time < started - 2 * s.interval_ms or busy:
                    continue
                reason = core.should_ask(f.signal, int(time.time() * 1000))
                if reason:
                    busy = True
                    core.mark_asked(int(time.time() * 1000))
                    t = asyncio.create_task(consult(f, reason))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
    finally:
        await llm.aclose()


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("agent", run)


if __name__ == "__main__":
    main()
