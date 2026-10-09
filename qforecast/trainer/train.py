"""Offline training job: download -> features -> train -> ONNX export -> evaluate.

    python -m qforecast.trainer.train --days 90 --epochs 20 --device auto

Artifacts (written atomically so the signal service can hot-reload them):
    artifacts/model.onnx, artifacts/model_meta.json, artifacts/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qforecast.config import Settings
from qforecast.core.features import B_CLOSE, FEATURE_NAMES, N_FEATURES, compute_features
from qforecast.exchange.binance import raw_to_bars
from qforecast.model.net import build_model
from qforecast.trainer.data import load_history_sync
from qforecast.trainer.dataset import (
    WindowBatcher,
    chronological_split,
    eligible_indices,
    fit_normaliser,
    make_targets,
)
from qforecast.trainer.evaluate import horizon_metrics, strategy_backtest

log = logging.getLogger("trainer")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model, batcher, idx, device, bs=4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x, _ = batcher.batch(idx[i:i + bs], device)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def train(args) -> dict:
    s = Settings.from_env()
    horizons = list(s.horizons)
    max_h, L = max(horizons), s.lookback
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    torch.set_num_threads(os.cpu_count() or 4)
    log.info("device=%s threads=%d", device, torch.get_num_threads())

    # ---- data & features --------------------------------------------------
    raw = load_history_sync(s, args.days)
    open_time, bars = raw_to_bars(raw)
    t0 = time.perf_counter()
    feats, valid = compute_features(bars)
    log.info("features for %d bars in %.3fs", len(bars), time.perf_counter() - t0)
    y = make_targets(bars[:, B_CLOSE], horizons)
    idx = eligible_indices(open_time, valid, s.interval_ms, L, max_h)
    sp = chronological_split(idx, embargo=max_h)
    log.info("samples train=%d val=%d test=%d", len(sp.train), len(sp.val), len(sp.test))

    mean, std = fit_normaliser(feats, sp.train, L)
    z = np.clip((np.nan_to_num(feats) - mean) / std, -5, 5).astype(np.float32)
    y_scale = np.nanstd(y[sp.train], axis=0).astype(np.float32)
    yz = np.nan_to_num(y / y_scale).astype(np.float32)
    batcher = WindowBatcher(z, yz, L)

    # ---- model --------------------------------------------------------------
    kw = {"hidden": args.hidden} if args.arch == "lstm" else {"scan_backend": args.scan_backend}
    model = build_model(args.arch, N_FEATURES, len(horizons), **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %d", n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(sp.train) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch)
    loss_fn = nn.HuberLoss(delta=1.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        perm = np.random.permutation(sp.train)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            xb, yb = batcher.batch(perm[i:i + args.batch_size], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(xb)
        val_pred = predict(model, batcher, sp.val, device)
        val_loss = float(loss_fn(torch.from_numpy(val_pred), torch.from_numpy(yz[sp.val])))
        log.info("epoch %2d  train %.5f  val %.5f  (%.1fs)", epoch, tot / len(perm), val_loss,
                 time.perf_counter() - t_ep)
        if val_loss < best - 1e-5:
            best, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("early stopping")
                break
    model.load_state_dict(best_state)

    # ---- evaluation (log-return units) ------------------------------------
    val_pred = predict(model, batcher, sp.val, device) * y_scale
    sigma = np.std(y[sp.val] - val_pred, axis=0).astype(np.float32)
    test_pred = predict(model, batcher, sp.test, device) * y_scale
    y_test = y[sp.test]
    logc = np.log(bars[:, B_CLOSE])
    per_h = {}
    for j, h in enumerate(horizons):
        past = logc[sp.test] - logc[sp.test - h]
        per_h[str(h)] = horizon_metrics(test_pred[:, j], y_test[:, j], past)
    bars_per_year = 365 * 86_400_000 / s.interval_ms
    strat = [strategy_backtest(test_pred[:, j], y_test[:, j], h, s.signal_threshold_bps,
                               args.cost_bps, bars_per_year) for j, h in enumerate(horizons)]

    # ---- export -------------------------------------------------------------
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model = model.float().cpu().eval()
    tmp_onnx = s.artifacts_dir / "model.onnx.tmp"
    dummy = torch.zeros(1, L, N_FEATURES)
    # Static batch=1 graph: serving is always batch-1 and fixed shapes let ORT
    # pre-plan memory and fuse more aggressively (~10% faster than dynamic axes).
    torch.onnx.export(model, (dummy,), str(tmp_onnx), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    parity = onnx_parity(model, tmp_onnx, z, sp.test[:128], L)
    version = hashlib.sha256(tmp_onnx.read_bytes()).hexdigest()[:12]

    meta = {
        "version": version, "arch": args.arch, "symbol": s.symbol, "interval": s.interval,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": list(FEATURE_NAMES), "lookback": L, "horizons": horizons,
        "feat_mean": mean.tolist(), "feat_std": std.tolist(),
        "y_scale": y_scale.tolist(), "sigma": sigma.tolist(), "n_params": n_params,
    }
    report = {
        "version": version, "arch": args.arch, "device": str(device), "days": args.days,
        "bars": int(len(bars)),
        "samples": {"train": int(len(sp.train)), "val": int(len(sp.val)), "test": int(len(sp.test))},
        "best_val_loss": best, "onnx_max_abs_diff": parity, "test_metrics": per_h,
        "strategy": strat,
    }
    _atomic_write(s.meta_path, json.dumps(meta, indent=2))
    os.replace(tmp_onnx, s.model_path)  # model last: its mtime triggers the hot reload
    _atomic_write(s.artifacts_dir / "report.json", json.dumps(report, indent=2))
    log.info("exported model %s (onnx parity %.2e)", version, parity)
    log.info("test metrics: %s", json.dumps(per_h, indent=1))
    return report


def onnx_parity(model, onnx_path: Path, z: np.ndarray, idx: np.ndarray, L: int) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for i in idx:
            x = np.ascontiguousarray(z[None, i - L + 1:i + 1], dtype=np.float32)
            ref = model(torch.from_numpy(x)).numpy()
            worst = max(worst, float(np.max(np.abs(ref - sess.run(None, {"x": x})[0]))))
    return worst


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--arch", default="lstm", choices=["lstm", "mamba"])
    p.add_argument("--scan-backend", default="auto", choices=["auto", "parallel", "chunked", "triton"],
                   help="mamba only: auto = fused triton kernel on CUDA, parallel scan otherwise")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--cost-bps", type=float, default=10.0, help="round-trip cost for the backtest")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
