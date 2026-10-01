"""Validate AdaRx on measured over-the-air SDR OFDM data."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from adarx.model import NeuralReceiver
from adarx.sdr_dataset import load_sdr_pth
from adarx.adaptation import pseudo_label_adapt
from adarx.metrics import real_sdr_metrics


def normalize_features(train_frames, frames):
    train_x = np.vstack([f["x"] for f in train_frames])
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    for f in frames:
        f["x_norm"] = (f["x"] - mean) / std
    return mean, std


def train_source_model(frames, seed=7, hidden=48, epochs=3, lr=0.01):
    model = NeuralReceiver(seed=seed, hidden=hidden, n_classes=16)
    x = np.vstack([f["x_norm"] for f in frames])
    y = np.concatenate([f["y"] for f in frames])
    rng = np.random.default_rng(seed)
    batch = 2048
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch):
            idx = order[start:start+batch]
            model.update(x[idx], y[idx], lr=lr, steps=1)
    model.anchor = model.state()
    return model


def run(dataset_path, train_ttis=100, eval_ttis=300, seed=7,
        output="results_real_sdr", confidence=0.92):
    frames = load_sdr_pth(dataset_path, max_ttis=train_ttis + eval_ttis)
    if len(frames) <= train_ttis:
        raise ValueError("dataset does not contain enough TTIs for requested split")

    train_frames = frames[:train_ttis]
    eval_frames = frames[train_ttis:]
    normalize_features(train_frames, frames)

    base = train_source_model(train_frames, seed=seed)
    initial = base.state()

    all_rows = []
    summary = {}

    for method in ("static", "pseudo"):
        receiver = NeuralReceiver(seed=seed, hidden=48, n_classes=16)
        receiver.load(initial)
        receiver.anchor = tuple(x.copy() for x in initial)
        rows = []

        for i, frame in enumerate(eval_frames):
            x = frame["x_norm"]
            pred = receiver.predict(x)
            metrics = real_sdr_metrics(frame["y"], pred)

            if method == "pseudo":
                loss, update_ms, update_samples = pseudo_label_adapt(
                    receiver, x, confidence=confidence
                )
            else:
                loss, update_ms, update_samples = 0.0, 0.0, 0

            row = {
                "method": method,
                "frame": i,
                "source_frame_id": frame["frame_id"],
                "sinr_db": frame["sinr_db"],
                **metrics,
                "update_loss": loss,
                "update_ms": update_ms,
                "update_samples": update_samples,
                "weight_drift": receiver.drift(),
            }
            rows.append(row)
            all_rows.append(row)

        summary[method] = {
            "mean_ber": float(np.mean([r["ber"] for r in rows])),
            "mean_ser": float(np.mean([r["ser"] for r in rows])),
            "mean_bler": float(np.mean([r["bler"] for r in rows])),
            "mean_throughput": float(np.mean([r["throughput"] for r in rows])),
            "mean_sinr_db": float(np.mean([r["sinr_db"] for r in rows])),
            "total_update_ms": float(sum(r["update_ms"] for r in rows)),
            "mean_update_samples": float(np.mean([r["update_samples"] for r in rows])),
            "final_weight_drift": float(rows[-1]["weight_drift"]),
            "evaluated_ttis": len(rows),
        }

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"frame_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "dataset": "rikluost/sdr_ofdm_dataset_v1",
        "dataset_file": str(dataset_path),
        "source_train_ttis": train_ttis,
        "evaluation_ttis": len(eval_frames),
        "seed": seed,
        "confidence_threshold": confidence,
        "adaptation": "high-confidence pseudo-label self-training with anchor regularization",
        "label_use_note": "Evaluation labels are not used for online adaptation.",
        "summary": summary,
    }
    (out/"summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return all_rows, metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/ofdm_valset_4_8_sdr.pth")
    p.add_argument("--train-ttis", type=int, default=100)
    p.add_argument("--eval-ttis", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--confidence", type=float, default=0.92)
    p.add_argument("--output", default="results_real_sdr")
    a = p.parse_args()
    run(a.dataset, a.train_ttis, a.eval_ttis, a.seed, a.output, a.confidence)
