"""Real measured SDR validation with pilot-aided equalization."""

from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np

from adarx.model import NeuralReceiver
from adarx.centroid import CentroidReceiver
from adarx.sdr_dataset import load_sdr_pth
from adarx.metrics import real_sdr_metrics

def standardize(train_frames, frames, key):
    x = np.vstack([f[key] for f in train_frames])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    for f in frames:
        f[key+"_norm"] = (f[key] - mean) / std

def train_raw_mlp(train_frames, seed=7, epochs=3, lr=0.01):
    m = NeuralReceiver(seed=seed, hidden=48, n_classes=16)
    x = np.vstack([f["raw_x_norm"] for f in train_frames])
    y = np.concatenate([f["y"] for f in train_frames])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), 2048):
            idx = order[s:s+2048]
            m.update(x[idx], y[idx], lr=lr, steps=1)
    return m

def train_equalized_centroid(train_frames):
    z = np.concatenate([f["eq_rx"] for f in train_frames])
    y = np.concatenate([f["y"] for f in train_frames])
    return CentroidReceiver(16).fit(z, y)

def run(dataset_path, train_ttis=50, eval_ttis=100, seed=7,
        output="results_equalized_sdr", adapt_confidence=0.45):
    frames = load_sdr_pth(dataset_path, max_ttis=train_ttis+eval_ttis)
    if len(frames) <= train_ttis:
        raise ValueError("not enough TTIs")
    train = frames[:train_ttis]
    test = frames[train_ttis:]
    standardize(train, frames, "raw_x")

    raw_mlp = train_raw_mlp(train, seed=seed)
    eq_static = train_equalized_centroid(train)
    eq_adapt = train_equalized_centroid(train)

    rows = []
    methods = ("raw_static", "equalized_static", "equalized_adapted")
    for i, f in enumerate(test):
        pred = raw_mlp.predict(f["raw_x_norm"])
        met = real_sdr_metrics(f["y"], pred)
        rows.append({
            "method":"raw_static", "frame":i, "source_frame_id":f["frame_id"],
            "sinr_db":f["sinr_db"], **met, "update_samples":0, "update_ms":0.0,
            "weight_drift":0.0, "mean_confidence":None,
            "channel_mag_mean":f["channel_mag_mean"], "channel_mag_std":f["channel_mag_std"],
        })

        pred = eq_static.predict(f["eq_rx"])
        met = real_sdr_metrics(f["y"], pred)
        rows.append({
            "method":"equalized_static", "frame":i, "source_frame_id":f["frame_id"],
            "sinr_db":f["sinr_db"], **met, "update_samples":0, "update_ms":0.0,
            "weight_drift":0.0, "mean_confidence":float(np.mean(eq_static.confidence(f["eq_rx"]))),
            "channel_mag_mean":f["channel_mag_mean"], "channel_mag_std":f["channel_mag_std"],
        })

        pred = eq_adapt.predict(f["eq_rx"])
        met = real_sdr_metrics(f["y"], pred)
        start = time.perf_counter()
        n_update, mean_conf = eq_adapt.adapt(
            f["eq_rx"], confidence=adapt_confidence, rate=0.01, max_samples=256
        )
        update_ms = (time.perf_counter()-start) * 1000.0
        rows.append({
            "method":"equalized_adapted", "frame":i, "source_frame_id":f["frame_id"],
            "sinr_db":f["sinr_db"], **met, "update_samples":n_update, "update_ms":update_ms,
            "weight_drift":eq_adapt.drift(), "mean_confidence":mean_conf,
            "channel_mag_mean":f["channel_mag_mean"], "channel_mag_std":f["channel_mag_std"],
        })

    summary = {}
    for method in methods:
        r = [x for x in rows if x["method"] == method]
        summary[method] = {
            "mean_ber":float(np.mean([x["ber"] for x in r])),
            "mean_ser":float(np.mean([x["ser"] for x in r])),
            "mean_bler":float(np.mean([x["bler"] for x in r])),
            "mean_throughput":float(np.mean([x["throughput"] for x in r])),
            "mean_sinr_db":float(np.mean([x["sinr_db"] for x in r])),
            "mean_update_samples":float(np.mean([x["update_samples"] for x in r])),
            "total_update_ms":float(np.sum([x["update_ms"] for x in r])),
            "final_weight_drift":float(r[-1]["weight_drift"]),
            "evaluated_ttis":len(r),
        }

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"frame_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    payload = {
        "dataset":"rikluost/sdr_ofdm_dataset_v1",
        "source_train_ttis":train_ttis,
        "evaluation_ttis":len(test),
        "seed":seed,
        "pilot_equalization":"Ypilot/Xpilot with complex linear interpolation over 102 active bins",
        "adaptation":"high-confidence centroid self-training; no evaluation labels used",
        "adapt_confidence":adapt_confidence,
        "summary":summary,
    }
    (out/"summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return rows, payload

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/ofdm_valset_4_8_sdr.pth")
    p.add_argument("--train-ttis", type=int, default=50)
    p.add_argument("--eval-ttis", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--adapt-confidence", type=float, default=0.45)
    p.add_argument("--output", default="results_equalized_sdr")
    a = p.parse_args()
    run(a.dataset, a.train_ttis, a.eval_ttis, a.seed, a.output, a.adapt_confidence)
