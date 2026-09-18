import argparse, csv, json
from pathlib import Path
import numpy as np
from adarx.channel import Domain, generate_frame, domain_schedule
from adarx.model import NeuralReceiver
from adarx.adaptation import adapt
from adarx.metrics import frame_metrics, adaptation_speed

def run(seed=7, frames=120, symbols=512, output="results"):
    rng = np.random.default_rng(seed)
    model = NeuralReceiver(seed=seed)
    source = Domain(snr_db=20, phase_rad=0.05, gain=1.0)
    for _ in range(100):
        f = generate_frame(rng, symbols, 0.15, source)
        model.update(f["x"], f["y"], lr=0.025, steps=1)
    model.anchor = model.state()
    initial = model.state()
    rows, summary = [], {}
    for method in ("static", "pilot", "hybrid"):
        receiver = NeuralReceiver(seed=seed); receiver.load(initial); receiver.anchor = tuple(x.copy() for x in initial)
        method_rows = []
        stream_rng = np.random.default_rng(seed + 100)
        for idx in range(frames):
            domain_name, domain = domain_schedule(idx, frames)
            frame = generate_frame(stream_rng, symbols, 0.10, domain)
            prediction = receiver.predict(frame["x"])
            metrics = frame_metrics(frame["y"], prediction, ~frame["pilot_mask"])
            loss, update_ms, n_update = adapt(receiver, frame, method)
            row = {"method":method,"frame":idx,"domain":domain_name,"snr_db":domain.snr_db,
                   **metrics,"update_loss":loss,"update_ms":update_ms,"update_samples":n_update,
                   "weight_drift":receiver.drift()}
            rows.append(row); method_rows.append(row)
        summary[method] = {
            "mean_ber": float(np.mean([r["ber"] for r in method_rows])),
            "mean_bler": float(np.mean([r["bler"] for r in method_rows])),
            "mean_throughput": float(np.mean([r["throughput"] for r in method_rows])),
            "total_update_ms": float(sum(r["update_ms"] for r in method_rows)),
            "final_weight_drift": method_rows[-1]["weight_drift"],
            "adaptation_frames_after_first_shift": adaptation_speed(method_rows, frames//3)
        }
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with (out/"frame_metrics.csv").open("w", newline="") as fh:
        writer=csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return rows, summary

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=7); p.add_argument("--frames",type=int,default=120)
    p.add_argument("--symbols",type=int,default=512); p.add_argument("--output",default="results"); a=p.parse_args()
    run(a.seed,a.frames,a.symbols,a.output)

