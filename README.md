# AdaRx — Online Self-Supervised Adaptation for AI Receivers

AdaRx is a thesis-oriented, software-only research prototype for evaluating how a neural PxSCH-like receiver adapts when wireless conditions change. It uses reproducible synthetic QPSK/OFDM-like frames because proprietary field-trial data is not included.

## Experiment

The receiver is trained offline in a source domain and evaluated on a stream whose SNR, phase, fading amplitude, and IQ imbalance change. Three strategies are compared:

1. **Static:** no parameter updates after offline training.
2. **Pilot self-supervision:** online updates use known pilot symbols already present in each frame.
3. **Hybrid:** pilot updates plus confidence-filtered pseudo-labels, entropy minimization, and an anchor penalty that limits model drift.

Reported metrics include BER, BLER, normalized throughput, adaptation time, update cost, and weight drift. Results are written to CSV and JSON.

## Run

```bash
cd adaptive_ai_receiver
python run_experiment.py --frames 120 --output results
python dashboard.py --csv results/frame_metrics.csv
```

The dashboard uses Matplotlib and opens a local window. For headless use:

```bash
python dashboard.py --csv results/frame_metrics.csv --save results/summary.png
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Field-data adapter

`adarx/data.py` accepts a CSV with columns `frame_id`, `rx_i`, `rx_q`, and optional `tx_class`, `is_pilot`, and `snr_db`. Field data without `tx_class` can still be used for inference and pilot-driven adaptation when pilot positions and expected pilot symbols are supplied by the experiment configuration.

## Research limitations

- This is a link-level proof of concept, not a standards-compliant 5G NR PxSCH implementation.
- Synthetic QPSK frames do not reproduce full MIMO, coding, HARQ, synchronization, or RF impairments.
- Pilot-based learning is self-supervised through known waveform structure; pseudo-label updates can reinforce errors and therefore use confidence filtering and drift regularization.
- Results from synthetic data must not be presented as Ericsson field-trial results.

## Suggested thesis extensions

- Replace the NumPy MLP with a CNN/Transformer receiver in PyTorch or JAX.
- Add LDPC coding, BLER after decoding, MIMO layers, channel estimation, mobility/Doppler, and 3GPP channel models.
- Compare test-time training, teacher-student adaptation, contrastive learning, replay buffers, and change-point-triggered updates.
- Report multiple seeds, confidence intervals, effect sizes, and failure cases.

