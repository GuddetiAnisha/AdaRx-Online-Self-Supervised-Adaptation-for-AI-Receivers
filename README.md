# AdaRx — Online Self-Supervised Adaptation for AI Receivers

AdaRx is a software-only research prototype for evaluating neural receiver robustness and online adaptation under changing wireless conditions.

## Real measured SDR-OFDM validation

The project supports the public **Measured Radio Channel Dataset for OFDM Receiver Validation** from `rikluost/sdr_ofdm_dataset_v1`.

The measured dataset contains:
- 435 MHz center frequency
- OFDM with FFT size 128
- 14 OFDM symbols per TTI
- 16-QAM payload modulation
- approximately 1000 TTIs
- complex received IQ samples
- transmitted 4-bit payload labels
- measured SINR

## Pilot-aided equalization

For each TTI, AdaRx now:
1. extracts the published pilot tones from OFDM symbol index 2,
2. estimates the complex channel using `H = Ypilot / Xpilot`,
3. linearly interpolates the real and imaginary channel components across the 102 active frequency bins,
4. equalizes the payload IQ,
5. evaluates raw and equalized receiver baselines,
6. optionally performs high-confidence online centroid self-training.

The held-out transmitted payload labels are used for evaluation only and are not used for online adaptation.

## Receiver variants

The measured-data benchmark compares:
- **raw_static** — lightweight 16-class MLP operating on raw received IQ features
- **equalized_static** — 16-QAM centroid receiver operating on pilot-equalized payload IQ
- **equalized_adapted** — equalized centroid receiver with confidence-filtered online pseudo-label updates

## Final measured-data result

Final configuration:
- 100 source-training TTIs
- 300 chronological held-out evaluation TTIs
- seed 7
- adaptation confidence threshold 0.85
- mean measured SINR 14.04 dB

| Receiver | BER | SER | BLER | Throughput proxy |
|---|---:|---:|---:|---:|
| Raw static | 46.03% | 85.78% | 100.00% | 0.00% |
| Equalized static | **9.00%** | **24.13%** | **66.19%** | **33.81%** |
| Equalized adapted | 9.04% | 24.22% | 66.37% | 33.63% |

### Main finding

Pilot-aided channel estimation and equalization produced the dominant improvement, reducing BER from about **46.0% on raw IQ to 9.0% on equalized IQ** across 300 held-out TTIs. SER decreased from about **85.8% to 24.1%**.

The online adaptation mechanism was active, averaging about **157 update samples per TTI**, but it did not outperform the equalized static receiver. It is therefore reported as an ablation result rather than as an adaptation gain.

See [`RESULTS.md`](RESULTS.md) and [`results/real_sdr_300_tti_summary.json`](results/real_sdr_300_tti_summary.json) for the recorded final result.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Download the real dataset

```powershell
.\.venv\Scripts\python.exe download_real_sdr.py
```

This downloads `ofdm_valset_4_8_sdr.pth` into `data/`.

## Quick validation

```powershell
.\.venv\Scripts\python.exe run_real_sdr_validation.py --train-ttis 50 --eval-ttis 100 --adapt-confidence 0.85
```

## Final 300-TTI validation

```powershell
.\.venv\Scripts\python.exe run_real_sdr_validation.py --train-ttis 100 --eval-ttis 300 --adapt-confidence 0.85
```

Outputs are written to:

```text
results_equalized_sdr/frame_metrics.csv
results_equalized_sdr/summary.json
```

Metrics include:
- bit error rate (BER)
- 16-QAM symbol error rate (SER)
- block error rate (BLER)
- normalized throughput proxy
- measured SINR
- online update time
- pseudo-label update count
- receiver drift

## Original synthetic experiment

The reproducible synthetic QPSK-like domain-shift experiment remains available:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --frames 120 --output results
```

## Research limitations

- This is a link-level research prototype, not a standards-compliant 5G NR receiver.
- The measured dataset uses 435 MHz SDR OFDM and 16-QAM.
- Results must not be presented as Ericsson or other vendor field-trial results.
- The equalized static receiver outperformed the tested pseudo-label adaptation configuration, so no adaptation improvement is claimed.
- The held-out measured stream is chronological rather than a randomized IID test split.

## CV-safe description

**AdaRx — Online Self-Supervised Adaptation for AI Receivers**
- Built a software-only neural receiver validation pipeline using public measured over-the-air SDR-OFDM IQ data, 16-QAM transmitted labels, and measured SINR.
- Implemented pilot-aided channel estimation and frequency-domain equalization, reducing held-out BER from 46.0% on raw IQ to 9.0% on equalized IQ across 300 TTIs.
- Evaluated confidence-filtered online pseudo-label adaptation as an ablation using BER, SER, BLER, throughput, update cost, and model-drift metrics.
