# AdaRx — Online Self-Supervised Adaptation for AI Receivers

AdaRx is a software-only research prototype for evaluating how a neural receiver adapts when wireless conditions change.

## Real measured SDR-OFDM validation

The project now supports the public **Measured Radio Channel Dataset for OFDM Receiver Validation** from `rikluost/sdr_ofdm_dataset_v1`.

The public dataset contains over-the-air SDR measurements with:
- 435 MHz center frequency
- OFDM, FFT size 128
- 14 OFDM symbols per TTI
- 16-QAM payload modulation
- approximately 1000 TTIs
- received complex IQ samples
- transmitted 4-bit labels for 1400 payload resource elements per TTI
- measured SINR

AdaRx converts each 4-bit payload label to one of 16 classes and uses the measured IQ samples as receiver input.

### Evaluation design

The real-data experiment is chronological:

1. The first TTIs form the source-training segment.
2. The remaining TTIs are held out for streaming evaluation.
3. **Static** keeps the source model fixed.
4. **Pseudo** performs confidence-filtered online self-training with anchor regularization.
5. Held-out transmitted labels are used only for BER/SER/BLER evaluation, not for online adaptation.

The dataset pilots are not used as 16-QAM class labels because the published pilot symbols are QPSK-like reference tones and the provided payload labels exclude pilots. This avoids inventing invalid pilot targets.

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

This downloads `ofdm_valset_4_8_sdr.pth` (about 27 MB) from the public dataset repository into `data/`.

## Quick real-data validation

```powershell
.\.venv\Scripts\python.exe run_real_sdr_validation.py --train-ttis 50 --eval-ttis 100
```

## Larger validation

```powershell
.\.venv\Scripts\python.exe run_real_sdr_validation.py --train-ttis 100 --eval-ttis 300
```

Outputs:

```text
results_real_sdr/frame_metrics.csv
results_real_sdr/summary.json
```

Metrics:
- true bit error rate (BER) from the 4 transmitted bits per symbol
- 16-QAM symbol error rate (SER)
- block error rate (BLER)
- normalized throughput proxy
- mean measured SINR
- online update time
- number of pseudo-label update samples
- model weight drift

Create a plot:

```powershell
.\.venv\Scripts\python.exe dashboard.py --csv results_real_sdr/frame_metrics.csv --save results_real_sdr/summary.png
```

## Original synthetic experiment

The original reproducible QPSK-like synthetic experiment is still available:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --frames 120 --output results
```

It compares static, pilot-based, and hybrid adaptation under controlled synthetic SNR/phase/gain/IQ shifts.

## Field CSV adapter

`adarx/data.py` also accepts a generic CSV with `frame_id`, `rx_i`, `rx_q`, and optional `tx_class`, `is_pilot`.

## Research limitations

- This is a link-level research prototype, not a standards-compliant 5G NR receiver.
- The measured dataset uses 435 MHz SDR OFDM and 16-QAM; results must not be presented as Ericsson or 5G field-trial results.
- The small NumPy MLP is deliberately lightweight.
- Confidence-based pseudo-label adaptation can reinforce errors, so results should be compared against the static baseline.
- The held-out stream is chronological rather than a randomized IID test split.

## CV-safe description

**AdaRx — Online Self-Supervised Adaptation for AI Receivers**
- Extended a software-only neural receiver with a real measured SDR-OFDM validation pipeline using public over-the-air IQ data, 16-QAM transmitted labels, and measured SINR.
- Evaluated static versus confidence-filtered online pseudo-label adaptation on a chronological held-out stream using BER, SER, BLER, throughput, update cost, and model-drift metrics.
- Preserved a reproducible synthetic domain-shift experiment for controlled ablations while clearly separating synthetic and measured-data results.
