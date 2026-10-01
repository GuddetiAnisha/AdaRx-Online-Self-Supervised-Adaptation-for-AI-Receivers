# AdaRx Validation Results

## Final measured SDR-OFDM experiment

Dataset: `rikluost/sdr_ofdm_dataset_v1`

Configuration:
- Source training TTIs: 100
- Held-out evaluation TTIs: 300
- Seed: 7
- Mean measured SINR: 14.04 dB
- Pilot equalization: `H = Ypilot / Xpilot` with complex linear interpolation across 102 active bins
- Adaptation threshold: 0.85
- Online adaptation does not use held-out transmitted labels

| Receiver | BER | SER | BLER | Throughput proxy |
|---|---:|---:|---:|---:|
| Raw static | 0.4603 | 0.8578 | 1.0000 | 0.0000 |
| Equalized static | 0.0900 | 0.2413 | 0.6619 | 0.3381 |
| Equalized adapted | 0.0904 | 0.2422 | 0.6637 | 0.3363 |

## Main finding

Pilot-aided equalization produced the dominant improvement. On the 300-TTI held-out stream, BER decreased from about **46.0%** on raw IQ to about **9.0%** after equalization, while SER decreased from about **85.8%** to **24.1%**.

The confidence-filtered online centroid adaptation was active (mean 157.32 update samples per TTI, final centroid drift 0.1672) but did not outperform the equalized static receiver. This is reported as an ablation result rather than as an adaptation gain.

## Interpretation

These results support the use of pilot-aided channel estimation and equalization for this measured SDR-OFDM dataset. They do not establish standards-compliant 5G NR performance and should not be presented as vendor or field-trial results.
