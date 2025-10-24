# Privacy-Preserving Federated Anomaly Detection Using Differentially Private Deep Models

**Short description**: End-to-end federated learning framework for anomaly detection with **differential privacy** (per-sample gradient clipping + Gaussian noise) on each client. Includes a lightweight simulator, DP-SGD, federated averaging, baselines (centralized / non-DP / DP), synthetic data generator with controllable anomaly rate, reproducible configs, CI, and evaluation (AUC/PR, F1, TPR@FPR).

<p align="center"><em>Train local autoencoders under DP, aggregate securely, and evaluate robustness and privacy-utility trade-offs.</em></p>

---

## Features
- **Federated Simulator** with N clients, partial participation, stragglers, and heterogeneous data splits.
- **Differential Privacy (Local)**: Per-sample gradient clipping and Gaussian noise (Rényi accountant estimates provided).
- **Models**: MLP and 1D CNN autoencoder backbones for tabular / time-series anomalies.
- **Baselines**: Centralized (no FL), Federated non-DP, Federated DP-SGD.
- **Metrics**: AUROC, AUPRC, F1, TPR@FPR, reconstruction error histograms.
- **Reproducible**: YAML configs, deterministic seeds, saved run metadata.
- **CI Ready**: pytest, flake8, mypy stubs.
- **Containers**: Dockerfile and optional Helm chart for a simple REST evaluator service.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a DP-FedAvg experiment
python src/train_fed.py --config configs/fed_dp_small.yml

# Evaluate a saved run
python src/eval.py --run_dir runs/fed_dp_*
```

## Repo Structure
```
.
├── configs/               # Experiment configs
├── docs/                  # Design docs & notes
├── src/                   # Core library & scripts
├── tests/                 # Unit tests
├── .github/workflows/     # CI
├── helm/                  # Optional demo service
├── scripts/               # Helper scripts
└── runs/                  # Output runs
```

## Citation
See [`CITATION.cff`](CITATION.cff).
