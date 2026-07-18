<div align="center">

<img src="assets/banner.svg" alt="STT-GMM banner" width="100%"/>

# 🌍 STT-GMM
### Spectral-Token Transformer Ground Motion Model

*Attention-based response spectrum prediction, trained end-to-end on NGA-West2*

![PyTorch](https://img.shields.io/badge/PyTorch-1E293B?style=for-the-badge&logo=pytorch&logoColor=22D3EE)
![Python](https://img.shields.io/badge/Python-1E293B?style=for-the-badge&logo=python&logoColor=FACC15)
![Dataset](https://img.shields.io/badge/Dataset-NGA--West2-1E293B?style=for-the-badge&labelColor=134E4A)
![Domain](https://img.shields.io/badge/Domain-Seismic%20Engineering-1E293B?style=for-the-badge&labelColor=7C2D12)
![Status](https://img.shields.io/badge/Status-Research%20Release-1E293B?style=for-the-badge&labelColor=065F46)
![License](https://img.shields.io/badge/License-See%20LICENSE-1E293B?style=for-the-badge&labelColor=334155)

</div>

<br>

## 📖 Table of Contents

- [Overview](#-overview)
- [Why Spectral Tokens?](#-why-spectral-tokens)
- [Repository Structure](#️-repository-structure)
- [Architecture at a Glance](#️-architecture-at-a-glance)
- [How It Works](#️-how-it-works)
- [Getting Started](#-getting-started)
- [Typical Workflow](#-typical-workflow)
- [Example Usage](#-example-usage)
- [What You Get](#-what-you-get)
- [Notes & Best Practices](#-notes--best-practices)
- [Limitations](#️-limitations)
- [Roadmap](#️-roadmap)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 📌 Overview

**STT-GMM** predicts earthquake response spectra by treating the spectrum as a **structured sequence over spectral periods** and modeling cross-period dependencies with a transformer — rather than fitting each period independently, as classical ground motion models (GMMs) typically do.

The architecture stays grounded in established GMM principles — magnitude scaling, distance attenuation, site effects — while learning richer, data-driven period-to-period correlations that hand-specified functional forms tend to miss.

> ⚠️ **Normalization matters.** Inputs must be normalized with the exact training-time statistics in `input_normalization_stats.csv` before inference. Skipping this produces physically inconsistent predictions — broken magnitude scaling, distorted distance attenuation, or incorrect site effects.

---

## 🧭 Why Spectral Tokens?

Traditional GMMs regress each spectral period (T = 0.01s, 0.02s, ... 4.0s) more or less independently, then smooth the result. This throws away information: real response spectra have strong period-to-period correlation driven by the same underlying source, path, and site physics.

STT-GMM instead represents the full spectrum as a **sequence of period tokens** and lets self-attention learn which periods should inform each other — the model discovers these correlations directly from the NGA-West2 database instead of having them imposed by a fixed smoothing rule.

---

## 🗂️ Repository Structure

<div align="center">

| File | Role |
|---|---|
| 📊 `input_normalization_stats.csv` | Training-time mean/std for continuous inputs (Mw, depth, R<sub>JB</sub>, log R<sub>JB</sub>, log V<sub>s30</sub>) |
| 🧠 `stt_gmm_weights.pt` | Full trained bundle — model weights, context/event encoders, config, output normalization, spectral periods, event ID map |
| ⚡ `llm_exec.ipynb` | **Inference entry point** — load the trained model and generate predicted spectra |
| 🏋️ `model_training.ipynb` | Full training pipeline — preprocessing → tokenization → training loop → checkpointing |
| ✅ `prelim_analysis.ipynb` | Diagnostics — per-period R², k-fold CV, overfitting/leakage checks |

</div>

---

## 🏗️ Architecture at a Glance

```
                ┌─────────────────────────┐
  Seismic       │   Context Encoder        │
  parameters ──▶│  Mw · depth · Rjb · Vs30  │──┐
  (Mw, depth,   │  fault type · direction   │  │
   Rjb, Vs30,   └─────────────────────────┘  │
   fault, dir)                                ▼
                ┌─────────────────────────────────────┐
  Event ID ────▶│         Event Embedding               │──┐
  (optional)    └─────────────────────────────────────┘  │
                                                            ▼
                ┌───────────────────────────────────────────────┐
                │      Spectral-Token Transformer (self-attn)     │
                │   one token per spectral period, T = 0.01–4.0s  │
                └───────────────────────────────────────────────┘
                                        │
                                        ▼
                        Predicted log response spectrum
                              (denormalized → SA vs. T)
```

---

## ⚙️ How It Works

**1. Inputs** — Moment magnitude, hypocentral depth, R<sub>JB</sub>, V<sub>s30</sub>, fault type, motion direction, and (optionally) event ID.

**2. Spectral tokenization** — The response spectrum is represented as a sequence of tokens, one per period, letting the model attend across the full spectrum instead of treating each period in isolation.

**3. Prediction modes**

<div align="center">

| Mode | When it applies | Behavior |
|---|---|---|
| 🎯 Event-conditioned | Event ID was seen in training | Uses the learned between-event term for a sharper, event-specific spectrum |
| 📈 Median (unseen events) | New / unseen earthquake | Falls back to a classical GMM-style median prediction |

</div>

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/stt-gmm.git
cd stt-gmm
pip install torch numpy pandas scikit-learn matplotlib
jupyter notebook
```

Open `llm_exec.ipynb` first if you just want predictions from the trained model — no training required.

---

## 🔁 Typical Workflow

```
model_training.ipynb  →  prelim_analysis.ipynb  →  llm_exec.ipynb
     (train)                 (validate)              (predict)
```

1. **Train** (optional) — run `model_training.ipynb` to reproduce the model from scratch on NGA-West2.
2. **Validate** — run `prelim_analysis.ipynb` to check per-period R² and cross-validation performance before trusting the model.
3. **Predict** — run `llm_exec.ipynb`, supply seismic parameters, and get a predicted response spectrum (SA vs. period) plus plots.

---

## 💻 Example Usage

```python
import torch

# Load the trained bundle
bundle = torch.load("stt_gmm_weights.pt", map_location="cpu", weights_only=True)

# Reconstruct model + context encoder from bundle["model_config"]
# (see llm_exec.ipynb for the full reconstruction snippet)

# Normalize a new input using input_normalization_stats.csv, then:
# log_sa = model(context_encoder(mw, depth, rjb, vs30, fault_type, direction))
# sa = np.exp(log_sa)   # convert back to physical spectral acceleration
```

For the complete, runnable version — including loading normalization stats, rebuilding the architecture, and plotting SA vs. period — see `llm_exec.ipynb`.

---

## 📈 What You Get

- Predicted **log response spectrum** and physical **SA vs. period** plots
- Reproducible inference — no retraining needed, everything ships in `stt_gmm_weights.pt`
- Sanity-checked outputs via k-fold validation and per-period R² in `prelim_analysis.ipynb`

---

## 📝 Notes & Best Practices

- Always normalize inputs with the provided `input_normalization_stats.csv` — never re-derive stats at inference time.
- Event-conditioned predictions are only valid for events **seen during training**.
- For unseen earthquakes, the model reverts to **median** predictions, consistent with standard GMM usage.
- Built for **engineering / seismic hazard applications**, not record-level waveform reconstruction.

---

## ⚠️ Limitations

- Predictions are only as reliable as the region and magnitude/distance ranges represented in NGA-West2 — extrapolating far outside the training distribution is not recommended.
- The model predicts **median** ground motion for unseen events; it does not currently produce full aleatory variability (sigma) estimates.
- Not intended for single-record waveform synthesis — this is a response-spectrum-level model.

---

## 🗺️ Roadmap

- [ ] Add uncertainty/sigma estimation alongside median predictions
- [ ] Extend to additional strong-motion databases beyond NGA-West2
- [ ] Package a lightweight inference-only CLI (no notebook required)

---

## 📚 Citation

If you use this repository in academic work, please cite the accompanying paper describing the Spectral-Token Transformer Ground Motion Model.

---

## 📬 Contact

For questions, extensions, or collaboration inquiries, please refer to the corresponding author information in the associated publication.
