<div align="center">

<img width="1919" height="1079" alt="STT-GMM banner" src="https://github.com/user-attachments/assets/030b6822-0d94-427b-a81d-f20e8729741f" />

# 🌍 STT-GMM
### Spectral-Token Transformer Ground Motion Model

*Attention-based response spectrum prediction, trained on NGA-West2*

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Data](https://img.shields.io/badge/dataset-NGA--West2-4B8BBE?style=flat-square)
![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20release-brightgreen?style=flat-square)

</div>

---

## 📌 Overview

**STT-GMM** predicts earthquake response spectra by treating the spectrum as a **structured sequence over spectral periods** and modeling cross-period dependencies with a transformer — rather than fitting each period independently, as classical ground motion models (GMMs) typically do. The architecture stays consistent with established GMM principles (magnitude scaling, distance attenuation, site effects) while learning richer period-to-period correlations directly from data.

> ⚠️ **Normalization matters.** Inputs must be normalized with the exact training-time statistics in `input_normalization_stats.csv` before inference — skipping this produces physically inconsistent predictions (broken magnitude scaling, distance attenuation, or site effects).

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

## ⚙️ How It Works

**1. Inputs** — Moment magnitude, hypocentral depth, R<sub>JB</sub>, V<sub>s30</sub>, fault type, motion direction, and (optionally) event ID.

**2. Spectral tokenization** — The response spectrum is represented as a sequence of tokens, one per period, letting the model attend across the full spectrum instead of treating each period in isolation.

**3. Prediction modes**

<div align="center">

| Mode | When it applies | Behavior |
|---|---|---|
| 🎯 Event-conditioned | Event ID was seen in training | Uses learned between-event term for a sharper, event-specific spectrum |
| 📈 Median (unseen events) | New / unseen earthquake | Falls back to a classical GMM-style median prediction |

</div>

---

## 🚀 Typical Workflow

```
model_training.ipynb  →  prelim_analysis.ipynb  →  llm_exec.ipynb
     (train)                 (validate)              (predict)
```

1. **Train** (optional) — run `model_training.ipynb` to reproduce the model from scratch on NGA-West2.
2. **Validate** — run `prelim_analysis.ipynb` to check per-period R² and cross-validation performance before trusting the model.
3. **Predict** — run `llm_exec.ipynb`, supply seismic parameters, and get a predicted response spectrum (SA vs. period) plus plots.

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

## 📚 Citation

If you use this repository in academic work, please cite the accompanying paper describing the Spectral-Token Transformer Ground Motion Model.

## 📬 Contact

For questions, extensions, or collaboration, please refer to the corresponding author information in the associated publication.

---

<div align="center">

**Created May 2026**

*@author: Pavan Mohan Neelamraju*
*Affiliation: Indian Institute of Technology Madras*

[📧 npavanmohan3@gmail.com](mailto:npavanmohan3@gmail.com) · [🌐 pavanmohan.netlify.app](https://pavanmohan.netlify.app/)

</div>
