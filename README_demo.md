# DiNof Time-Series — Supervisor Demo Guide

**Project:** DiNof for Time-Series Anomaly Detection  
**Title:** Extending Flow-Guided Diffusion Priors Beyond Images  
**Dataset:** Synthetic SWaT (51 water-treatment sensors)  
**Base Paper:** Zand et al. 2024 — "Diffusion Models with Deterministic Normalizing Flow Priors"

---

## ▶️ How to Run the Demo (3 commands)

Open a terminal in this folder:
```
cd C:\Users\garga\.gemini\antigravity\scratch\DiNof
```

### Option A — Full end-to-end demo (5 mins)
```bash
python demo_run.py
```
This will:
1. ✅ Generate synthetic SWaT sensor data (51 features, 8 attack types)
2. ✅ Train DiNof model for 500 steps
3. ✅ Evaluate anomaly detection (F1, AUROC, Precision, Recall)
4. ✅ Save 4 plots to `runs/swat_demo/`
5. ✅ Print final metrics

### Option B — Show pre-trained results (30 seconds)
```bash
python demo_run.py --no_train --workdir runs/final
```
Loads the already-trained checkpoint and shows metrics + plots instantly.

### Option C — CLI training (full control)
```bash
python main_ts.py --mode train --workdir runs/my_run --steps 1000
python main_ts.py --mode eval  --workdir runs/my_run
```

---

## 📁 Project Structure

```
DiNof/
├── demo_run.py              ← START HERE for supervisor demo
├── live_train.py            ← Full training with live progress bar
├── main_ts.py               ← CLI entry point
├── runner_ts.py             ← Training + evaluation engine
├── losses_ts.py             ← Joint DiNof loss (score + flow)
├── datasets_ts.py           ← SWaT sliding-window dataset
├── visualize_results.py     ← Plots generator
│
├── models/
│   ├── score_model_1d.py    ← 1D U-Net score model
│   └── flow_1d.py           ← Feature-wise RealNVP1D flow
│
├── configs/ts/
│   └── swat_vpsde.py        ← Config: Tm=0.6, VPSDE, 51 channels
│
├── data/
│   └── synthetic_swat.py    ← Synthetic SWaT data generator
│
└── runs/
    ├── final/               ← Best training run
    │   ├── final_results.txt
    │   ├── loss_curves.png
    │   ├── roc_curve.png
    │   ├── anomaly_scores.png
    │   └── ts_with_anomalies.png
    └── swat_demo/           ← Quick demo run
```

---

## 🔑 Key Architecture Points (for supervisor Q&A)

### What is DiNof?
DiNof = **Di**ffusion + **No**rmalizing **f**low.  
Instead of pure Gaussian noise as the diffusion prior, it uses a Normalizing Flow trained at intermediate timestep **Tm = 0.6T**.

### What did we change from the base paper?
| Aspect | Base Paper | Our Extension |
|--------|-----------|---------------|
| Domain | CIFAR-10 images | SWaT time-series |
| Score model | 2D NCSNpp | 1D U-Net |
| Flow model | 2D Glow | Feature-wise RealNVP1D |
| Task | Image generation (FID) | Anomaly detection (F1, AUROC) |
| Inference | Reverse diffusion | **NF forward pass only** ✅ |

### How does anomaly detection work?
```
Test window x  →  perturb to Tm  →  Flow forward pass  →  -log p(x_Tm)  →  Anomaly Score
                                     (single pass, ~ms)
```
Higher score = more anomalous.  
Threshold at 90th percentile of training scores.

### Why is this real-time capable? (Panel concern)
- NF forward pass = 8 MLP layers on 51-dim vector
- **No iterative reverse diffusion** at test time
- Sub-millisecond per 64-sample window on CPU

---

## 📊 Results

See `runs/final/final_results.txt` for complete metrics.

Plots in `runs/final/`:
- **loss_curves.png** — Score loss + Flow loss vs training step
- **roc_curve.png** — ROC curve with AUROC score
- **anomaly_scores.png** — Normal vs Anomaly score distributions
- **ts_with_anomalies.png** — Sensor signal with anomaly regions highlighted

---

## ⚙️ Requirements

```bash
pip install torch torch-ema scikit-learn tqdm ml-collections matplotlib
```

Python 3.9+ required. No GPU needed (CPU demo works fine).

---

*Group project — Final Year, Nov 2026*
