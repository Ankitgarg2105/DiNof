# DiNof for Time-Series Anomaly Detection: Extending Flow-Guided Diffusion Priors Beyond Images

> **Final Year Project** | Extended from *"Diffusion Models with Deterministic Normalizing Flow Priors"* (Zand et al., TMLR 2024).

This repository adapts the **DiNof** framework (hybrid Score-Based Diffusion + Normalizing Flow) from 2D image synthesis (CIFAR-10 / CelebA) to **industrial cyber-physical time-series anomaly detection** on the Secure Water Treatment (**SWaT**) benchmark.

---

## 🚀 Quick Start (Zero-Setup Demo)

Clone the repository and install requirements:
```bash
git clone https://github.com/Ankitgarg2105/DiNof.git
cd DiNof
pip install torch torch-ema scikit-learn tqdm ml-collections matplotlib
```

### 1. Instant Evaluation (Pre-trained weights, ~5 seconds)
```bash
python demo_run.py --no_train --workdir runs/final
```
Loads the pre-trained checkpoint (`runs/final/checkpoints/checkpoint_latest.pth`), computes anomaly scores, and outputs AUROC, F1, and plots.

### 2. Live Interactive Case Study Demo
```bash
python case_study_demo.py
```
Tests 4 operational cases (Normal, Synthetic Cyber-Attack, Sensor Spike, Gradual Sensor Drift) with sub-millisecond per-window inference and explanatory breakdowns.

### 3. Continuous Real-Time Streaming Detection
```bash
python live_detect.py
```
Simulates continuous real-time ICS sensor telemetry with dynamic ASCII alert visualization.

### 4. Full End-to-End Training
```bash
python demo_run.py --steps 500
```
Trains DiNof (1D U-Net + RealNVP1D) from scratch, saves checkpoints, evaluates, and generates visualization curves.

---

## 📊 Benchmark Results

| Metric | Score-Based Denoising ($t=0.05$) | Flow NLL ($t=0.10$) |
|---|:---:|:---:|
| **AUROC** | **0.9035** | 0.8480 |
| **F1 Score** | **0.7826** | 0.6190 |
| **Precision** | **0.8438** | 0.5532 |
| **Recall** | **0.7297** | 0.7027 |
| **Inference Latency** | **$\sim$5.4 ms / window** | **$\sim$4.2 ms / window** |

---

## 🏗️ Architecture & Adaptations

1. **Score Model (`models/score_model_1d.py`):**
   - 1D U-Net operating on 4D tensors `[B, 51, 1, 64]` (51 sensors, window 64).
   - Sinusoidal time embedding + residual blocks + multi-head temporal self-attention.
   - SDE compatible broadcasting (`std[:, None, None, None]`).

2. **Normalizing Flow (`models/flow_1d.py`):**
   - Feature-wise RealNVP1D with 8 alternating masked affine coupling layers.
   - Operates on 51-dimensional sensor space with shared weights across temporal window.
   - Fast convergence ($\mathcal{L}_{\text{flow}} \approx 71.8$ nats, matching 51-dim theoretical Gaussian optimum).

3. **Inference Acceleration (Panel Feedback Response):**
   - Full reverse diffusion requires 600–1000 iterative steps.
   - Our test-time scoring uses **single-pass denoising score matching error** at $t=0.05$:
     $$\text{Score}(x) = \|s_\theta(x_t, t) \cdot \sigma_t + \epsilon\|^2$$
   - Executes in **$<10$ms on standard CPU**, enabling real-time industrial deployment.

---

## 📂 Repository Structure

```
DiNof/
├── case_study_demo.py        # 4 interactive case studies for presentations
├── live_detect.py            # Real-time streaming sensor detection demo
├── demo_run.py               # Zero-argument benchmark & training runner
├── runner_ts.py              # Time-series training and evaluation engine
├── losses_ts.py              # Joint DiNof loss (Score DSM + Flow NLL)
├── datasets_ts.py            # Sliding window loader [B, 51, 1, 64]
├── visualize_results.py      # Publication-quality loss/ROC/telemetry plots
├── configs/ts/swat_vpsde.py  # SWaT VPSDE configuration (Tm=0.6)
├── models/
│   ├── score_model_1d.py     # 1D U-Net score network (447K params)
│   └── flow_1d.py            # Feature-wise RealNVP1D flow (291K params)
├── data/
│   ├── synthetic_swat.py     # 51-sensor cyber-physical simulator
│   └── swat_synthetic/       # Benchmark dataset & normalization files
└── runs/final/               # Pre-trained checkpoints & evaluation figures
```

---

## 📖 Citation (Base Paper)

```bibtex
@article{zand2024diffusion,
  title={Diffusion Models with Deterministic Normalizing Flow Priors},
  author={Mohsen Zand and Ali Etemad and Michael Greenspan},
  journal={Transactions on Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=ACMNVwcR6v}
}
```
