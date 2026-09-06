# When Distilled Students Outperform Their Teachers
### An Analysis Through Loss Landscape Geometry — *Pilot code (Sections 5.1 + 5.2)*

This repository contains the **pilot** experiments for the paper. It is deliberately
scoped: a single teacher, two student widths, two training precisions, ID + OOD
evaluation, and loss-landscape geometry measurement. The full experimental grid
(feature distillation, more widths, CIFAR-100, …) is **not** implemented here, but
the code is structured so those extensions drop in cleanly.

---

## 1. Research question

> Does knowledge distillation steer a smaller **student** toward **flatter minima**
> than its **teacher**, and does that flatness — not merely model capacity — help
> explain cases where the student generalizes better **out of distribution**?
>
> Sub-question: is the effect **robust to numerical precision** (fp32 vs. AMP) at
> training time?

The pilot produces, for each trained model, a row containing:

| Field | Source |
|---|---|
| ID accuracy (CIFAR-10 test) | `src/evaluate.py` |
| OOD accuracy (CIFAR-10-C, 15 corruptions × 5 severities) | `src/evaluate.py` |
| Adaptive sharpness `S_ρ` (ρ reported) | `src/measure_geometry.py` |
| Hessian trace | `src/measure_geometry.py` (PyHessian) |
| Top Hessian eigenvalue `λ_max` | `src/measure_geometry.py` (PyHessian) |
| Training precision (`fp32` / `amp`) | config |
| Width multiplier | config |
| Seed | config |

Analysis notebooks in `notebooks/` then test whether *"student beats teacher on OOD"*
correlates with *"student sits in a flatter minimum"*.

---

## 2. Repository layout

```
student-teacher-landscape/
├── configs/                 # one YAML per run (teacher / students × fp32 / amp)
├── data/
│   └── download_cifar10c.py # downloads + unpacks the official Zenodo tarball
├── src/
│   ├── models/
│   │   └── resnet.py        # CIFAR-adapted ResNet-18 with width multipliers
│   ├── distillation/
│   │   └── kd.py            # response-based KD loss (Hinton et al., 2015)
│   ├── geometry/
│   │   ├── bn_utils.py      # eval mode + frozen BN + fp32 enforcement
│   │   ├── sharpness.py     # adaptive (m-)sharpness
│   │   └── hessian.py       # Hessian trace + top eigenvalue via PyHessian
│   ├── data.py              # CIFAR-10 (ID) and CIFAR-10-C (OOD) loaders
│   ├── config.py            # typed dataclass config schema + YAML loader
│   ├── utils.py             # seeding, env capture, CSV/JSON logging
│   ├── engine.py            # shared train-epoch / eval loops, optim + sched
│   ├── checkpoint.py        # checkpoint save/load (+ rebuild model from ckpt)
│   ├── train.py             # train teacher OR distill student (one entry point)
│   ├── evaluate.py          # ID + OOD evaluation
│   └── measure_geometry.py  # sharpness + Hessian measurement
├── scripts/                 # thin shell/python wrappers to run the whole pilot
├── results/                 # run outputs: checkpoints + metrics.csv + summary.json
├── notebooks/               # analysis: flatness vs. OOD-gain
├── requirements.txt
└── README.md
```

> **Implementation status (built stage-by-stage):**
> - [x] Stage 1 — models (`models/resnet.py`), data loaders (`data.py`),
>   config system (`config.py`), seeding + logging (`utils.py`)
> - [x] Stage 2 — KD loss (`distillation/kd.py`), shared train/eval loops
>   (`engine.py`), checkpointing (`checkpoint.py`), training + distillation
>   entry point (`train.py`)
> - [x] Stage 3 — CIFAR-10-C downloader (`data/download_cifar10c.py`) +
>   ID/OOD evaluation with mCE + aggregation (`evaluate.py`)
> - [ ] Stage 4 — geometry module (`geometry/*`, `measure_geometry.py`)
> - [ ] Stage 5 — pilot configs (`configs/*.yaml`) + run scripts (`scripts/`)
>
> Modules for not-yet-built stages carry documented placeholders
> (docstring + `NotImplementedError`).

---

## 3. Key design decisions (locked for the pilot)

These are fixed so results are unambiguous and reproducible. Rationale is given
because it matters for correctness.

### 3.1 Geometry metrics

* **Adaptive sharpness — the headline metric.** We report worst-case
  **adaptive** sharpness following **Kwon et al. (2021), ASAM**, estimated with
  the **m-sharpness** estimator of **Foret et al. (2021), SAM**:

  ```
  S_ρ(w) = E_B [ max_{|| T_w^{-1} ε ||₂ ≤ ρ}  L_B(w + ε) − L_B(w) ]
  ```

  * `T_w = diag(|w|)` is the elementwise adaptive rescaling (with `+1` on
    normalization-layer and bias parameters), making the metric invariant to
    parameter re-scaling — important when comparing models of different width.
  * The inner maximization is approximated by a **single normalized
    gradient-ascent step** per micro-batch (the standard SAM ascent step).
  * `ρ` is **fixed** (`0.05`) and **reported with every number**; `m = 128`.
  * A non-adaptive m-sharpness (`T_w = I`) is available behind a flag for
    comparison but is not the reported number.

* **Hessian trace — the secondary metric.** Computed with **PyHessian**
  (Hutchinson estimator). The **top Hessian eigenvalue** `λ_max` (PyHessian
  power iteration) is also computed and stored, but adaptive sharpness + Hessian
  trace are the two numbers plotted and discussed in Section 5.2.

* **Loss used for geometry:** plain **cross-entropy** (never the KD loss), so
  teacher and students are measured against the same objective.

* **Data used for geometry:** a **fixed, seeded 2,000-example subset** of the
  CIFAR-10 *training* set, identical for every checkpoint, so curvature
  comparisons are fair.

### 3.2 Precision

* Precision (`fp32` vs. `amp`) is a **training-time variable only**.
* `amp` runs use `torch.cuda.amp` autocast + `GradScaler` **in the training loop
  only**. Checkpoints always store **fp32** weights.
* **All geometry measurements run in full fp32.** Geometry code is **never**
  wrapped in `autocast`, even for AMP-trained checkpoints — the checkpoint is
  cast to fp32 first.
* Evaluation (ID + OOD) also always runs in fp32.

### 3.3 BatchNorm during geometry

Before **any** sharpness or Hessian computation, `prepare_model_for_geometry()`:

1. sets `model.eval()` — BN uses running statistics, not batch statistics;
2. freezes BN running stats so no forward pass can update them;
3. casts the model to fp32.

This is mandatory: measuring curvature while BN statistics drift produces
contaminated estimates.

### 3.4 CIFAR-10-C

* Not available in torchvision. `data/download_cifar10c.py` downloads the
  official **Zenodo** archive (`CIFAR-10-C.tar`, record `2535967`), verifies its
  MD5, and unpacks it to `data/CIFAR-10-C/*.npy`.
* `src/data.py` builds the OOD loader on top: 15 corruptions × 5 severity levels,
  labels shared with the CIFAR-10 test set, normalization only (no augmentation).

### 3.5 Logging

* **Primary logging is offline:** every run writes `results/<run>/metrics.csv`
  and `results/<run>/summary.json`.
* **wandb is optional**, behind `--wandb` (config `wandb.enabled`, default
  `false`). Nothing touches the network unless you opt in.

---

## 4. Pilot configuration (locked)

> These live in `configs/` as YAML. All values below are confirmed for the pilot.

**Models**

| Role | Architecture | Width mult | ~Params |
|---|---|---|---|
| Teacher | ResNet-18 (CIFAR) | 1.0 | ~11.2 M |
| Student A | ResNet-18 (CIFAR) | 0.5 | ~2.8 M |
| Student B | ResNet-18 (CIFAR) | 0.25 | ~0.7 M |

**Teacher training:** SGD, momentum 0.9, weight decay 5e-4, batch size 128,
LR 0.1 with cosine decay, **200 epochs**, standard CIFAR augmentation
(random crop w/ padding 4, horizontal flip).

**Distillation:** same optimizer/schedule as the teacher (200 epochs);
temperature `T = 4`, `alpha = 0.9` (KD weight), teacher frozen
(`eval()` + `no_grad()`). Each student is distilled from the teacher of the
**matching precision and seed**.

**Precision:** every model trained twice — once `fp32`, once `amp`.

**Seeds:** `{0, 1, 2}`.

**Runs:** 3 models × 2 precisions × 3 seeds = **18 training runs** for the pilot.

---

## 5. Installation

```bash
# 1. Clone and enter the repo
git clone <your-remote-url> student-teacher-landscape
cd student-teacher-landscape

# 2. Create an environment (conda or venv both fine)
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# 3. Install PyTorch matching your CUDA version FIRST
#    (see https://pytorch.org/get-started/locally/ — the pin in
#     requirements.txt targets CUDA 12.1; change it if needed)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r requirements.txt
```

CIFAR-10 itself is downloaded automatically by torchvision on first use.

---

## 6. Download the OOD data (CIFAR-10-C)

```bash
python data/download_cifar10c.py --dest data/
```

This fetches `CIFAR-10-C.tar` (~2.9 GB) from Zenodo, checks its MD5, and unpacks
to `data/CIFAR-10-C/`. The script is idempotent — re-running it does nothing if
the data is already present.

---

## 7. Run the full pilot end-to-end

Every command writes to `results/<run_name>/`. Run names encode role, width,
precision, and seed, e.g. `teacher_fp32_s0`, `student_w0.5_amp_s2`.

```bash
# ============================================================
# STEP 0 — one-time setup
# ============================================================
python data/download_cifar10c.py --dest data/

# ============================================================
# STEP 1 — train the teachers (needed before any distillation)
#          6 runs: {fp32, amp} × seeds {0,1,2}
# ============================================================
python -m src.train --config configs/teacher_fp32.yaml --seed 0
python -m src.train --config configs/teacher_fp32.yaml --seed 1
python -m src.train --config configs/teacher_fp32.yaml --seed 2
python -m src.train --config configs/teacher_amp.yaml  --seed 0
python -m src.train --config configs/teacher_amp.yaml  --seed 1
python -m src.train --config configs/teacher_amp.yaml  --seed 2

# ============================================================
# STEP 2 — distill the students
#          Each student config points at the matching teacher
#          checkpoint (same precision + seed).
#          12 runs: {w0.5, w0.25} × {fp32, amp} × seeds {0,1,2}
# ============================================================
python -m src.train --config configs/student_w0.5_fp32.yaml  --seed 0
python -m src.train --config configs/student_w0.5_fp32.yaml  --seed 1
python -m src.train --config configs/student_w0.5_fp32.yaml  --seed 2
python -m src.train --config configs/student_w0.5_amp.yaml   --seed 0
python -m src.train --config configs/student_w0.5_amp.yaml   --seed 1
python -m src.train --config configs/student_w0.5_amp.yaml   --seed 2
python -m src.train --config configs/student_w0.25_fp32.yaml --seed 0
python -m src.train --config configs/student_w0.25_fp32.yaml --seed 1
python -m src.train --config configs/student_w0.25_fp32.yaml --seed 2
python -m src.train --config configs/student_w0.25_amp.yaml  --seed 0
python -m src.train --config configs/student_w0.25_amp.yaml  --seed 1
python -m src.train --config configs/student_w0.25_amp.yaml  --seed 2

# ============================================================
# STEP 3 — evaluate every checkpoint on ID + OOD
#          (loops over results/*/checkpoints/best.pt)
# ============================================================
python -m src.evaluate --all --results-dir results/

# ============================================================
# STEP 4 — measure loss-landscape geometry for every checkpoint
#          (fp32, BN frozen, no autocast)
# ============================================================
python -m src.measure_geometry --all --results-dir results/

# ============================================================
# STEP 5 — aggregate + analyze
# ============================================================
python -m src.evaluate --aggregate --results-dir results/ --out results/pilot_summary.csv
jupyter notebook notebooks/analysis.ipynb
```

`scripts/` contains wrappers (`scripts/run_pilot.sh`, `scripts/run_pilot.ps1`)
that execute Steps 1–5 in order.

---

## 8. Confirmed decisions

| Choice | Value | Status |
|---|---|---|
| Headline geometry metric | Adaptive sharpness (ASAM-style), `ρ = 0.05`, `m = 128`, 1 ascent step | confirmed |
| Secondary geometry metric | Hessian trace (PyHessian); `λ_max` also stored | confirmed |
| Geometry data subset | Fixed, seeded **2,000** CIFAR-10 train examples, identical for all models | confirmed |
| Training length | **200 epochs**, teacher and students, cosine LR | confirmed |
| Seeds | **3** per configuration → 18 training runs | confirmed |
| KD hyperparameters | `T = 4`, `alpha = 0.9`, single fixed setting (no sweep in the pilot) | default — say if you want a small `(T, alpha)` grid |
| OOD headline number | Mean top-1 accuracy over 15 corruptions × 5 severities; per-corruption/severity breakdown stored; mCE vs. fp32 teacher also computed | default — say if you want a different aggregation |

---

## 9. Reproducibility notes

* `src/utils.set_seed()` seeds Python, NumPy, and PyTorch, sets
  `cudnn.deterministic = True` / `cudnn.benchmark = False`, and enables
  `torch.use_deterministic_algorithms(True)` where it does not break the models.
* AMP training is inherently slightly less deterministic across hardware; this is
  expected and is part of what the precision-robustness check measures.
* Exact package versions are pinned in `requirements.txt`. Record your CUDA /
  driver / GPU in `results/<run>/summary.json` (the logger captures this
  automatically).

---

## 10. References

* Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015.
* Foret, Kleiner, Mobahi, Neyshabur. *Sharpness-Aware Minimization for Efficiently
  Improving Generalization.* ICLR 2021.
* Kwon, Kim, Park, Choi. *ASAM: Adaptive Sharpness-Aware Minimization.* ICML 2021.
* Yao, Gholami, Keutzer, Mahoney. *PyHessian: Neural Networks Through the Lens of
  the Hessian.* 2020.
* Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations.* ICLR 2019. (CIFAR-10-C)
