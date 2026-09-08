# When Distilled Students Outperform Their Teachers

Pilot code for a study of knowledge distillation through the lens of loss
landscape geometry.

The question: when a distilled student generalizes better than its teacher on
corrupted data, is that because the student has landed in a *flatter* minimum,
rather than just because of a difference in capacity? And does the effect survive
a change in training precision (fp32 vs mixed precision)?

This repo is the pilot only. It trains one teacher (ResNet-18 on CIFAR-10),
distils students at two widths, evaluates everything on CIFAR-10 and CIFAR-10-C,
and measures sharpness and Hessian curvature at each solution. Feature
distillation, more widths, and CIFAR-100 are out of scope but the code is
organized so they slot in without rewrites.

For each trained model the pipeline records: ID accuracy (CIFAR-10 test), OOD
accuracy (mean over CIFAR-10-C, 15 corruptions x 5 severities), adaptive
sharpness, Hessian trace, top Hessian eigenvalue, plus the training precision,
width, and seed. `notebooks/analysis.ipynb` then checks whether "student beats
teacher on OOD" lines up with "student sits in a flatter minimum".


## Layout

```
configs/                 one YAML per run (teacher / student x width x precision)
data/download_cifar10c.py  fetch and unpack the CIFAR-10-C archive
src/
  models/resnet.py       CIFAR ResNet-18 with a width multiplier
  distillation/kd.py      logit KD loss
  geometry/
    bn_utils.py           eval + frozen BN + fp32 setup
    sharpness.py          adaptive m-sharpness
    hessian.py            trace and top eigenvalue via PyHessian
  data.py                 CIFAR-10 and CIFAR-10-C loaders
  config.py               dataclass config schema + YAML loader
  utils.py                seeding, environment capture, CSV/JSON logging
  engine.py               train and eval loops
  checkpoint.py           save/load, rebuild a model from a checkpoint
  train.py                train a teacher or distil a student
  evaluate.py             ID + OOD evaluation, aggregation
  measure_geometry.py     sharpness + Hessian
scripts/
  run_pilot.sh / .ps1     full pilot end to end
  phase1.sh               reduced fp32-only run for a first look
  smoke.sh / .ps1         quick end-to-end check
notebooks/
  analysis.ipynb          flatness vs OOD, fp32 vs AMP
  colab_pilot.ipynb       run the pilot on a Colab GPU
```


## How the measurements are defined

**Sharpness (headline metric).** Adaptive m-sharpness, i.e. the ASAM operator of
Kwon et al. (2021) with the m-sharpness estimator of Foret et al. (2021). For
micro-batches B of size m from the geometry subset,

    S_rho(w) = mean_B [ max over ||inv(T_w) eps|| <= rho of L_B(w + eps) - L_B(w) ]

The inner max is one normalized ascent step: eps* = rho * T_w^2 g / ||T_w g||
with g = grad L_B(w). T_w = |w| + 0.01 for weight tensors and 1 for biases and
BatchNorm parameters, which makes the number invariant to node-wise rescaling
(needed when comparing different widths). rho is fixed at 0.05 and reported with
every result. Set `--no-adaptive` for plain m-sharpness (T_w = 1 everywhere).

**Hessian trace and top eigenvalue (secondary).** PyHessian: Hutchinson's
estimator for the trace, power iteration for the top eigenvalue.

**Common to both.** Plain cross-entropy (not the KD loss), so every model is
measured against the same objective. A fixed 2,000-image subset of the CIFAR-10
training set, the same images for every model. Everything in fp32 with BatchNorm
in eval mode and its running stats frozen (`prepare_model_for_geometry`); nothing
runs under autocast.

**Precision.** fp32 vs AMP is a training-time choice only. AMP training uses
autocast and a GradScaler in the training loop; the saved weights, all
evaluation, and all geometry are fp32.

**CIFAR-10-C.** Not in torchvision. `data/download_cifar10c.py` pulls the archive
from Zenodo (record 2535967), checks the MD5, and unpacks it to
`data/CIFAR-10-C/`. `src/data.py` builds a loader per (corruption, severity) with
normalization only.

**Logging.** Every run writes `results/<run>/metrics.csv` and `summary.json`, no
network needed. `--wandb` adds Weights & Biases on top; it is off by default.


## Pilot settings

| Role      | width_mult | params  |
|-----------|-----------:|--------:|
| teacher   | 1.0        | ~11.2 M |
| student A | 0.5        | ~2.8 M  |
| student B | 0.25       | ~0.7 M  |

SGD (momentum 0.9, weight decay 5e-4), batch size 128, LR 0.1 with cosine decay,
200 epochs, random crop + horizontal flip. Distillation uses the same optimizer
and schedule with T = 4 and alpha = 0.9 (weight on the soft term); the teacher is
frozen. Every model is trained in fp32 and in AMP, with seeds 0, 1, 2. That is
3 models x 2 precisions x 3 seeds = 18 training runs.

Geometry uses rho = 0.05, m = 128, one ascent step, a 2,000-image subset, and
100 Hutchinson iterations.

The OOD headline number is the mean top-1 accuracy over the 15 test corruptions
x 5 severities; the full grid and per-corruption breakdown are saved, and mCE
relative to the matching teacher is computed when a baseline is available.

**Phase 1.** `configs/teacher_fp32.yaml` and `configs/student_w0.5_fp32.yaml` are
currently set to 30 epochs. This is a reduced first pass (teacher + w0.5 student,
fp32, seeds 0 and 1) to see whether the flatness/OOD pattern shows up at all
before committing to the full run. Set `epochs` back to 200 in those two files
for the real pilot.


## Setup

```bash
git clone <your-remote> student-teacher-landscape
cd student-teacher-landscape
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1

# install the torch build for your CUDA version first (see pytorch.org)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python data/download_cifar10c.py --dest data/          # ~2.9 GB, one time
```

CIFAR-10 downloads itself on the first training run.


## Running

Quick check that everything works (a few minutes):

```bash
bash scripts/smoke.sh
```

Full pilot:

```bash
bash scripts/run_pilot.sh                 # SKIP_AMP=1 for a CPU-only machine
./scripts/run_pilot.ps1                    # Windows; -SkipAmp for CPU only
```

Or by hand. Train the teachers first (students load the matching teacher, same
precision and seed):

```bash
for p in fp32 amp; do for s in 0 1 2; do
  python -m src.train --config configs/teacher_$p.yaml --seed $s
done; done

for w in w0.5 w0.25; do for p in fp32 amp; do for s in 0 1 2; do
  python -m src.train --config configs/student_${w}_${p}.yaml --seed $s
done; done; done

python -m src.evaluate --all --results-dir results/
python -m src.measure_geometry --all --results-dir results/
python -m src.evaluate --aggregate --results-dir results/ --out results/pilot_summary.csv
```

Then open `notebooks/analysis.ipynb` on `results/pilot_summary.csv`.

No GPU handy? `notebooks/colab_pilot.ipynb` runs the phase-1 version on a free
Colab GPU in about 20 minutes.

Each run writes `results/<run_name>/` with `checkpoints/{best,last}.pt`,
`metrics.csv`, `summary.json`, `eval.json`, `eval_cifar10c.csv`, and
`geometry.json`. Run names encode the config, e.g. `teacher_fp32_s0`,
`student_w0.5_amp_s2`.


## Reproducibility

`set_seed` seeds Python, NumPy and torch, disables cuDNN autotuning, and requests
deterministic kernels (with `warn_only`, so ops without a deterministic
implementation fall back rather than raise). AMP training is slightly less
reproducible across hardware, which is part of what the precision comparison is
there to check. Package versions are pinned in `requirements.txt`; each run's
`summary.json` records the CUDA, driver, GPU and git commit it ran on.

PyHessian 0.1 prints a couple of deprecation warnings (`torch.autograd.Variable`,
`only_inputs=`); they are harmless. Its double-backward hits a few
non-deterministic cuDNN kernels on GPU, so the trace and eigenvalue are seeded
for repeatability rather than made bit-exact.


## References

- Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015.
- Foret, Kleiner, Mobahi, Neyshabur. Sharpness-Aware Minimization for Efficiently
  Improving Generalization. ICLR 2021.
- Kwon, Kim, Park, Choi. ASAM: Adaptive Sharpness-Aware Minimization. ICML 2021.
- Yao, Gholami, Keutzer, Mahoney. PyHessian: Neural Networks Through the Lens of
  the Hessian. 2020.
- Hendrycks, Dietterich. Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations. ICLR 2019.
