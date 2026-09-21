"""Crash a run mid-training, resume it, and check the LR schedule is unchanged.

    python tests/test_resume.py

Runs on CPU with 2 batches per epoch, a few minutes in total. Needs CIFAR-10 in
data/ (torchvision downloads it on first use).

The reference is an uninterrupted 4-epoch run. The interrupted run dies during
epoch 2's evaluation, so last.pt holds epoch 1, exactly as after a real crash.
It is then resumed three ways, and in each case the per-epoch LRs must match the
reference:

  intact     last.pt exactly as the crash left it
  legacy     optimizer state removed, as in checkpoints from before resume existed
  stale_lr   saved optimizer LR overwritten with a wrong value, which is what an
             earlier buggy resume left behind
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 4
RUN = "teacher_fp32_s0"
ARGS = ["--config", "configs/teacher_fp32.yaml", "--seed", "0", "--set",
        f"schedule.epochs={EPOCHS}", "debug.limit_train_batches=2",
        "debug.limit_val_batches=2", "data.num_workers=0"]

# Runs train.main() but raises on the 3rd evaluation (epoch 2), after epochs 0 and 1
# have been trained and checkpointed.
CRASH = textwrap.dedent("""
    import sys
    import src.train as T
    real, calls = T.evaluate_classifier, {"n": 0}
    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated crash")
        return real(*a, **k)
    T.evaluate_classifier = dying
    sys.argv = ["train"] + __ARGS__
    T.main()
""")


def train(results_dir: Path, crash: bool = False) -> int:
    args = ARGS + [f"results_dir={results_dir}"]
    cmd = ([sys.executable, "-c", CRASH.replace("__ARGS__", repr(args))] if crash
           else [sys.executable, "-m", "src.train", *args])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode


def logged_lrs(results_dir: Path) -> list[float]:
    with open(results_dir / RUN / "metrics.csv", newline="") as fh:
        return [float(r["lr"]) for r in csv.DictReader(fh)
                if r["split"] == "train" and r["lr"]]


def edit_last(results_dir: Path, variant: str) -> None:
    path = results_dir / RUN / "checkpoints" / "last.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if variant == "legacy":
        ckpt.pop("optimizer_state", None)
        ckpt.pop("scaler_state", None)
    elif variant == "stale_lr":
        ckpt["optimizer_state"]["param_groups"][0]["lr"] = 0.0825
    torch.save(ckpt, path)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="resume_test_"))
    try:
        assert train(tmp / "ref") == 0, "reference run failed"
        want = logged_lrs(tmp / "ref")

        assert train(tmp / "crashed", crash=True) != 0, "crash harness did not crash"
        assert len(logged_lrs(tmp / "crashed")) == 2, "expected 2 epochs before the crash"

        failures = []
        for variant in ("intact", "legacy", "stale_lr"):
            d = tmp / variant
            shutil.copytree(tmp / "crashed", d)
            edit_last(d, variant)
            assert train(d) == 0, f"{variant}: resume run failed"
            got = logged_lrs(d)
            ok = len(got) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(got, want))
            print(f"{'PASS' if ok else 'FAIL'}  {variant:9s} lrs={[round(x, 5) for x in got]}")
            if not ok:
                failures.append(variant)
        print(f"reference   lrs={[round(x, 5) for x in want]}")
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
