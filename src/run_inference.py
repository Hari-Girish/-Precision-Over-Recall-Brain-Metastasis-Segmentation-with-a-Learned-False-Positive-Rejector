"""Docker entrypoint for the BraTS-METS 2026 Task 1 submission.

Reproduces, case-by-case, the exact pipeline that produced the paper's scored
submission (9771640, ET 0.7187 / mean 0.6660):

  1. Stage each /input/<case>/ folder's 4 modality files into nnU-Net's expected
     flat <case>_0000..0003.nii.gz naming (channel order confirmed from the
     vendored dataset.json: 0=T1c, 1=T1n, 2=T2f, 3=T2w -- matching the challenge's
     own -t1c/-t1n/-t2f/-t2w suffixes in the same order, so no reordering needed).
  2. Run nnUNetv2_predict once per fold (0-4) with --save_probabilities -step_size 0.5
     -- identical flags to pipelines/nnunet/SLURM_SCRIPTS/R37b_dicetopk_5000ep_real/predict.slurm.
  3. nnUNetv2_ensemble the 5 folds (no postprocessing.pkl -- the real submission
     never applied one).
  4. Apply the learned ET false-positive rejector (apply_rejector.py --no-zip),
     identical to pipelines/fp_rejector/SLURM_SCRIPTS/apply_r37b.slurm.
  5. Copy the final flat masks into /output and verify the case count and the
     absence of any subdirectories before exiting successfully.

No network access is used or required at runtime; the model, plans, and rejector
are all baked into the image under /app/model.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path("/app")
MODEL_ROOT = APP_ROOT / "model"                     # -> nnUNet_results
SCRIPTS = APP_ROOT / "scripts"
REJECTOR = MODEL_ROOT / "rejector_R37b_leakfree.joblib"

DATASET_ID = "501"
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainerDiceTopK10Loss_5000epochs"
PLANS = "nnUNetPlans"
FOLDS = (0, 1, 2, 3, 4)

# challenge suffix -> nnU-Net channel index (verified against the vendored dataset.json:
# channel_names {"0":"T1c","1":"T1n","2":"T2f","3":"T2w"} -- an exact order match)
SUFFIX_TO_CHANNEL = {"t1c": "0000", "t1n": "0001", "t2f": "0002", "t2w": "0003"}


def run(cmd: list[str], **kw) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def discover_cases(input_dir: Path) -> list[str]:
    cases = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if not cases:
        raise SystemExit(f"No case folders found under {input_dir}")
    return cases


def stage_case(case_dir: Path, case: str, staging_dir: Path) -> None:
    for suffix, channel in SUFFIX_TO_CHANNEL.items():
        matches = list(case_dir.glob(f"*-{suffix}.nii.gz"))
        if len(matches) != 1:
            raise SystemExit(f"{case}: expected exactly one *-{suffix}.nii.gz file, "
                             f"found {len(matches)} in {case_dir}")
        dst = staging_dir / f"{case}_{channel}.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(matches[0].resolve(), dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--workdir", default="/work")
    ap.add_argument("--ensemble-workers", type=int, default=2,
                    help="matches the value used operationally in predict.slurm")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    work = Path(args.workdir)
    staging = work / "staged_input"
    final = work / "final"
    for d in (work, staging, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # nnU-Net probes these env vars even for predict-only use; only nnUNet_results
    # (-> the vendored model) is functionally load-bearing here.
    os.environ["nnUNet_raw"] = str(work / "_unused_raw")
    os.environ["nnUNet_preprocessed"] = str(work / "_unused_preprocessed")
    os.environ["nnUNet_results"] = str(MODEL_ROOT)
    os.environ["nnUNet_compile"] = "false"
    Path(os.environ["nnUNet_raw"]).mkdir(exist_ok=True)
    Path(os.environ["nnUNet_preprocessed"]).mkdir(exist_ok=True)

    print(f"=== Discovering cases in {input_dir} ===", flush=True)
    cases = discover_cases(input_dir)
    print(f"{len(cases)} cases: {cases[:5]}{' ...' if len(cases) > 5 else ''}", flush=True)

    print("=== Staging inputs (t1c/t1n/t2f/t2w -> _0000.._0003) ===", flush=True)
    for case in cases:
        stage_case(input_dir / case, case, staging)

    fold_dirs = []
    for fold in FOLDS:
        fold_out = work / f"fold_{fold}"
        fold_out.mkdir(exist_ok=True)
        fold_dirs.append(fold_out)
        print(f"=== Predicting fold {fold} ===", flush=True)
        run([
            "nnUNetv2_predict", "-i", str(staging), "-o", str(fold_out),
            "-d", DATASET_ID, "-c", CONFIGURATION,
            "-tr", TRAINER, "-p", PLANS, "-f", str(fold),
            "--save_probabilities", "--continue_prediction", "-step_size", "0.5",
        ])

    ensemble_out = work / "ensemble"
    ensemble_out.mkdir(exist_ok=True)
    print("=== Ensembling 5 folds ===", flush=True)
    run(["nnUNetv2_ensemble", "-i", *[str(d) for d in fold_dirs], "-o", str(ensemble_out),
        "-np", str(args.ensemble_workers)])

    print("=== Applying the ET false-positive rejector ===", flush=True)
    final.mkdir(exist_ok=True)
    run([
        sys.executable, str(SCRIPTS / "apply_rejector.py"),
        "--rejector", str(REJECTOR),
        "--base-masks", str(ensemble_out),
        "--et-prob-root", str(work),
        "--images", str(staging),
        "--out", str(final),
        "--run-tag", "docker",
        "--no-zip",
    ])

    print(f"=== Finalizing: copying results to {output_dir} ===", flush=True)
    written = 0
    for case in cases:
        src = final / f"{case}.nii.gz"
        if not src.exists():
            raise SystemExit(f"Missing predicted mask for case {case}: {src}")
        dst = output_dir / f"{case}.nii.gz"
        dst.write_bytes(src.read_bytes())
        written += 1

    subdirs = [p for p in output_dir.iterdir() if p.is_dir()]
    if subdirs:
        raise SystemExit(f"Output must be flat but found subdirectories: {subdirs}")
    if written != len(cases):
        raise SystemExit(f"Wrote {written} files but discovered {len(cases)} cases")

    print(f"=== DONE: {written} flat .nii.gz files written to {output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
