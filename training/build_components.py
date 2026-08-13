"""Exp21 — build the ET connected-component feature table for the FP-rejector.

For each training case, read the base model's OOF ET-probability map, extract ET
connected components above a permissive threshold, label each component TP vs FP by
overlap with GT ET (label 3), and compute per-component features. Output one CSV row
per component keyed by (case, comp_id) with the TP/FP target.

The rejector learns "which ET blobs are false" so we can DROP them at inference —
attacking the precision half of the ET-0.70 wall (the recall half is Exp20).

Feature set (from Deep Research Report 2): ET mean/peak/std prob, volume, shape
(elongation, extent, solidity), centroid location (normalized), T1c mean intensity +
contrast to shell, T1c-T1 subtraction contrast, overlap fraction with TC and WT,
fold-disagreement uncertainty.

Axis convention (the E3 v1 trap): nnUNet npz probabilities are (C,Z,Y,X) internal order;
GT/image NIfTI are (X,Y,Z). We transpose prob to the image axes and assert shapes match.

Usage:
  python build_components.py --oof-root /fs/scratch/.../R48_oof \
      --out pipelines/fp_rejector/data/components_R37b.csv --threshold 0.30 --workers 8
"""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label
from skimage.measure import regionprops

NN = "/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet"
D501 = Path(NN) / "data/raw/Dataset501_BraTS_MET_2026_Training"
LABELS = D501 / "labelsTr"
IMAGES = D501 / "imagesTr"

ET_LABEL = 3
TC_LABELS = (1, 3)          # tumor core = NETC + ET (BraTS-MET)
WT_LABELS = (1, 2, 3, 4)    # whole tumor = all foreground
CONN_26 = np.ones((3, 3, 3), dtype=int)
T1C_CH, T1N_CH = "0000", "0001"


def load_oof_et_prob(oof_root: Path, case: str):
    """Average per-fold ET foreground softmax → (Z,Y,X). Also return per-fold ET
    probs (list) for a disagreement/uncertainty feature. R37b is 4-class, so ET is
    channel index 3 (labels 0=bg,1,2,3=ET,4)."""
    per_fold = []
    for fold in range(5):
        p = oof_root / f"fold_{fold}" / f"{case}.npz"
        if not p.exists():
            continue
        probs = np.load(p)["probabilities"].astype(np.float32)  # (C, Z, Y, X)
        # 4-class model: ET is channel 3. Guard for binary models (2ch → ch1).
        et_ch = 3 if probs.shape[0] >= 4 else probs.shape[0] - 1
        per_fold.append(probs[et_ch])
    if not per_fold:
        return None, None
    stack = np.stack(per_fold, 0)          # (F, Z, Y, X)
    return stack.mean(0), stack


def _align(prob_zyx: np.ndarray, ref_xyz_shape):
    """Transpose (Z,Y,X) prob to the reference (X,Y,Z) shape; assert match."""
    if prob_zyx.shape != ref_xyz_shape:
        prob_zyx = prob_zyx.transpose(2, 1, 0)
    assert prob_zyx.shape == ref_xyz_shape, f"shape {prob_zyx.shape} != ref {ref_xyz_shape}"
    return prob_zyx


def case_features(args):
    case, oof_root, threshold, min_vox = args
    oof_root = Path(oof_root)

    gt_p = LABELS / f"{case}.nii.gz"
    if not gt_p.exists():
        return []
    gt = np.asanyarray(nib.load(gt_p).dataobj).astype(np.uint8)   # (X,Y,Z)

    et_prob, et_stack = load_oof_et_prob(oof_root, case)
    if et_prob is None:
        return []
    et_prob = _align(et_prob, gt.shape)
    et_stack = np.stack([_align(f, gt.shape) for f in et_stack], 0)

    # multimodal context
    t1c = np.asanyarray(nib.load(IMAGES / f"{case}_{T1C_CH}.nii.gz").dataobj).astype(np.float32)
    t1n = np.asanyarray(nib.load(IMAGES / f"{case}_{T1N_CH}.nii.gz").dataobj).astype(np.float32)
    sub = t1c - t1n

    gt_et = gt == ET_LABEL
    tc = np.isin(gt, TC_LABELS)
    wt = np.isin(gt, WT_LABELS)

    comps, ncomp = cc_label(et_prob >= threshold, structure=CONN_26)
    if ncomp == 0:
        return []

    shape = np.array(gt.shape, dtype=np.float32)
    rows = []
    props = {p.label: p for p in regionprops(comps)}
    for cid in range(1, ncomp + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_vox:
            continue
        pv = et_prob[comp]
        # per-fold disagreement inside the component (uncertainty)
        fold_means = et_stack[:, comp].mean(1) if et_stack.shape[0] > 1 else np.array([pv.mean()])
        # a dilated shell around the component for local contrast features
        pr = props.get(cid)
        centroid = np.array(pr.centroid) if pr else np.array([c.mean() for c in np.where(comp)])

        # TP/FP target: component overlaps GT ET meaningfully
        inter = int((comp & gt_et).sum())
        is_tp = 1 if inter >= max(1, int(0.10 * size)) else 0   # ≥10% of blob is real ET

        row = {
            "case": case,
            "comp_id": cid,
            "is_tp": is_tp,
            "size_vox": size,
            "et_mean": float(pv.mean()),
            "et_peak": float(pv.max()),
            "et_std": float(pv.std()),
            "et_p25": float(np.percentile(pv, 25)),
            "fold_disagree": float(fold_means.std()),
            "cz": float(centroid[0] / shape[0]),
            "cy": float(centroid[1] / shape[1]),
            "cx": float(centroid[2] / shape[2]),
            "extent": float(pr.extent) if pr else np.nan,
            "solidity": _safe_solidity(pr) if pr else np.nan,
            "elongation": float(_elongation(pr)) if pr else np.nan,
            "t1c_mean": float(t1c[comp].mean()),
            "t1c_contrast": float(t1c[comp].mean() - _shell_mean(t1c, comp)),
            "sub_mean": float(sub[comp].mean()),
            "sub_contrast": float(sub[comp].mean() - _shell_mean(sub, comp)),
            "tc_overlap": float((comp & tc).sum() / size),
            "wt_overlap": float((comp & wt).sum() / size),
        }
        rows.append(row)
    return rows


ELONGATION_CAP = 100.0   # degenerate (flat/thin) blobs otherwise reach ~1e6


def _elongation(pr):
    """Ratio of major to minor inertia-tensor eigenvalue (≥1; larger = more elongated).

    Degenerate components (flat or 1-voxel-thick) have a near-zero smallest eigenvalue,
    which makes the raw ratio explode (observed up to 2.9e6) and pollutes the feature
    with a heavy tail. Clamp the denominator and cap the ratio: past ~100 the blob is
    "extremely elongated" and the exact value carries no extra signal.
    """
    try:
        ev = np.sort(np.asarray(pr.inertia_tensor_eigvals, dtype=np.float64))
        lo = max(float(ev[0]), 1e-3)          # clamp, not epsilon-add
        hi = max(float(ev[-1]), 0.0)
        val = (hi + 1e-3) / lo
        if not np.isfinite(val):
            return ELONGATION_CAP
        return float(min(val, ELONGATION_CAP))
    except Exception:
        return np.nan


def _safe_solidity(pr):
    """skimage solidity is inf/NaN when the convex hull is degenerate (flat blobs).
    Those are exactly the thin false-positive shapes we care about → map to 1.0
    (a degenerate hull equals the region), never inf."""
    try:
        s = float(pr.solidity)
        return s if np.isfinite(s) and 0.0 < s <= 1.0 else 1.0
    except Exception:
        return 1.0


def _shell_mean(vol, comp):
    """Mean intensity in a 1-voxel dilated shell just outside the component."""
    from scipy.ndimage import binary_dilation
    shell = binary_dilation(comp, iterations=2) & ~comp
    if not shell.any():
        return float(vol[comp].mean())
    return float(vol[shell].mean())


def list_cases():
    return sorted(p.name[:-len(".nii.gz")] for p in LABELS.glob("*.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-root", required=True, help="dir with fold_{0..4}/<case>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--min-vox", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (unit test)")
    args = ap.parse_args()

    cases = list_cases()
    if args.limit:
        cases = cases[: args.limit]
    tasks = [(c, args.oof_root, args.threshold, args.min_vox) for c in cases]

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rows in enumerate(ex.map(case_features, tasks)):
            all_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(cases)} cases, {len(all_rows)} components")

    if not all_rows:
        print("NO COMPONENTS — check --oof-root has fold_*/<case>.npz")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    n_tp = sum(r["is_tp"] for r in all_rows)
    print(f"WROTE {args.out}: {len(all_rows)} components from {len(cases)} cases "
          f"({n_tp} TP / {len(all_rows) - n_tp} FP, {100*n_tp/len(all_rows):.1f}% TP)")


if __name__ == "__main__":
    main()
