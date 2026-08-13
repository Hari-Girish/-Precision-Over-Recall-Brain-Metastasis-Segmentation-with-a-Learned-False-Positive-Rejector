"""Exp21 — apply the ET FP-rejector to a base model's test predictions.

For each test case: extract ET connected components from the base 4-class mask (or from
an ET-prob map), compute the same features as build_components.py, score them with the
trained rejector, and DROP components below the operating-point threshold (relabel those
ET voxels to background — protecting recall via the threshold's TP-recall floor). Write
the filtered 4-class masks + a submission zip.

This is the precision-side postprocessing: it removes false ET blobs WITHOUT adding any
(the recall-cascade line that failed 3× is not reopened). Filter target = E3 champion or
R37b, using that model's own ET-prob map for the probability features.

Usage:
  python apply_rejector.py \
      --rejector pipelines/fp_rejector/data/rejector_R37b.joblib \
      --base-masks <dir of BraTS-MET-*.nii.gz 4-class test masks> \
      --et-prob-root <dir with fold_*/<case>.npz for the SAME base model on test> \
      --images <Dataset501.../imagesTs> \
      --out <out dir> --run-tag R48_R37bfilt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import joblib
from scipy.ndimage import label as cc_label

# reuse the exact feature extraction + axis guard from the builder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_components import (  # noqa: E402
    load_oof_et_prob, _align, _shell_mean, _elongation, _safe_solidity,
    CONN_26, ET_LABEL, TC_LABELS, WT_LABELS, T1C_CH, T1N_CH,
)
from skimage.measure import regionprops  # noqa: E402


def component_feature_row(cid, comp, et_prob, et_stack, t1c, sub, gt_free_tc, gt_free_wt,
                          props, shape):
    """Same features as build_components.case_features, MINUS the is_tp target.
    NOTE: tc/wt overlap here uses the BASE MASK's own TC/WT (not GT — none at test)."""
    size = int(comp.sum())
    pv = et_prob[comp]
    fold_means = et_stack[:, comp].mean(1) if et_stack.shape[0] > 1 else np.array([pv.mean()])
    pr = props.get(cid)
    centroid = np.array(pr.centroid) if pr else np.array([c.mean() for c in np.where(comp)])
    return {
        "size_vox": size,
        "et_mean": float(pv.mean()), "et_peak": float(pv.max()), "et_std": float(pv.std()),
        "et_p25": float(np.percentile(pv, 25)),
        "fold_disagree": float(fold_means.std()),
        "cz": float(centroid[0] / shape[0]), "cy": float(centroid[1] / shape[1]),
        "cx": float(centroid[2] / shape[2]),
        "extent": float(pr.extent) if pr else np.nan,
        "solidity": _safe_solidity(pr) if pr else np.nan,
        "elongation": float(_elongation(pr)) if pr else np.nan,
        "t1c_mean": float(t1c[comp].mean()),
        "t1c_contrast": float(t1c[comp].mean() - _shell_mean(t1c, comp)),
        "sub_mean": float(sub[comp].mean()),
        "sub_contrast": float(sub[comp].mean() - _shell_mean(sub, comp)),
        "tc_overlap": float((comp & gt_free_tc).sum() / size),
        "wt_overlap": float((comp & gt_free_wt).sum() / size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejector", required=True)
    ap.add_argument("--base-masks", required=True)
    ap.add_argument("--et-prob-root", required=True, help="fold_*/<case>.npz for base model on TEST")
    ap.add_argument("--images", required=True, help="imagesTs dir for T1c/T1n context")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-tag", default="R48_filtered")
    ap.add_argument("--threshold", type=float, default=None, help="override operating point")
    ap.add_argument("--no-zip", action="store_true",
                    help="skip building the internal-project submission zip (the hardcoded "
                         "/fs/ess/PDE0022/... path doesn't exist outside the HPC project — use "
                         "this in the Docker container, where --out is already the final flat "
                         "output directory). Default unchanged for existing internal callers.")
    args = ap.parse_args()

    bundle = joblib.load(args.rejector)
    model, feat_cols = bundle["model"], bundle["features"]
    thr = args.threshold if args.threshold is not None else bundle["operating_point"]["thr"]
    print(f"rejector features: {len(feat_cols)} | drop-below p={thr:.3f}")

    base_dir = Path(args.base_masks)
    img_dir = Path(args.images)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name[:-len(".nii.gz")] for p in base_dir.glob("BraTS-MET-*.nii.gz"))
    print(f"{len(cases)} test cases")

    total_dropped = total_comps = 0
    for case in cases:
        base_img = nib.load(base_dir / f"{case}.nii.gz")
        base = np.asanyarray(base_img.dataobj).astype(np.uint8)   # (X,Y,Z)

        et_prob, et_stack = load_oof_et_prob(Path(args.et_prob_root), case)
        if et_prob is None:
            nib.save(base_img, out / f"{case}.nii.gz")             # no probs → passthrough
            continue
        et_prob = _align(et_prob, base.shape)
        et_stack = np.stack([_align(f, base.shape) for f in et_stack], 0)

        t1c = np.asanyarray(nib.load(img_dir / f"{case}_{T1C_CH}.nii.gz").dataobj).astype(np.float32)
        t1n = np.asanyarray(nib.load(img_dir / f"{case}_{T1N_CH}.nii.gz").dataobj).astype(np.float32)
        sub = t1c - t1n
        base_tc = np.isin(base, TC_LABELS)
        base_wt = np.isin(base, WT_LABELS)

        # components of the BASE mask's ET (label 3) — these are what we filter
        comps, ncomp = cc_label(base == ET_LABEL, structure=CONN_26)
        if ncomp == 0:
            nib.save(base_img, out / f"{case}.nii.gz")
            continue
        props = {p.label: p for p in regionprops(comps)}
        shape = np.array(base.shape, dtype=np.float32)

        final = base.copy()
        for cid in range(1, ncomp + 1):
            comp = comps == cid
            feats = component_feature_row(cid, comp, et_prob, et_stack, t1c, sub,
                                          base_tc, base_wt, props, shape)
            x = np.array([[feats.get(c, np.nan) for c in feat_cols]], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0)
            p_keep = float(model.predict_proba(x)[0, 1])
            total_comps += 1
            if p_keep < thr:
                # DROP: relabel this ET component to background (protects TC/WT since
                # ET⊂TC⊂WT, we only remove the ET label at these voxels)
                final[comp] = 0
                total_dropped += 1
        nib.save(nib.Nifti1Image(final, base_img.affine, base_img.header), out / f"{case}.nii.gz")

    print(f"DONE — dropped {total_dropped}/{total_comps} ET components across {len(cases)} cases")
    if args.no_zip:
        print(f"--no-zip set: final masks already written flat to {out}; skipping submission zip.")
    else:
        # build submission zip (internal HPC-project convenience path; not used in Docker)
        import zipfile
        zpath = Path("/fs/ess/PDE0022/hari/brats2026/submissions") / f"{args.run_tag}_submission.zip"
        zpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.glob("BraTS-MET-*.nii.gz")):
                z.write(f, f.name)
        print(f"SUBMISSION ZIP: {zpath}")


if __name__ == "__main__":
    main()
