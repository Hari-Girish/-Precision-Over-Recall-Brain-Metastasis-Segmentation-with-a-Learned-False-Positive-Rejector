"""Exp21 — train the ET connected-component false-positive rejector.

Input: the component feature table from build_components.py (one row per ET blob,
target is_tp ∈ {0,1}). Output: a fitted classifier + a chosen keep/drop operating
point, validated with GROUPED-BY-CASE 5-fold CV (no case leaks across folds).

The classifier learns P(true ET | features). At inference we DROP components below a
threshold. We pick that threshold to maximize the BraTS-relevant objective: keep true
lesions (recall) while cutting false blobs (precision) — reported here as per-component
precision/recall and, more importantly, the net effect on lesion counts (TP kept, FP
dropped). Because a dropped TP is a lost lesion and a kept FP is a 0-Dice lesion, the
default operating point maximizes F1 with a guard that TP-recall stays ≥ a floor.

Default model: sklearn GradientBoostingClassifier (always available). --model lightgbm
is used if installed (optional: `uv add lightgbm`).

Usage:
  python train_rejector.py --table pipelines/fp_rejector/data/components_R37b.csv \
      --out pipelines/fp_rejector/data/rejector_R37b.joblib --min-tp-recall 0.97
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import joblib

NON_FEATURES = {"case", "comp_id", "is_tp"}

# GT-derived or train/test-mismatched features that MUST be dropped (the R48 lesson):
#   tc_overlap / wt_overlap — computed from ground-truth TC/WT at build time -> a target
#     LEAK (a real ET blob is inside GT-TC by definition; impossible to know at test).
#   fold_disagree — identically 0 in OOF (1 fold/case) but nonzero at test (5 folds) ->
#     a train/test distribution mismatch.
# Excluding them here yields the 15-feature leak-free rejector (AUC 0.86).
LEAKY_FEATURES_DEFAULT = "tc_overlap,wt_overlap,fold_disagree"


def choose_threshold_by_fpdrop(y_true, p, target_fp_drop):
    """LEAK-FREE operating point (the R48 methodology). Drop the lowest-scoring
    `target_fp_drop` fraction of FALSE components; report the lesion-DSC economics.

    thr = the score at the target_fp_drop quantile of FP scores, so ~that fraction of
    FP fall below it. drop_precision = of everything we drop, the fraction that were
    truly FP. Break-even on lesion-wise DSC is ~0.33-0.41 (a dropped TP is a marginal
    low-Dice lesion; a dropped FP is a 0-Dice lesion removed), so drop_precision above
    that is a net gain."""
    fp_scores = p[y_true == 0]
    thr = float(np.quantile(fp_scores, target_fp_drop))
    keep = p >= thr
    n_fp = int((y_true == 0).sum()); n_tp = int((y_true == 1).sum())
    fp_dropped = int((~keep & (y_true == 0)).sum())
    tp_lost = int((~keep & (y_true == 1)).sum())
    drop_prec = fp_dropped / max(1, fp_dropped + tp_lost)
    clears = drop_prec > 0.41
    return {"thr": thr, "auc": float(roc_auc_score(y_true, p)),
            "fp_dropped_frac": fp_dropped / max(1, n_fp),
            "tp_lost_frac": tp_lost / max(1, n_tp),
            "drop_precision": drop_prec,
            "rationale": (f"{int(round(target_fp_drop*100))}% FP-drop; "
                          + ("clears break-even (~0.33-0.41) under any plausible d_lost"
                             if clears else
                             f"drop_precision {drop_prec:.2f} BELOW break-even — reconsider"))}


def get_model(kind: str):
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=0)
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=0)


def choose_threshold(y_true, p, min_tp_recall):
    """Pick the drop-threshold that maximizes F1 subject to TP-recall ≥ floor.
    A component is KEPT if p >= thr. TP-recall = fraction of true ET blobs kept."""
    best = {"thr": 0.5, "f1": -1, "tp_recall": 0, "fp_kept_frac": 1}
    for thr in np.linspace(0.05, 0.95, 91):
        keep = p >= thr
        tp = int((keep & (y_true == 1)).sum())
        fp = int((keep & (y_true == 0)).sum())
        fn = int((~keep & (y_true == 1)).sum())        # true ET we DROPPED (bad)
        tp_recall = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * tp_recall / max(1e-9, prec + tp_recall)
        if tp_recall >= min_tp_recall and f1 > best["f1"]:
            n_fp_total = int((y_true == 0).sum())
            best = {"thr": float(thr), "f1": float(f1), "precision": float(prec),
                    "tp_recall": float(tp_recall),
                    "fp_dropped_frac": float((~keep & (y_true == 0)).sum() / max(1, n_fp_total))}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["gb", "lightgbm"], default="gb")
    ap.add_argument("--target-fp-drop", type=float, default=0.20,
                    help="LEAK-FREE default: drop the lowest-scoring 20%% of FP blobs "
                         "(operating point by lesion-DSC break-even, the R48 method)")
    ap.add_argument("--min-tp-recall", type=float, default=None,
                    help="LEGACY (leaky-era): if set, use the F1-under-recall-floor "
                         "threshold instead of --target-fp-drop")
    ap.add_argument("--drop-features", default=LEAKY_FEATURES_DEFAULT,
                    help="comma-sep features to exclude (default: the 3 GT-leaky/mismatch ones)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    dropped = {c.strip() for c in args.drop_features.split(",") if c.strip()}
    exclude = NON_FEATURES | dropped
    feat_cols = [c for c in df.columns if c not in exclude]
    present_drop = sorted(dropped & set(df.columns))
    print(f"excluded leaky/mismatch features: {present_drop or '(none present)'}")

    # Sanitize: ±inf -> NaN -> per-column median, then clip extreme tails. skimage can
    # emit inf (degenerate convex hull -> solidity) and near-zero eigenvalues can blow
    # up ratio features; sklearn rejects any non-finite value.
    feats = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    n_bad = int(feats.isna().sum().sum())
    feats = feats.fillna(feats.median())
    # guard any remaining all-NaN column (median is NaN) and absurd magnitudes
    feats = feats.fillna(0.0).clip(lower=-1e6, upper=1e6)
    X = feats.values.astype(np.float64)
    assert np.isfinite(X).all(), "non-finite values survived sanitization"
    if n_bad:
        print(f"sanitized {n_bad} non-finite/missing feature values")

    y = df["is_tp"].values
    groups = df["case"].values

    print(f"{len(df)} components | {y.sum()} TP / {(y==0).sum()} FP "
          f"({100*y.mean():.1f}% TP) | {df['case'].nunique()} cases | {len(feat_cols)} features")

    # grouped-by-case CV — out-of-fold predictions, no case leakage
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = get_model(args.model)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        print(f"  fold {k}: {len(tr)} train / {len(va)} val components")

    if args.min_tp_recall is not None:
        op = choose_threshold(y, oof, args.min_tp_recall)                 # legacy leaky-era path
        print(f"OPERATING POINT (LEGACY, min TP-recall {args.min_tp_recall}): "
              f"drop below p={op['thr']:.3f} | TP-recall {op['tp_recall']:.3f} "
              f"| FP dropped {op['fp_dropped_frac']:.3f} | F1 {op['f1']:.3f}")
    else:
        op = choose_threshold_by_fpdrop(y, oof, args.target_fp_drop)      # leak-free default
        print(f"OPERATING POINT (leak-free, {int(round(args.target_fp_drop*100))}% FP-drop): "
              f"drop below p={op['thr']:.3f} | AUC {op['auc']:.4f} | "
              f"FP-dropped {op['fp_dropped_frac']:.3f} | TP-lost {op['tp_lost_frac']:.3f} | "
              f"drop-precision {op['drop_precision']:.3f}\n  -> {op['rationale']}")

    # refit on all data for the deployable model
    final = get_model(args.model)
    final.fit(X, y)

    # feature importances (interpretability — which signals separate FP from TP)
    try:
        imp = sorted(zip(feat_cols, final.feature_importances_), key=lambda t: -t[1])
        print("Top features:", ", ".join(f"{n}={v:.3f}" for n, v in imp[:8]))
    except Exception:
        imp = []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": feat_cols, "operating_point": op,
                 "importances": imp}, args.out)
    # sidecar json for quick inspection
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"operating_point": op, "features": feat_cols,
         "importances": [[n, float(v)] for n, v in imp]}, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
