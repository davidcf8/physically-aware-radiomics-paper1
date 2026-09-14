#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve
)

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# ---------------------------------------------------------------------
# IMPORTANT: names must match objects saved by modeling.py
# ---------------------------------------------------------------------
@dataclass
class ModelBundle:
    feature_names: List[str]
    model: object
    model_name: str
    arm_id: str
    threshold: float
    used_combat: bool
    used_feature_mode: str
    selected_features_source: str
    outer_train_size: int
    outer_test_size: int
    combat_batch_mode: Optional[str] = None


@dataclass
class MLPBundle:
    feature_names: List[str]
    model_state_dict: Dict[str, object]
    model_name: str
    arm_id: str
    threshold: float
    used_combat: bool
    used_feature_mode: str
    selected_features_source: str
    outer_train_size: int
    outer_test_size: int
    mlp_params: dict
    imputer_statistics: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    combat_batch_mode: Optional[str] = None


class SmallMLP(nn.Module):
    def __init__(self, n_in: int, hidden1: int = 64, hidden2: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


BENIGN = {1, 2}
MALIGN = {4, 5}


def parse_arm_id(arm_id: str) -> Dict[str, str]:
    out = {}
    for token in arm_id.split("|"):
        k, v = token.split("=")
        out[k] = v
    return out


def parse_A_from_filename(fp: Path) -> str:
    name = fp.name.lower()
    if "fake" in name or re.search(r"(^|_)fk(_|\.|$)", name) or "nr_fake" in name or "a-fk" in name:
        return "FK"
    if "no_resample" in name or re.search(r"(^|_)nr(_|\.|$)", name) or "a-nr" in name:
        return "NR"
    if ("resample" in name and "no_resample" not in name) or re.search(r"(^|_)rs(_|\.|$)", name) or "a-rs" in name:
        return "RS"
    if re.search(r"(^|_)vs(_|\.|$)", name) or "voxelspacing" in name or "voxel_spacing" in name or "a-vs" in name:
        return "VS"
    raise ValueError(f"Cannot infer A from: {fp.name}")


def is_axisA_pure_csv(fp: Path) -> bool:
    name = fp.name.lower()
    if "density" in name or "combat" in name:
        return False
    return True


def find_external_csvs(external_ready: Path) -> Dict[str, Path]:
    merged = external_ready / "merged_homogeneous"
    if not merged.exists():
        raise FileNotFoundError(f"Missing: {merged}")

    out = {}
    for fp in sorted(merged.glob("*.csv")):
        if not is_axisA_pure_csv(fp):
            continue
        A = parse_A_from_filename(fp)
        out[A] = fp

    missing = {"NR", "RS", "VS", "FK"} - set(out)
    if missing:
        print(f"[WARN] Missing external CSVs for: {sorted(missing)}")

    return out


def ensure_binary_label(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    label_col: str = "malignancy_label",
    score_col: str = "malignancy_score",
) -> pd.DataFrame:
    df = df.copy()

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        y = pd.to_numeric(df[target_col], errors="coerce")
        df[label_col] = y
    elif label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce")
        mapped = []
        for v in y:
            if pd.isna(v):
                mapped.append(np.nan)
            else:
                iv = int(v)
                if iv in (0, 1):
                    mapped.append(iv)
                elif iv in BENIGN:
                    mapped.append(0)
                elif iv in MALIGN:
                    mapped.append(1)
                else:
                    mapped.append(np.nan)
        df[label_col] = mapped
    elif score_col in df.columns:
        y = pd.to_numeric(df[score_col], errors="coerce")
        mapped = []
        for v in y:
            if pd.isna(v):
                mapped.append(np.nan)
            else:
                iv = int(v)
                if iv in BENIGN:
                    mapped.append(0)
                elif iv in MALIGN:
                    mapped.append(1)
                else:
                    mapped.append(np.nan)
        df[label_col] = mapped
    else:
        raise ValueError(f"Need {target_col or label_col} or {score_col}.")

    before = len(df)
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype(int)

    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df)} rows with invalid labels.")

    if df[label_col].nunique() < 2:
        print("[WARN] External cohort has only one class. AUC will be NaN.")

    return df


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "auc": np.nan,
        "ap": np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob)),
        "acc": float((tp + tn) / max(1, len(y_true))),
        "sens": float(tp / max(1, tp + fn)),
        "spec": float(tn / max(1, tn + fp)),
        "ppv": float(tp / max(1, tp + fp)),
        "npv": float(tn / max(1, tn + fn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(thr),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }

    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass

    return out


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = 1337):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(vals) < 50:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def predict_model_bundle(bundle: ModelBundle, X_df: pd.DataFrame) -> np.ndarray:
    fitted = bundle.model

    if isinstance(fitted, dict) and "clf" in fitted:
        if "imputer" in fitted:
            X = fitted["imputer"].transform(X_df)
        else:
            X = X_df.to_numpy(dtype=float)

        if "scaler" in fitted:
            X = fitted["scaler"].transform(X)

        return fitted["clf"].predict_proba(X)[:, 1]

    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X_df)[:, 1]

    raise TypeError(f"Unsupported model object for {bundle.model_name}")


def predict_mlp_bundle(bundle: MLPBundle, X_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch not available.")

    X = X_df.to_numpy(dtype=float)

    # median imputation using frozen training medians
    stats = np.asarray(bundle.imputer_statistics, dtype=float)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(stats, inds[1])

    mean = np.asarray(bundle.scaler_mean, dtype=float)
    scale = np.asarray(bundle.scaler_scale, dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    X = ((X - mean) / scale).astype(np.float32)

    params = bundle.mlp_params
    model = SmallMLP(
        n_in=X.shape[1],
        hidden1=int(params.get("hidden1", 64)),
        hidden2=int(params.get("hidden2", 16)),
        dropout=float(params.get("dropout", 0.2)),
    )
    model.load_state_dict(bundle.model_state_dict)
    model.eval()
    model.to(device)

    probs = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            xb = torch.tensor(X[start:start + 256], dtype=torch.float32, device=device)
            p = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs.append(p)

    return np.concatenate(probs)


def evaluate_one_bundle(
    bundle_path: Path,
    external_csvs: Dict[str, Path],
    outdir: Path,
    target_col: Optional[str],
    id_col: str,
    label_col: str,
    device: str,
    seed: int,
) -> Dict[str, object]:
    obj = joblib.load(bundle_path)

    if not hasattr(obj, "arm_id"):
        raise TypeError(f"{bundle_path.name}: object has no arm_id")

    info = parse_arm_id(obj.arm_id)
    A = info["A"]

    if A not in external_csvs:
        raise FileNotFoundError(f"No external CSV found for A={A}")

    df = pd.read_csv(external_csvs[A])
    df = ensure_binary_label(df, target_col=target_col, label_col=label_col)

    required = list(obj.feature_names)
    missing = [f for f in required if f not in df.columns]
    if missing:
        raise ValueError(f"{bundle_path.name}: missing {len(missing)} features, e.g. {missing[:5]}")

    X_df = df[required].apply(pd.to_numeric, errors="coerce")
    y = df[label_col].to_numpy(dtype=int)

    if isinstance(obj, MLPBundle) or obj.model_name == "MLP":
        y_prob = predict_mlp_bundle(obj, X_df, device=device)
    else:
        y_prob = predict_model_bundle(obj, X_df)

    metrics = evaluate_binary(y, y_prob, thr=float(getattr(obj, "threshold", 0.5)))
    ci_lo, ci_hi = bootstrap_auc_ci(y, y_prob, seed=seed)
    metrics["auc_ci95_low"] = ci_lo
    metrics["auc_ci95_high"] = ci_hi

    metrics.update({
        "analysis_run": bundle_path.parent.parent.name,
        "bundle": bundle_path.name,
        "arm_id": obj.arm_id,
        "A": A,
        "B": info.get("B"),
        "N_axis": info.get("N"),
        "C": info.get("C"),
        "model": obj.model_name,
        "feature_mode": getattr(obj, "used_feature_mode", None),
        "n_features": len(required),
        "external_csv": str(external_csvs[A]),
        "bundle_path": str(bundle_path),
    })

    run_out = outdir / bundle_path.parent.parent.name
    run_out.mkdir(parents=True, exist_ok=True)

    tag = bundle_path.stem
    pred = pd.DataFrame({
        id_col: df[id_col].astype(str).values if id_col in df.columns else np.arange(len(df)).astype(str),
        "y_true": y,
        "y_prob": y_prob,
        "y_pred": (y_prob >= metrics["threshold"]).astype(int),
        "A": A,
        "model": obj.model_name,
        "arm_id": obj.arm_id,
    })
    pred.to_csv(run_out / f"external_predictions__{tag}.csv", index=False)

    with open(run_out / f"external_metrics__{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_root", required=True, type=Path,
                    help="Root containing modeling run folders, e.g. modelado_axisA_CT.")
    ap.add_argument("--external_ready", required=True, type=Path,
                    help="External ready folder containing merged_homogeneous/*.csv.")
    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--target_col", default=None,
                    help="For MRI use pathological_complete_response. For CT leave empty.")
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--label_col", default="malignancy_label")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--run_glob", default="*",
                    help="Which modeling runs to evaluate, e.g. '*common_neutral*'.")
    ap.add_argument("--bundle_glob", default="bundles/*.joblib")

    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    external_csvs = find_external_csvs(args.external_ready)
    print("[INFO] External CSVs:")
    for k, v in external_csvs.items():
        print(f"  {k}: {v}")

    all_metrics = []
    failures = []

    run_dirs = sorted([p for p in args.model_root.glob(args.run_glob) if p.is_dir()])

    for run_dir in run_dirs:
        bundle_paths = sorted(run_dir.glob(args.bundle_glob))
        if not bundle_paths:
            print(f"[WARN] No bundles in {run_dir}")
            continue

        for bp in bundle_paths:
            try:
                m = evaluate_one_bundle(
                    bundle_path=bp,
                    external_csvs=external_csvs,
                    outdir=args.outdir,
                    target_col=args.target_col,
                    id_col=args.id_col,
                    label_col=args.label_col,
                    device=args.device,
                    seed=args.seed,
                )
                all_metrics.append(m)
                print(f"[OK] {run_dir.name} | {bp.name} | A={m['A']} | {m['model']} | AUC={m['auc']:.3f}")
            except Exception as e:
                failures.append({
                    "analysis_run": run_dir.name,
                    "bundle": bp.name,
                    "error": str(e),
                })
                print(f"[FAIL] {run_dir.name} | {bp.name}: {e}")

    summary = pd.DataFrame(all_metrics)
    summary_path = args.outdir / "external_validation_summary_all.csv"
    summary.to_csv(summary_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(args.outdir / "external_validation_failures.csv", index=False)

    print("\nSaved:")
    print(summary_path)

    if len(summary):
        cols = ["analysis_run", "A", "model", "feature_mode", "n", "n_features", "auc",
                "auc_ci95_low", "auc_ci95_high", "ap", "brier", "ece", "sens", "spec"]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].sort_values(["analysis_run", "model", "A"]).to_string(index=False))


if __name__ == "__main__":
    main()


""" python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_CT \
  --external_ready <REPO_PARENT>/LIDL_lung/results/CTexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/CT_common_neutral \
  --run_glob "*common_neutral*"

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_CT \
  --external_ready <REPO_PARENT>/LIDL_lung/results/CTexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/CT_per_arm_ensemble \
  --run_glob "*per_arm_ensemble*"

python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_CT \
  --external_ready <REPO_PARENT>/LIDL_lung/results/CTexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/CT_MLP_common_neutral \
  --run_glob "*common_neutral_mlp*"

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --external_ready <REPO_PARENT>/LIDL_lung/results/MRIexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/MRI_common_neutral \
  --target_col pathological_complete_response \
  --run_glob "*common_neutral*"

  python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --external_ready <REPO_PARENT>/LIDL_lung/results/MRIexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/MRI_per_arm_ensemble \
  --target_col pathological_complete_response \
  --run_glob "*per_arm_ensemble*"

python3 <REPO_PARENT>/LIDL_lung/scripts/analisis/external_val.py \
  --model_root <REPO_PARENT>/LIDL_lung/study_results/modelado_axisA_MRI \
  --external_ready <REPO_PARENT>/LIDL_lung/results/MRIexvalready \
  --outdir <REPO_PARENT>/LIDL_lung/study_results/externalvalidation/MRI_MLP_common_neutral \
  --target_col pathological_complete_response \
  --run_glob "*common_neutral_mlp*"


"""