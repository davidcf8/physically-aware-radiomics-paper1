#!/usr/bin/env python3
"""
Post-hoc patient-level bootstrap ROC AUC confidence intervals for the Paper 1
NUEVOS Axis A-only common-neutral geometry comparison.

This script reads existing held-out and external prediction files only. It
does not fit, tune, select features, or modify any source result files.
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "results" / "reference_outputs"
OUTDIR = REPO_ROOT / "results" / "reproduced" / "bootstrap_auc_ci"
FINAL_MODELING = ROOT / "modeling"
FINAL_EXTERNAL = ROOT / "external_validation"
N_BOOT = 10_000
SEED = 1337
ARMS = ("NR", "RS", "VS", "FK")
MODELS = ("LR", "XGB", "MLP")
BASELINE_B = "OFF"
BASELINE_N = "OFF"
BASELINE_C = "OFF"
ANALYSIS = "Paper 1 NUEVOS common-neutral Axis A-only baseline (B=OFF, N=OFF, C=OFF)"
AUC_TOL = 1e-10


@dataclass(frozen=True)
class CohortSpec:
    modality: str
    validation: str
    run_dir: Path
    summary_file: Path
    external_dir: Path | None = None


FINAL_SPECS = (
    CohortSpec(
        modality="CT",
        validation="internal held-out",
        run_dir=FINAL_MODELING / "CT_axisA_common_neutral_mlp",
        summary_file=FINAL_MODELING / "CT_axisA_common_neutral_mlp" / "results_models.csv",
    ),
    CohortSpec(
        modality="MRI",
        validation="internal held-out",
        run_dir=FINAL_MODELING / "MRI_axisA_common_neutral_mlp",
        summary_file=FINAL_MODELING / "MRI_axisA_common_neutral_mlp" / "results_models.csv",
    ),
    CohortSpec(
        modality="CT",
        validation="external",
        run_dir=FINAL_MODELING / "CT_axisA_common_neutral_mlp",
        summary_file=FINAL_EXTERNAL / "CT_MLP_common_neutral" / "external_validation_summary_all.csv",
        external_dir=FINAL_EXTERNAL / "CT_MLP_common_neutral" / "CT_axisA_common_neutral_mlp",
    ),
    CohortSpec(
        modality="MRI",
        validation="external",
        run_dir=FINAL_MODELING / "MRI_axisA_common_neutral_mlp",
        summary_file=FINAL_EXTERNAL / "MRI_MLP_common_neutral" / "external_validation_summary_all.csv",
        external_dir=FINAL_EXTERNAL / "MRI_MLP_common_neutral" / "MRI_axisA_common_neutral_mlp",
    ),
)


def arm_id(arm: str) -> str:
    return f"A={arm}|B={BASELINE_B}|N={BASELINE_N}|C={BASELINE_C}"


def filename_arm_id(arm: str) -> str:
    return f"A-{arm}__B-{BASELINE_B}__N-{BASELINE_N}__C-{BASELINE_C}"


def infer_modality(path: Path, df: pd.DataFrame | None = None) -> str:
    text = str(path)
    if df is not None and "modality" in df.columns:
        vals = sorted({str(v) for v in df["modality"].dropna().unique()})
        if len(vals) == 1 and vals[0] in {"CT", "MRI"}:
            return vals[0]
    if re.search(r"(^|[/_ -])MRI([/_ -]|$)", text, flags=re.I):
        return "MRI"
    if re.search(r"(^|[/_ -])CT([/_ -]|$)", text, flags=re.I):
        return "CT"
    return "unknown"


def infer_analysis(path: Path, df: pd.DataFrame | None = None, cfg: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    text = str(path).lower()
    feature_mode = None
    if cfg and cfg.get("feature_mode"):
        feature_mode = str(cfg["feature_mode"])
    elif df is not None and "feature_mode" in df.columns and df["feature_mode"].notna().any():
        vals = sorted({str(v) for v in df["feature_mode"].dropna().unique()})
        feature_mode = ",".join(vals)
    if feature_mode:
        parts.append(feature_mode)
    if "common_neutral" in text or "common-neutral" in text:
        parts.append("common-neutral")
    if "common_vs" in text:
        parts.append("common-vs")
    if "per_arm" in text or "per-arm" in text:
        parts.append("per-arm")
    if "full_features" in text or "full-feature" in text:
        parts.append("full-feature")
    if "mwu" in text:
        parts.append("MWU")
    if "ensemble" in text:
        parts.append("ensemble")
    if not parts:
        parts.append("unclassified")
    return "; ".join(dict.fromkeys(parts))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - audit message only
        return {"_read_error": str(exc)}


def models_from_filenames(files: list[Path]) -> list[str]:
    models: set[str] = set()
    for f in files:
        match = re.search(r"__(LR|XGB|MLP)__", f.name)
        if match:
            models.add(match.group(1))
    return sorted(models)


def arms_from_series(values: pd.Series) -> list[str]:
    arms: set[str] = set()
    for value in values.dropna().astype(str):
        match = re.search(r"A[=-](NR|RS|VS|FK)", value)
        if match:
            arms.add(match.group(1))
    return sorted(arms)


def candidate_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for pattern in ("results_models.csv", "preds_test_all.csv", "external_validation_summary_all.csv"):
        for f in ROOT.rglob(pattern):
            p = f.parent
            text = str(p).lower()
            if any(k in text for k in ("axisa", "axis_a", "common_neutral", "common-neutral", "per_arm", "full_features", "common_vs")):
                paths.add(p)
    for f in ROOT.rglob("external_predictions__bundle__*.csv"):
        p = f.parent
        text = str(p).lower()
        if any(k in text for k in ("axisa", "axis_a", "common_neutral", "common-neutral", "per_arm", "full_features", "common_vs")):
            paths.add(p)

    for p in sorted(paths):
        results = p / "results_models.csv"
        preds_all = p / "preds_test_all.csv"
        external_summary = p / "external_validation_summary_all.csv"
        ext_pred_files = sorted(p.glob("external_predictions__bundle__*.csv"))
        cfg = read_json(p / "run_config.json")
        df: pd.DataFrame | None = None
        cols: list[str] = []
        models: list[str] = []
        arms: list[str] = []
        unique_patients: str | int = "NA"

        if results.exists():
            df = pd.read_csv(results)
            cols = list(df.columns)
            if "model" in df.columns:
                models = sorted(df["model"].dropna().astype(str).unique())
            if "A" in df.columns:
                arms = sorted(df["A"].dropna().astype(str).unique())
            elif "arm_id" in df.columns:
                arms = arms_from_series(df["arm_id"])
        elif external_summary.exists():
            df = pd.read_csv(external_summary)
            cols = list(df.columns)
            if "model" in df.columns:
                models = sorted(df["model"].dropna().astype(str).unique())
            if "A" in df.columns:
                arms = sorted(df["A"].dropna().astype(str).unique())
            elif "arm_id" in df.columns:
                arms = arms_from_series(df["arm_id"])
        elif preds_all.exists():
            df = pd.read_csv(preds_all, nrows=0)
            cols = list(df.columns)

        if preds_all.exists():
            pred_df = pd.read_csv(preds_all, usecols=lambda c: c in {"patient_id", "model", "arm_id"})
            if "patient_id" in pred_df.columns:
                unique_patients = int(pred_df["patient_id"].nunique())
            if not models and "model" in pred_df.columns:
                models = sorted(pred_df["model"].dropna().astype(str).unique())
            if not arms and "arm_id" in pred_df.columns:
                arms = arms_from_series(pred_df["arm_id"])
            if not cols:
                cols = list(pred_df.columns)
        elif ext_pred_files:
            first = pd.read_csv(ext_pred_files[0], nrows=0)
            cols = list(first.columns)
            if not models:
                models = models_from_filenames(ext_pred_files)
            if not arms:
                arms = arms_from_series(pd.Series([f.name for f in ext_pred_files]))
            patient_counts = []
            for f in ext_pred_files[:12]:
                try:
                    tmp = pd.read_csv(f, usecols=["patient_id"])
                    patient_counts.append(tmp["patient_id"].nunique())
                except Exception:
                    pass
            unique_patients = ",".join(map(str, sorted(set(patient_counts)))) if patient_counts else "NA"

        rows.append(
            {
                "path": str(p),
                "modality": infer_modality(p, df),
                "analysis": infer_analysis(p, df, cfg),
                "include_mlp_config": cfg.get("include_mlp", "NA") if cfg else "NA",
                "mlp_included": "MLP" in models,
                "predictions_present": preds_all.exists() or bool(list(p.glob("preds/preds_test__*.csv"))),
                "external_predictions_present": bool(ext_pred_files),
                "external_summary_present": external_summary.exists(),
                "models_present": ",".join(models) if models else "NA",
                "arms_present": ",".join(arms) if arms else "NA",
                "unique_patients": unique_patients,
                "columns_available": ", ".join(cols) if cols else "NA",
                "results_file": str(results) if results.exists() else "NA",
                "run_config": str(p / "run_config.json") if (p / "run_config.json").exists() else "NA",
            }
        )
    return pd.DataFrame(rows)


def check_prediction_frame(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"patient_id", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{source}: missing required columns {sorted(missing)}")
    nulls = {c: int(df[c].isna().sum()) for c in required}
    if any(nulls.values()):
        raise RuntimeError(f"{source}: NaN values present in required columns: {nulls}")
    probs = pd.to_numeric(df["y_prob"], errors="coerce")
    if probs.isna().any():
        raise RuntimeError(f"{source}: non-numeric y_prob values present")
    if not np.isfinite(probs.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{source}: non-finite probabilities present")
    if ((probs < 0) | (probs > 1)).any():
        raise RuntimeError(f"{source}: probabilities outside [0, 1] present")
    df = df.copy()
    df["y_prob"] = probs.astype(float)
    label_classes = pd.Series(df["y_true"]).drop_duplicates()
    if len(label_classes) != 2:
        raise RuntimeError(f"{source}: expected exactly two outcome classes, found {label_classes.tolist()}")
    if df["patient_id"].duplicated().any():
        dup = df[df["patient_id"].duplicated(keep=False)].sort_values("patient_id")
        conflicts = []
        for pid, group in dup.groupby("patient_id", sort=False):
            pairs = group[["y_true", "y_prob"]].drop_duplicates()
            if len(pairs) > 1:
                conflicts.append(str(pid))
        if conflicts:
            raise RuntimeError(f"{source}: conflicting duplicate predictions for patient IDs {conflicts[:10]}")
        df = df.drop_duplicates(subset=["patient_id", "y_true", "y_prob"]).copy()
    return df.sort_values("patient_id").reset_index(drop=True)


def class_counts(y: pd.Series) -> tuple[Any, int, Any, int]:
    counts = y.value_counts(dropna=False).sort_index()
    if len(counts) != 2:
        raise RuntimeError(f"Expected exactly two classes, found {counts.to_dict()}")
    neg_label = counts.index[0]
    pos_label = counts.index[1]
    return pos_label, int(counts.iloc[1]), neg_label, int(counts.iloc[0])


def auc_score(y: pd.Series | np.ndarray, prob: pd.Series | np.ndarray) -> float:
    return float(roc_auc_score(y, prob))


def stable_seed(*parts: Any) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def bootstrap_count_matrix(y: np.ndarray, seed: int) -> np.ndarray:
    n = len(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise RuntimeError(f"Expected two classes for bootstrap, found {classes.tolist()}")
    pos_mask = y == classes[1]
    neg_mask = ~pos_mask
    chunks: list[np.ndarray] = []
    valid = 0
    probs = np.full(n, 1.0 / n)
    while valid < N_BOOT:
        batch_size = max(1024, N_BOOT - valid)
        counts = rng.multinomial(n, probs, size=batch_size).astype(float)
        keep = (counts[:, pos_mask].sum(axis=1) > 0) & (counts[:, neg_mask].sum(axis=1) > 0)
        if not keep.any():
            continue
        counts = counts[keep]
        take = min(len(counts), N_BOOT - valid)
        chunks.append(counts[:take])
        valid += take
    return np.vstack(chunks)


def weighted_auc_from_counts(y: np.ndarray, prob: np.ndarray, counts: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    pos_label = classes[1]
    pos = (y == pos_label)
    order = np.argsort(prob, kind="mergesort")
    scores = prob[order]
    pos_sorted = pos[order]
    counts_sorted = counts[:, order]
    pos_counts = counts_sorted * pos_sorted
    neg_counts = counts_sorted * (~pos_sorted)
    total_pos = pos_counts.sum(axis=1)
    total_neg = neg_counts.sum(axis=1)
    if (total_pos <= 0).any() or (total_neg <= 0).any():
        raise RuntimeError("Invalid bootstrap count matrix contains one-class samples")

    contrib = np.zeros(counts.shape[0], dtype=float)
    neg_before = np.zeros(counts.shape[0], dtype=float)
    start = 0
    n = len(scores)
    while start < n:
        end = start + 1
        while end < n and scores[end] == scores[start]:
            end += 1
        pos_w = pos_counts[:, start:end].sum(axis=1)
        neg_w = neg_counts[:, start:end].sum(axis=1)
        contrib += pos_w * neg_before + 0.5 * pos_w * neg_w
        neg_before += neg_w
        start = end
    return contrib / (total_pos * total_neg)


def bootstrap_auc(y: np.ndarray, prob: np.ndarray, seed: int) -> tuple[float, float, np.ndarray]:
    counts = bootstrap_count_matrix(y, seed)
    arr = weighted_auc_from_counts(y, prob, counts)
    if len(arr):
        sklearn_first = float(roc_auc_score(y, prob, sample_weight=counts[0]))
        if not math.isclose(float(arr[0]), sklearn_first, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Weighted bootstrap AUC cross-check failed: {arr[0]} vs sklearn {sklearn_first}")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), arr


def bootstrap_delta(
    y: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    counts = bootstrap_count_matrix(y, seed)
    arr1 = weighted_auc_from_counts(y, prob1, counts)
    arr2 = weighted_auc_from_counts(y, prob2, counts)
    if len(arr1):
        sklearn_first_1 = float(roc_auc_score(y, prob1, sample_weight=counts[0]))
        sklearn_first_2 = float(roc_auc_score(y, prob2, sample_weight=counts[0]))
        if not math.isclose(float(arr1[0]), sklearn_first_1, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Weighted paired bootstrap AUC cross-check failed for arm1")
        if not math.isclose(float(arr2[0]), sklearn_first_2, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Weighted paired bootstrap AUC cross-check failed for arm2")
    arr = arr1 - arr2
    low = float(np.percentile(arr, 2.5))
    high = float(np.percentile(arr, 97.5))
    p_two = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
    return low, high, min(p_two, 1.0), arr


def internal_result_row(spec: CohortSpec, arm: str, model: str) -> pd.Series:
    results = pd.read_csv(spec.summary_file)
    mask = (results["arm_id"].astype(str) == arm_id(arm)) & (results["model"].astype(str) == model)
    matches = results[mask]
    if len(matches) != 1:
        raise RuntimeError(f"{spec.summary_file}: expected one row for {arm_id(arm)} {model}, found {len(matches)}")
    return matches.iloc[0]


def external_summary_row(spec: CohortSpec, arm: str, model: str) -> pd.Series:
    summary = pd.read_csv(spec.summary_file)
    run_label = f"{spec.modality}_axisA_common_neutral_mlp"
    mask = (
        (summary["analysis_run"].astype(str) == run_label)
        & (summary["arm_id"].astype(str) == arm_id(arm))
        & (summary["model"].astype(str) == model)
    )
    matches = summary[mask]
    if len(matches) != 1:
        raise RuntimeError(f"{spec.summary_file}: expected one row for {run_label} {arm_id(arm)} {model}, found {len(matches)}")
    return matches.iloc[0]


def prediction_source(spec: CohortSpec, arm: str, model: str) -> Path:
    if spec.validation == "internal held-out":
        return spec.run_dir / "preds" / f"preds_test__{model}__{filename_arm_id(arm)}.csv"
    assert spec.external_dir is not None
    return spec.external_dir / f"external_predictions__bundle__{model}__{filename_arm_id(arm)}.csv"


def load_prediction(spec: CohortSpec, arm: str, model: str) -> tuple[pd.DataFrame, Path, pd.Series]:
    pred_file = prediction_source(spec, arm, model)
    if not pred_file.exists():
        raise RuntimeError(f"Missing prediction file: {pred_file}")
    df = pd.read_csv(pred_file)
    if "arm_id" in df.columns:
        observed_arms = sorted(df["arm_id"].dropna().astype(str).unique())
        if observed_arms != [arm_id(arm)]:
            raise RuntimeError(f"{pred_file}: expected only {arm_id(arm)}, found {observed_arms}")
    if "model" in df.columns:
        observed_models = sorted(df["model"].dropna().astype(str).unique())
        if observed_models != [model]:
            raise RuntimeError(f"{pred_file}: expected only {model}, found {observed_models}")
    df = check_prediction_frame(df, pred_file)
    if spec.validation == "internal held-out":
        summary_row = internal_result_row(spec, arm, model)
        expected_auc = float(summary_row["test_auc"])
    else:
        summary_row = external_summary_row(spec, arm, model)
        expected_auc = float(summary_row["auc"])
    observed_auc = auc_score(df["y_true"], df["y_prob"])
    if not math.isclose(observed_auc, expected_auc, rel_tol=0, abs_tol=AUC_TOL):
        raise RuntimeError(
            f"{pred_file}: recomputed AUC {observed_auc:.15f} does not match "
            f"{spec.summary_file} value {expected_auc:.15f}"
        )
    return df, pred_file, summary_row


def selected_predictions() -> dict[tuple[str, str, str, str], pd.DataFrame]:
    data: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for spec in FINAL_SPECS:
        for arm in ARMS:
            for model in MODELS:
                df, _, _ = load_prediction(spec, arm, model)
                data[(spec.modality, spec.validation, arm, model)] = df[["patient_id", "y_true", "y_prob"]].copy()
    return data


def compute_ci_table() -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], pd.DataFrame], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    pred_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for spec in FINAL_SPECS:
        for arm in ARMS:
            for model in MODELS:
                df, pred_file, summary_row = load_prediction(spec, arm, model)
                y = df["y_true"].to_numpy()
                prob = df["y_prob"].to_numpy(dtype=float)
                auc = auc_score(y, prob)
                boot_seed = stable_seed(SEED, spec.modality, spec.validation, model)
                ci_low, ci_high, _ = bootstrap_auc(y, prob, boot_seed)
                pos_label, n_pos, neg_label, n_neg = class_counts(df["y_true"])
                rows.append(
                    {
                        "modality": spec.modality,
                        "validation": spec.validation,
                        "analysis": ANALYSIS,
                        "arm": arm,
                        "model": model,
                        "n": int(len(df)),
                        "n_positive": n_pos,
                        "n_negative": n_neg,
                        "auc": auc,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n_bootstrap": N_BOOT,
                        "source_prediction_file": str(pred_file),
                        "source_results_file": str(spec.summary_file),
                    }
                )
                pred_cache[(spec.modality, spec.validation, arm, model)] = df[["patient_id", "y_true", "y_prob"]].copy()
                if len(df) != int(summary_row["n_test"] if spec.validation == "internal held-out" else summary_row["n"]):
                    notes.append(f"{pred_file}: row count {len(df)} differs from summary count")
                notes.append(
                    f"{spec.modality} {spec.validation} {arm} {model}: AUC matched "
                    f"{spec.summary_file.name}; n={len(df)}, positive_label={pos_label}, "
                    f"negative_label={neg_label}, positives={n_pos}, negatives={n_neg}."
                )
    return pd.DataFrame(rows), pred_cache, notes


def compute_paired_table(pred_cache: dict[tuple[str, str, str, str], pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    comparisons = (("VS", "NR"), ("RS", "NR"), ("FK", "NR"), ("VS", "RS"))
    for modality in ("CT", "MRI"):
        for validation in ("internal held-out", "external"):
            for model in MODELS:
                for arm1, arm2 in comparisons:
                    d1 = pred_cache[(modality, validation, arm1, model)].rename(
                        columns={"y_true": "y_true_1", "y_prob": "y_prob_1"}
                    )
                    d2 = pred_cache[(modality, validation, arm2, model)].rename(
                        columns={"y_true": "y_true_2", "y_prob": "y_prob_2"}
                    )
                    merged = d1.merge(d2, on="patient_id", how="inner", validate="one_to_one")
                    if len(merged) == 0:
                        raise RuntimeError(f"{modality} {validation} {model} {arm1}-{arm2}: no paired patients")
                    disagree = merged[merged["y_true_1"] != merged["y_true_2"]]
                    if len(disagree):
                        raise RuntimeError(
                            f"{modality} {validation} {model} {arm1}-{arm2}: labels disagree for "
                            f"{len(disagree)} paired patients"
                        )
                    y = merged["y_true_1"].to_numpy()
                    p1 = merged["y_prob_1"].to_numpy(dtype=float)
                    p2 = merged["y_prob_2"].to_numpy(dtype=float)
                    auc1 = auc_score(y, p1)
                    auc2 = auc_score(y, p2)
                    delta = auc1 - auc2
                    boot_seed = stable_seed(SEED, modality, validation, model, arm1, arm2)
                    low, high, p_exploratory, _ = bootstrap_delta(y, p1, p2, boot_seed)
                    rows.append(
                        {
                            "modality": modality,
                            "validation": validation,
                            "analysis": ANALYSIS,
                            "model": model,
                            "comparison": f"{arm1} - {arm2}",
                            "n_paired": int(len(merged)),
                            "auc_arm1": auc1,
                            "auc_arm2": auc2,
                            "delta_auc": delta,
                            "ci95_low": low,
                            "ci95_high": high,
                            "bootstrap_p_exploratory": p_exploratory,
                            "n_bootstrap": N_BOOT,
                        }
                    )
                    notes.append(f"{modality} {validation} {model} {arm1}-{arm2}: paired n={len(merged)}, labels agree.")
    return pd.DataFrame(rows), notes


def identical_prediction_notes(pred_cache: dict[tuple[str, str, str, str], pd.DataFrame], ci_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for modality in ("CT", "MRI"):
        for validation in ("internal held-out", "external"):
            for model in MODELS:
                pairs = []
                for i, arm1 in enumerate(ARMS):
                    for arm2 in ARMS[i + 1 :]:
                        d1 = pred_cache[(modality, validation, arm1, model)].sort_values("patient_id").reset_index(drop=True)
                        d2 = pred_cache[(modality, validation, arm2, model)].sort_values("patient_id").reset_index(drop=True)
                        same_ids = d1["patient_id"].astype(str).tolist() == d2["patient_id"].astype(str).tolist()
                        same_y = d1["y_true"].tolist() == d2["y_true"].tolist()
                        same_prob = np.array_equal(d1["y_prob"].to_numpy(float), d2["y_prob"].to_numpy(float))
                        if same_ids and same_y and same_prob:
                            pairs.append(f"{arm1}={arm2}")
                            r1 = ci_df[(ci_df.modality == modality) & (ci_df.validation == validation) & (ci_df.model == model) & (ci_df.arm == arm1)].iloc[0]
                            r2 = ci_df[(ci_df.modality == modality) & (ci_df.validation == validation) & (ci_df.model == model) & (ci_df.arm == arm2)].iloc[0]
                            if not (
                                math.isclose(r1.auc, r2.auc, abs_tol=0, rel_tol=0)
                                and math.isclose(r1.ci95_low, r2.ci95_low, abs_tol=0, rel_tol=0)
                                and math.isclose(r1.ci95_high, r2.ci95_high, abs_tol=0, rel_tol=0)
                            ):
                                raise RuntimeError(
                                    f"Identical vectors for {modality} {validation} {model} {arm1}/{arm2} "
                                    "did not produce identical AUC/CI"
                                )
                if pairs:
                    notes.append(f"{modality} {validation} {model}: identical prediction vectors found for {', '.join(pairs)}.")
    return notes


def sample_size_notes(ci_df: pd.DataFrame) -> list[str]:
    expected = {
        ("CT", "internal held-out"): 135,
        ("MRI", "internal held-out"): 45,
        ("CT", "external"): 46,
        ("MRI", "external"): 63,
    }
    notes: list[str] = []
    for key, exp in expected.items():
        modality, validation = key
        vals = sorted(ci_df[(ci_df.modality == modality) & (ci_df.validation == validation)]["n"].unique())
        if vals == [exp]:
            notes.append(f"{modality} {validation}: n={exp}, as expected.")
        else:
            notes.append(f"{modality} {validation}: expected approximately {exp}; observed n values {vals} from source files.")
    return notes


def format_auc(value: float) -> str:
    return f"{value:.3f}"


def manuscript_tables(ci_df: pd.DataFrame) -> str:
    labels = [
        ("A) CT Internal Common-Neutral", "CT", "internal held-out"),
        ("B) MRI Internal Common-Neutral", "MRI", "internal held-out"),
        ("C) CT External Common-Neutral", "CT", "external"),
        ("D) MRI External Common-Neutral", "MRI", "external"),
    ]
    out: list[str] = []
    for title, modality, validation in labels:
        sub = ci_df[(ci_df.modality == modality) & (ci_df.validation == validation)].copy()
        sub["arm"] = pd.Categorical(sub["arm"], categories=list(ARMS), ordered=True)
        sub["model"] = pd.Categorical(sub["model"], categories=list(MODELS), ordered=True)
        sub = sub.sort_values(["arm", "model"])
        out.append(f"### {title}")
        out.append("")
        out.append("| arm | LR | XGB | MLP |")
        out.append("| --- | --- | --- | --- |")
        for arm in ARMS:
            cells = [arm]
            for model in MODELS:
                row = sub[(sub.arm == arm) & (sub.model == model)].iloc[0]
                cells.append(f"{format_auc(row.auc)} (95% CI {format_auc(row.ci95_low)}–{format_auc(row.ci95_high)})")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
        display[col] = display[col].str.replace("|", "\\|", regex=False)
    widths = {
        col: max(len(str(col)), int(display[col].map(len).max()) if len(display) else 0)
        for col in display.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in display.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = []
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(row[col].ljust(widths[col]) for col in display.columns) + " |")
    return "\n".join([header, sep] + rows)


def compact_table(ci_df: pd.DataFrame) -> str:
    sub = ci_df.copy()
    sub["auc_ci"] = sub.apply(
        lambda r: f"{r['auc']:.3f} ({r['ci95_low']:.3f}-{r['ci95_high']:.3f})",
        axis=1,
    )
    sub["arm"] = pd.Categorical(sub["arm"], categories=list(ARMS), ordered=True)
    sub["model"] = pd.Categorical(sub["model"], categories=list(MODELS), ordered=True)
    sub = sub.sort_values(["modality", "validation", "arm", "model"])
    return markdown_table(sub[["modality", "validation", "arm", "model", "n", "n_positive", "n_negative", "auc_ci"]])


def selected_run_text() -> str:
    lines = [
        "- CT internal: " + str(FINAL_MODELING / "CT_axisA_common_neutral_mlp"),
        "- MRI internal: " + str(FINAL_MODELING / "MRI_axisA_common_neutral_mlp"),
        "- CT external: " + str(FINAL_EXTERNAL / "CT_MLP_common_neutral" / "CT_axisA_common_neutral_mlp"),
        "- MRI external: " + str(FINAL_EXTERNAL / "MRI_MLP_common_neutral" / "MRI_axisA_common_neutral_mlp"),
    ]
    return "\n".join(lines)


def write_audit(
    inventory: pd.DataFrame,
    ci_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    auc_notes: list[str],
    paired_notes: list[str],
    identical_notes: list[str],
) -> None:
    final_dirs = {str(s.run_dir) for s in FINAL_SPECS}
    final_dirs.update(str(s.external_dir) for s in FINAL_SPECS if s.external_dir is not None)
    conflicts = inventory[~inventory["path"].isin(final_dirs)].copy()
    selected = inventory[inventory["path"].isin(final_dirs)].copy()
    lines: list[str] = []
    lines.append("# Paper 1 NUEVOS Axis A Bootstrap ROC AUC CI Audit")
    lines.append("")
    lines.append("Post-hoc uncertainty estimation from already generated prediction files only. No model fitting, feature selection, hyperparameter optimization, or source result modification was performed.")
    lines.append("")
    lines.append("## Selected Final Runs")
    lines.append("")
    lines.append(selected_run_text())
    lines.append("")
    lines.append("Selection rationale: the selected runs are under `NUEVOS/MODELING` and `NUEVOS/extvalidation`, use `COMMON_NEUTRAL`, include MLP (`include_mlp=true`) while retaining LR and XGB rows, and were generated by the Paper 1 Axis A-only workflow. The source inventories contain only the four geometry arms NR/RS/VS/FK with `B=OFF, N=OFF`, and `run_config.json` has `include_combat_arms=false`, so the analyzed data flow is the baseline geometry-only comparison with `C=OFF`.")
    lines.append("")
    lines.append("## Candidate Axis A Runs Found")
    lines.append("")
    inv_show = inventory.copy()
    for col in ("columns_available",):
        inv_show[col] = inv_show[col].astype(str).str.replace("|", "\\|", regex=False)
    lines.append(markdown_table(inv_show))
    lines.append("")
    lines.append("## Duplicated Or Conflicting Runs")
    lines.append("")
    lines.append("The inventory contains older or non-selected mirrors, including `modelado_axisA_* and other superseded/non-selected paths. These were not mixed with the selected `NUEVOS` Axis A-only common-neutral MLP-inclusive runs.")
    if not conflicts.empty:
        conflict_summary = conflicts[["path", "modality", "analysis", "models_present", "arms_present", "unique_patients"]].copy()
        lines.append("")
        lines.append(markdown_table(conflict_summary))
    lines.append("")
    lines.append("## AUC Agreement And Cohort Checks")
    lines.append("")
    for note in auc_notes:
        lines.append(f"- {note}")
    for note in sample_size_notes(ci_df):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("All selected prediction files contained exactly two outcome classes, finite probabilities in `[0, 1]`, and no NaN values in required fields. Duplicate patient IDs, if encountered, were required to be exact duplicate records; conflicting duplicate predictions would abort the run.")
    lines.append("")
    lines.append("## Identical Prediction Vector Checks")
    lines.append("")
    if identical_notes:
        for note in identical_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No identical full patient-level prediction vectors were found among the selected arm pairs.")
    lines.append("")
    lines.append("## Bootstrap Methods")
    lines.append("")
    lines.append(f"- Bootstrap unit: patient/case ID.")
    lines.append(f"- Replicates: {N_BOOT} valid bootstrap samples per marginal AUC and paired delta.")
    lines.append(f"- Seed: {SEED}.")
    lines.append("- Invalid bootstrap samples containing only one outcome class were rejected and resampled until the requested valid replicate count was reached.")
    lines.append("- Observed ROC AUC and source-file agreement checks were computed with `sklearn.metrics.roc_auc_score`; bootstrap replicates used exact patient bootstrap count weights and a vectorized weighted AUC formula cross-checked against sklearn `sample_weight` calculations.")
    lines.append("- Marginal intervals are percentile 95% confidence intervals from the 2.5th and 97.5th bootstrap percentiles.")
    lines.append("")
    lines.append("These confidence intervals quantify uncertainty in discrimination on the evaluated cohort. They do not validate calibration. They do not establish superiority of VS over other preprocessing strategies. Overlap or non-overlap of marginal confidence intervals was not used as a formal paired comparison; paired bootstrap delta analyses are reported separately and p-values are exploratory.")
    lines.append("")
    lines.append("## Paired Comparisons")
    lines.append("")
    for note in paired_notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(markdown_table(paired_df))
    lines.append("")
    lines.append("## Compact AUC CI Table")
    lines.append("")
    lines.append(compact_table(ci_df))
    lines.append("")
    lines.append("## Manuscript-Ready Tables")
    lines.append("")
    lines.append(manuscript_tables(ci_df))
    (OUTDIR / "axisA_auc_bootstrap_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inventory = candidate_inventory()
    ci_df, pred_cache, auc_notes = compute_ci_table()
    paired_df, paired_notes = compute_paired_table(pred_cache)
    identical_notes = identical_prediction_notes(pred_cache, ci_df)

    ci_df.to_csv(OUTDIR / "axisA_auc_bootstrap_ci.csv", index=False)
    paired_df.to_csv(OUTDIR / "axisA_auc_paired_differences.csv", index=False)
    write_audit(inventory, ci_df, paired_df, auc_notes, paired_notes, identical_notes)

    print("Generated:")
    print(OUTDIR / "axisA_auc_bootstrap_ci.csv")
    print(OUTDIR / "axisA_auc_paired_differences.csv")
    print(OUTDIR / "axisA_auc_bootstrap_audit.md")
    print(Path(__file__).resolve())
    print("")
    print("Selected final runs:")
    print(selected_run_text())
    print("")
    print("Compact AUC CI table:")
    print(compact_table(ci_df))


if __name__ == "__main__":
    main()
