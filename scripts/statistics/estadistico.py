#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radiomics statistics pipeline for factorial design:
- Axis A: geometry      {NR, RS, VS, FK}
- Axis B: intensity     modality-dependent
- Axis C: ComBat        {OFF, COMBAT}

Philosophy:
1) Export full feature-level results first.
2) Then generate curated summaries.
3) Never lose information by over-collapsing too early.
4) Intragroup stability should be interpreted mainly with:
   - CV_benign / CV_malignant
   - rMAD_benign / rMAD_malignant

Author: adapted and cleaned for full + curated statistical export.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


# ============================================================
# Constants
# ============================================================
A_LEVELS = ["NR", "RS", "VS", "FK"]
C_LEVELS = ["OFF", "COMBAT"]

REQUIRED_COLS = []

EXCLUDE_COLS = {
    "ID", "malignancy_score", "malignancy_label",
    "patient_id", "annotation_id", "nodule_id", "subject_id",
    "case_id", "method", "arm", "bio_group", "A", "B", "C",
    "Manufacturer", "ConvolutionKernel", "batch", "split",
    "geometry_mode", "intensity_mode",
    "diameter_mm",
    "slice_thickness", "dicom_slice_thickness", "spacing_between_slices",
    "z_spacing_real", "pixel_spacing_x", "pixel_spacing_y",
    "dicom_pixel_spacing_x", "dicom_pixel_spacing_y",
    "rows", "columns",
    "patient_sex", "patient_age",
    "acquisition_date", "study_date",
    "kvp", "reconstruction_diameter", "exposure",
}

EXCLUDE_PREFIXES = ("diagnostics_",)

BENIGN_LABELS = {1, 2}
MALIGN_LABELS = {4, 5}

TEXTURE_FAMILIES = {"glcm", "glrlm", "glszm", "gldm", "ngtdm"}

PRIMARY_BASELINE_LABEL_CT = "no_density"
PRIMARY_BASELINE_LABEL_MRI = "density"

PAIRWISE_CORE = [
    ("NR", "VS"),
    ("NR", "RS"),
    ("RS", "VS"),
    ("FK", "NR"),
    ("FK", "VS"),
    ("FK", "RS"),
]


# ============================================================
# Basic helpers
# ============================================================
def safe_median(s: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(s), errors="coerce").dropna().values
    return float(np.median(vals)) if len(vals) > 0 else np.nan


def safe_mean(s: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(s), errors="coerce").dropna().values
    return float(np.mean(vals)) if len(vals) > 0 else np.nan


def bh_correction(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = ~np.isnan(pvals)
    if mask.sum() == 0:
        return out
    out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return out


def robust_minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    vmin = valid.min()
    vmax = valid.max()
    if vmax == vmin:
        out.loc[valid.index] = 1.0
        return out
    z = (valid - vmin) / (vmax - vmin)
    if not higher_is_better:
        z = 1.0 - z
    out.loc[valid.index] = z
    return out


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
        return df

    if "patient_id" in df.columns and "annotation_id" in df.columns:
        df["ID"] = df["patient_id"].astype(str) + "__ann_" + df["annotation_id"].astype(str)
        return df

    if "patient_id" in df.columns and "nodule_id" in df.columns:
        df["ID"] = df["patient_id"].astype(str) + "__nod_" + df["nodule_id"].astype(str)
        return df

    if "subject_id" in df.columns and "annotation_id" in df.columns:
        df["ID"] = df["subject_id"].astype(str) + "__ann_" + df["annotation_id"].astype(str)
        return df

    if "patient_id" in df.columns:
        df["ID"] = df["patient_id"].astype(str)
        return df

    raise ValueError(
        "Could not construct unique ID. Expected one of: "
        "ID, or patient_id+annotation_id, or patient_id+nodule_id, "
        "or subject_id+annotation_id."
    )


def group_label(m):
    if pd.isna(m):
        return np.nan
    m = int(m)
    if m in BENIGN_LABELS:
        return "benign"
    if m in MALIGN_LABELS:
        return "malignant"
    return np.nan


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = []

    valid_tokens = (
        "_firstorder_",
        "_shape_",
        "_glcm_",
        "_glrlm_",
        "_glszm_",
        "_gldm_",
        "_ngtdm_",
    )

    for c in numeric_cols:
        if c in EXCLUDE_COLS:
            continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(tok in c.lower() for tok in valid_tokens):
            feat_cols.append(c)

    return sorted(feat_cols)


def infer_feature_family(feature: str) -> str:
    f = feature.lower()
    if "_shape_" in f:
        return "shape"
    if "_firstorder_" in f:
        return "firstorder"
    if "_glcm_" in f:
        return "glcm"
    if "_glrlm_" in f:
        return "glrlm"
    if "_glszm_" in f:
        return "glszm"
    if "_gldm_" in f:
        return "gldm"
    if "_ngtdm_" in f:
        return "ngtdm"
    return "other"


def infer_image_type(feature: str) -> str:
    f = feature.lower()
    if f.startswith("original_"):
        return "original"
    if f.startswith("gradient_"):
        return "gradient"
    if f.startswith("logarithm_"):
        return "logarithm"
    if f.startswith("exponential_"):
        return "exponential"
    if f.startswith("square_"):
        return "square"
    if f.startswith("squareroot_"):
        return "squareroot"
    if f.startswith("log-sigma") or f.startswith("log_"):
        return "log"
    if f.startswith("wavelet-"):
        return "wavelet"
    if f.startswith("lbp-"):
        return "lbp"
    return "other"


def build_feature_metadata(feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for f in feature_cols:
        fam = infer_feature_family(f)
        img = infer_image_type(f)
        rows.append({
            "feature": f,
            "feature_family": fam,
            "image_type": img,
            "is_texture": fam in TEXTURE_FAMILIES,
        })
    return pd.DataFrame(rows)


def attach_feature_metadata(df: pd.DataFrame, feature_meta: pd.DataFrame) -> pd.DataFrame:
    if "feature" not in df.columns:
        return df
    return df.merge(feature_meta, on="feature", how="left")


def aligned_wide(df_long: pd.DataFrame, feature: str, methods: List[str]) -> pd.DataFrame:
    wide = df_long.pivot_table(index="ID", columns="method", values=feature, aggfunc="first")
    wide = wide.reindex(columns=methods)
    wide = wide.dropna(axis=0, how="any")
    return wide


# ============================================================
# Filename parsing
# ============================================================
def parse_A_from_name(name: str) -> str:
    lower = name.lower()

    # FK / fake isotropic control
    if (
        "fake" in lower
        or "_fk_" in lower
        or lower.startswith("radiomics_fk")
        or lower.startswith("radiomics_mri_fk")
        or "_nr_fake" in lower
    ):
        return "FK"

    # VS / voxel spacing aware
    if (
        "_vs_" in lower
        or "radiomics_vs_" in lower
        or lower.startswith("radiomics_vs")
        or lower.startswith("radiomics_mri_vs")
        or "voxelspacing" in lower
        or "voxel_spacing" in lower
    ):
        return "VS"

    # NR / no resample
    if (
        "no_resample" in lower
        or "_nr_" in lower
        or lower.startswith("radiomics_mri_nr")
        or lower.startswith("radiomics_nr")
    ):
        return "NR"

    # RS / resample
    if (
        "resample" in lower
        or "_rs_" in lower
        or lower.startswith("radiomics_mri_rs")
        or lower.startswith("radiomics_rs")
    ) and "no_resample" not in lower:
        return "RS"

    raise ValueError(f"Could not infer Axis A from filename: {name}")

def parse_B_from_name(name: str, modality: str) -> str:
    lower = name.lower()

    has_density_norm = "density_norm" in lower
    has_density = "density" in lower

    if modality == "CT":
        if has_density_norm:
            return "density_norm"
        elif has_density:
            return "density"
        else:
            return "no_density"

    elif modality == "MRI":
        if has_density_norm:
            return "density_norm"
        elif has_density:
            return "density"
        else:
            return "no_density_norm"

    else:
        raise ValueError(f"Unknown modality: {modality}")

def parse_C_from_parent(parent_name: str) -> str:
    p = parent_name.lower()
    if p == "merged_homogeneous":
        return "OFF"
    if p == "combat_transformed":
        return "COMBAT"
    raise ValueError(f"Unknown parent folder for Axis C: {parent_name}")


def expected_arms(modality: str) -> set:
    if modality == "CT":
        b_levels = ["no_density", "density", "density_norm"]
    else:
        b_levels = ["no_density_norm", "density"]
    return {f"A={A}|B={B}|C={C}" for A, B, C in itertools.product(A_LEVELS, b_levels, C_LEVELS)}


def check_arm_completeness(found_arms: set, modality: str, strict: bool = False):
    exp = expected_arms(modality)
    missing = sorted(exp - found_arms)
    extra = sorted(found_arms - exp)

    if missing or extra:
        msg = []
        if missing:
            msg.append(f"Missing arms ({len(missing)}):\n  " + "\n  ".join(missing))
        if extra:
            msg.append(f"Unexpected arms ({len(extra)}):\n  " + "\n  ".join(extra))
        msg = "\n".join(msg)

        if strict:
            raise RuntimeError("Arm completeness check failed.\n" + msg)
        else:
            print("\n⚠️ Arm completeness warning")
            print(msg)


# ============================================================
# Statistical helpers
# ============================================================
def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, np.nan

    d = y - x
    if np.allclose(d, 0, equal_nan=True):
        return 0.0, 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def within_subject_cv(values: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=float)
    mu = np.mean(v)
    if not np.isfinite(mu) or abs(mu) < eps or v.size < 2:
        return np.nan
    return float(np.std(v, ddof=1) / abs(mu))


def relative_mad(values: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    if not np.isfinite(med) or abs(med) < eps or v.size < 2:
        return np.nan
    mad = np.median(np.abs(v - med))
    return float(mad / abs(med))


def group_cv(values: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    mu = np.mean(v)
    if abs(mu) < eps:
        return np.nan
    return float(np.std(v, ddof=1) / abs(mu))


def group_rmad(values: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    med = np.median(v)
    if abs(med) < eps:
        return np.nan
    mad = np.median(np.abs(v - med))
    return float(mad / abs(med))


def icc_a1_absolute(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        return np.nan

    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan

    mean_rows = X.mean(axis=1, keepdims=True)
    mean_cols = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()

    ss_rows = k * np.sum((mean_rows - grand_mean) ** 2)
    ss_cols = n * np.sum((mean_cols - grand_mean) ** 2)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    if df_rows <= 0 or df_cols <= 0 or df_error <= 0:
        return np.nan

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denom = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    if denom == 0 or np.isnan(denom):
        return np.nan

    return float((ms_rows - ms_error) / denom)


def icc_c1_consistency(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        return np.nan

    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan

    mean_rows = X.mean(axis=1, keepdims=True)
    mean_cols = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()

    ss_rows = k * np.sum((mean_rows - grand_mean) ** 2)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_cols = n * np.sum((mean_cols - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_error = (n - 1) * (k - 1)

    if df_rows <= 0 or df_error <= 0:
        return np.nan

    ms_rows = ss_rows / df_rows
    ms_error = ss_error / df_error

    denom = ms_rows + (k - 1) * ms_error
    if denom == 0 or np.isnan(denom):
        return np.nan

    return float((ms_rows - ms_error) / denom)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0 or np.isnan(pooled):
        return np.nan
    return float((np.mean(y) - np.mean(x)) / np.sqrt(pooled))


def rank_biserial_from_mwu(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return np.nan
    try:
        u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(2 * u / (nx * ny) - 1)
    except Exception:
        return np.nan


def run_binary_inference(g1: np.ndarray, g2: np.ndarray) -> Dict[str, float | str]:
    if len(g1) < 3 or len(g2) < 3:
        return {
            "test": "NA",
            "stat": np.nan,
            "p": np.nan,
            "effect_size": np.nan,
            "effect_type": "NA",
            "normal": np.nan,
            "homoscedastic": np.nan,
            "mean_benign": np.nan,
            "mean_malignant": np.nan,
            "median_benign": np.nan,
            "median_malignant": np.nan,
        }

    try:
        normal = (stats.shapiro(g1)[1] > 0.05) and (stats.shapiro(g2)[1] > 0.05)
    except Exception:
        normal = False

    try:
        homo = stats.levene(g1, g2)[1] > 0.05
    except Exception:
        homo = False

    mean_b = float(np.mean(g1))
    mean_m = float(np.mean(g2))
    med_b = float(np.median(g1))
    med_m = float(np.median(g2))

    if normal and homo:
        try:
            stat, p = stats.ttest_ind(g1, g2, equal_var=True)
        except Exception:
            stat, p = np.nan, np.nan
        eff = cohens_d(g1, g2)
        eff_type = "cohens_d"
        test_name = "t-test"
    else:
        try:
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        eff = rank_biserial_from_mwu(g1, g2)
        eff_type = "rank_biserial"
        test_name = "MWU"

    return {
        "test": test_name,
        "stat": float(stat) if pd.notna(stat) else np.nan,
        "p": float(p) if pd.notna(p) else np.nan,
        "effect_size": float(eff) if pd.notna(eff) else np.nan,
        "effect_type": eff_type,
        "normal": bool(normal),
        "homoscedastic": bool(homo),
        "mean_benign": mean_b,
        "mean_malignant": mean_m,
        "median_benign": med_b,
        "median_malignant": med_m,
    }


def paired_robustness_from_joined(joined: pd.DataFrame, id_to_group: Dict[str, str], min_complete: int):
    n_complete = len(joined)
    if n_complete < min_complete:
        return {
            "n": n_complete,
            "icc_a1_overall": np.nan,
            "icc_c1_overall": np.nan,
            "wcv_median_overall": np.nan,
            "rmad_median_overall": np.nan,
            "icc_a1_benign": np.nan,
            "icc_c1_benign": np.nan,
            "wcv_median_benign": np.nan,
            "rmad_median_benign": np.nan,
            "icc_a1_malignant": np.nan,
            "icc_c1_malignant": np.nan,
            "wcv_median_malignant": np.nan,
            "rmad_median_malignant": np.nan,
        }

    X = joined.values.astype(float)
    groups = np.array([id_to_group.get(i, np.nan) for i in joined.index])

    def subgroup(mask):
        Xs = X[mask]
        if Xs.shape[0] < min_complete:
            return (np.nan, np.nan, np.nan, np.nan)
        icc_a = icc_a1_absolute(Xs)
        icc_c = icc_c1_consistency(Xs)
        wcv = np.nanmedian([within_subject_cv(r) for r in Xs])
        rmad = np.nanmedian([relative_mad(r) for r in Xs])
        return icc_a, icc_c, wcv, rmad

    icc_a_all, icc_c_all, wcv_all, rmad_all = subgroup(np.ones(len(X), dtype=bool))
    icc_a_b, icc_c_b, wcv_b, rmad_b = subgroup(groups == "benign")
    icc_a_m, icc_c_m, wcv_m, rmad_m = subgroup(groups == "malignant")

    return {
        "n": n_complete,
        "icc_a1_overall": icc_a_all,
        "icc_c1_overall": icc_c_all,
        "wcv_median_overall": wcv_all,
        "rmad_median_overall": rmad_all,
        "icc_a1_benign": icc_a_b,
        "icc_c1_benign": icc_c_b,
        "wcv_median_benign": wcv_b,
        "rmad_median_benign": rmad_b,
        "icc_a1_malignant": icc_a_m,
        "icc_c1_malignant": icc_c_m,
        "wcv_median_malignant": wcv_m,
        "rmad_median_malignant": rmad_m,
    }


# ============================================================
# Generic summary writers
# ============================================================
def summarize_discrimination_by_group(df_disc: pd.DataFrame, out_path: Path, group_cols: List[str]):
    rows = []
    for keys, sub in df_disc.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tmp = {col: val for col, val in zip(group_cols, keys)}
        tmp["n_features"] = len(sub)
        tmp["n_signif_fdr_005"] = int((sub["p_fdr"] < 0.05).sum()) if "p_fdr" in sub.columns else np.nan
        tmp["prop_signif_fdr_005"] = float((sub["p_fdr"] < 0.05).mean()) if "p_fdr" in sub.columns else np.nan
        tmp["median_p_fdr"] = safe_median(sub["p_fdr"]) if "p_fdr" in sub.columns else np.nan
        tmp["median_abs_effect"] = safe_median(np.abs(sub["effect_size"])) if "effect_size" in sub.columns else np.nan
        rows.append(tmp)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def summarize_robustness_by_group(df_rob: pd.DataFrame, out_path: Path, group_cols: List[str]):
    rows = []
    for keys, sub in df_rob.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tmp = {col: val for col, val in zip(group_cols, keys)}
        tmp["n_features"] = len(sub)
        tmp["median_icc_a1_overall"] = safe_median(sub["icc_a1_overall"]) if "icc_a1_overall" in sub.columns else np.nan
        tmp["median_icc_c1_overall"] = safe_median(sub["icc_c1_overall"]) if "icc_c1_overall" in sub.columns else np.nan
        tmp["median_wcv_overall"] = safe_median(sub["wcv_median_overall"]) if "wcv_median_overall" in sub.columns else np.nan
        tmp["median_rmad_overall"] = safe_median(sub["rmad_median_overall"]) if "rmad_median_overall" in sub.columns else np.nan
        tmp["median_icc_a1_benign"] = safe_median(sub["icc_a1_benign"]) if "icc_a1_benign" in sub.columns else np.nan
        tmp["median_icc_a1_malignant"] = safe_median(sub["icc_a1_malignant"]) if "icc_a1_malignant" in sub.columns else np.nan
        rows.append(tmp)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def summarize_shift_by_group(df_shift: pd.DataFrame, out_path: Path, group_cols: List[str]):
    rows = []
    for keys, sub in df_shift.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tmp = {col: val for col, val in zip(group_cols, keys)}
        tmp["n_features"] = len(sub)
        tmp["n_changed_fdr_005"] = int((sub["p_fdr"] < 0.05).sum()) if "p_fdr" in sub.columns else np.nan
        tmp["prop_changed_fdr_005"] = float((sub["p_fdr"] < 0.05).mean()) if "p_fdr" in sub.columns else np.nan
        tmp["median_abs_delta"] = safe_median(sub["median_abs_delta"]) if "median_abs_delta" in sub.columns else np.nan
        tmp["median_signed_delta"] = safe_median(sub["median_delta"]) if "median_delta" in sub.columns else np.nan
        rows.append(tmp)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def summarize_within_group_stability_by_group(df_stab: pd.DataFrame, out_path: Path, group_cols: List[str]):
    rows = []
    for keys, sub in df_stab.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tmp = {col: val for col, val in zip(group_cols, keys)}
        tmp["n_features"] = len(sub)
        tmp["median_cv_benign"] = safe_median(sub["cv_benign"]) if "cv_benign" in sub.columns else np.nan
        tmp["median_rmad_benign"] = safe_median(sub["rmad_benign"]) if "rmad_benign" in sub.columns else np.nan
        tmp["median_cv_malignant"] = safe_median(sub["cv_malignant"]) if "cv_malignant" in sub.columns else np.nan
        tmp["median_rmad_malignant"] = safe_median(sub["rmad_malignant"]) if "rmad_malignant" in sub.columns else np.nan
        tmp["median_cv_all"] = safe_median(sub["cv_all"]) if "cv_all" in sub.columns else np.nan
        tmp["median_rmad_all"] = safe_median(sub["rmad_all"]) if "rmad_all" in sub.columns else np.nan
        rows.append(tmp)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def add_composite_score(
    df: pd.DataFrame,
    icc_col: str = "icc_a1_overall",
    wcv_col: str = "wcv_median_overall",
    discrim_col: str = "best_discrimination_score",
    effect_col: str = "best_abs_effect_size"
) -> pd.DataFrame:
    out = df.copy()
    out["score_icc"] = robust_minmax(out[icc_col], higher_is_better=True) if icc_col in out.columns else np.nan
    out["score_wcv"] = robust_minmax(out[wcv_col], higher_is_better=False) if wcv_col in out.columns else np.nan
    out["score_discrimination"] = robust_minmax(out[discrim_col], higher_is_better=True) if discrim_col in out.columns else np.nan
    out["score_effect"] = robust_minmax(out[effect_col], higher_is_better=True) if effect_col in out.columns else np.nan
    out["composite_score"] = out[["score_icc", "score_wcv", "score_discrimination", "score_effect"]].mean(axis=1, skipna=True)
    return out


# ============================================================
# Primary Axis A baseline
# ============================================================
def run_primary_axisA_within_group_stability(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    modality: str,
    min_n_group: int = 5,
):
    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI

    df_main = df_all[(df_all["B"] == baseline_B) & (df_all["C"] == "OFF")].copy()
    df_main["method"] = df_main["A"]

    rows = []
    for method, df_m in df_main.groupby("method"):
        for feature in feature_cols:
            x_all = df_m[feature].dropna().values
            x_b = df_m[df_m["bio_group"] == "benign"][feature].dropna().values
            x_m = df_m[df_m["bio_group"] == "malignant"][feature].dropna().values

            rows.append({
                "method": method,
                "feature": feature,
                "n_all": len(x_all),
                "n_benign": len(x_b),
                "n_malignant": len(x_m),
                "cv_all": group_cv(x_all) if len(x_all) >= min_n_group else np.nan,
                "rmad_all": group_rmad(x_all) if len(x_all) >= min_n_group else np.nan,
                "cv_benign": group_cv(x_b) if len(x_b) >= min_n_group else np.nan,
                "rmad_benign": group_rmad(x_b) if len(x_b) >= min_n_group else np.nan,
                "cv_malignant": group_cv(x_m) if len(x_m) >= min_n_group else np.nan,
                "rmad_malignant": group_rmad(x_m) if len(x_m) >= min_n_group else np.nan,
                "median_all": float(np.median(x_all)) if len(x_all) else np.nan,
                "median_benign": float(np.median(x_b)) if len(x_b) else np.nan,
                "median_malignant": float(np.median(x_m)) if len(x_m) else np.nan,
            })

    out = pd.DataFrame(rows)
    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_within_group_stability_by_method.csv", index=False)

    summarize_within_group_stability_by_group(
        out,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_within_group_stability_summary_by_method_family.csv",
        ["method", "feature_family"]
    )
    summarize_within_group_stability_by_group(
        out,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_within_group_stability_summary_by_method_image_type.csv",
        ["method", "image_type"]
    )


def run_primary_axisA(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    modality: str,
    min_complete: int
):
    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI
    methods = A_LEVELS

    df_main = df_all[(df_all["B"] == baseline_B) & (df_all["C"] == "OFF")].copy()
    df_main["method"] = df_main["A"]

    # -------- Robustness across methods --------
    rob_rows = []
    id_to_group = df_main.drop_duplicates("ID").set_index("ID")["bio_group"].to_dict()

    for feature in feature_cols:
        wide = aligned_wide(df_main, feature, methods)
        stats_dict = paired_robustness_from_joined(wide, id_to_group, min_complete)
        rob_rows.append({"feature": feature, **stats_dict})

    rob_df = pd.DataFrame(rob_rows)
    rob_df = attach_feature_metadata(rob_df, feature_meta)
    rob_df.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_robustness.csv", index=False)

    summarize_robustness_by_group(
        rob_df,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_robustness_summary_by_family.csv",
        ["feature_family"]
    )
    summarize_robustness_by_group(
        rob_df,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_robustness_summary_by_image_type.csv",
        ["image_type"]
    )
    summarize_robustness_by_group(
        rob_df[rob_df["is_texture"] == True].copy(),
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_robustness_summary_textures_by_family.csv",
        ["feature_family"]
    )

    # -------- Friedman + pairwise Wilcoxon --------
    fried_rows = []
    pairwise_rows = []

    for feature in feature_cols:
        wide = aligned_wide(df_main, feature, methods)
        n_complete = len(wide)

        if n_complete < min_complete:
            fried_rows.append({"feature": feature, "n": n_complete, "stat": np.nan, "p": np.nan})
            continue

        try:
            stat, p = stats.friedmanchisquare(*[wide[m].values for m in methods])
        except Exception:
            stat, p = np.nan, np.nan

        fried_rows.append({"feature": feature, "n": n_complete, "stat": stat, "p": p})

        for m1, m2 in itertools.combinations(methods, 2):
            wstat, wp = wilcoxon_paired(wide[m1].values, wide[m2].values)
            delta = wide[m2].values - wide[m1].values
            pairwise_rows.append({
                "feature": feature,
                "n": n_complete,
                "m1": m1,
                "m2": m2,
                "wilcoxon_stat": wstat,
                "p": wp,
                "median_delta": float(np.median(delta)) if len(delta) else np.nan,
                "median_abs_delta": float(np.median(np.abs(delta))) if len(delta) else np.nan,
            })

    fried_df = pd.DataFrame(fried_rows)
    fried_df["p_fdr"] = bh_correction(fried_df["p"].values)
    fried_df = attach_feature_metadata(fried_df, feature_meta)
    fried_df.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_friedman.csv", index=False)

    pairs_df = pd.DataFrame(pairwise_rows)
    pairs_df["p_fdr"] = bh_correction(pairs_df["p"].values)
    pairs_df = attach_feature_metadata(pairs_df, feature_meta)
    pairs_df.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_pairwise_wilcoxon.csv", index=False)

    summarize_shift_by_group(
        pairs_df,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_pairwise_wilcoxon_summary_by_pair_family.csv",
        ["m1", "m2", "feature_family"]
    )

    # -------- Discrimination per method --------
    disc_rows = []
    for method in methods:
        df_m = df_main[df_main["method"] == method].copy()

        for feature in feature_cols:
            g1 = df_m[df_m["bio_group"] == "benign"][feature].dropna().values
            g2 = df_m[df_m["bio_group"] == "malignant"][feature].dropna().values

            res = run_binary_inference(g1, g2)
            disc_rows.append({
                "method": method,
                "feature": feature,
                **res
            })

    disc_df = pd.DataFrame(disc_rows)
    disc_df["p_fdr"] = np.nan
    for method in methods:
        mask = disc_df["method"] == method
        disc_df.loc[mask, "p_fdr"] = bh_correction(disc_df.loc[mask, "p"].values)

    disc_df["abs_effect_size"] = np.abs(disc_df["effect_size"])
    disc_df = attach_feature_metadata(disc_df, feature_meta)
    disc_df.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_discrimination.csv", index=False)

    summarize_discrimination_by_group(
        disc_df,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_discrimination_summary_by_method_family.csv",
        ["method", "feature_family"]
    )
    summarize_discrimination_by_group(
        disc_df,
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_discrimination_summary_by_method_image_type.csv",
        ["method", "image_type"]
    )
    summarize_discrimination_by_group(
        disc_df[disc_df["is_texture"] == True].copy(),
        out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_discrimination_summary_textures_by_method_family.csv",
        ["method", "feature_family"]
    )

    # -------- Integrated summaries (keep, but interpret with caution) --------
    rob_family = rob_df.groupby("feature_family", as_index=False).agg(
        median_icc_a1_overall=("icc_a1_overall", safe_median),
        median_icc_c1_overall=("icc_c1_overall", safe_median),
        median_wcv_overall=("wcv_median_overall", safe_median),
        median_rmad_overall=("rmad_median_overall", safe_median),
    )

    disc_family = disc_df.groupby(["method", "feature_family"], as_index=False).agg(
        n_features=("feature", "count"),
        n_signif_fdr_005=("p_fdr", lambda s: int((s < 0.05).sum())),
        prop_signif_fdr_005=("p_fdr", lambda s: float((s < 0.05).mean())),
        median_p_fdr=("p_fdr", safe_median),
        median_abs_effect=("abs_effect_size", safe_median),
    )

    primary_family = disc_family.merge(rob_family, on="feature_family", how="left")
    primary_family.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_integrated_summary_by_method_family.csv", index=False)

    rob_img = rob_df.groupby("image_type", as_index=False).agg(
        median_icc_a1_overall=("icc_a1_overall", safe_median),
        median_icc_c1_overall=("icc_c1_overall", safe_median),
        median_wcv_overall=("wcv_median_overall", safe_median),
        median_rmad_overall=("rmad_median_overall", safe_median),
    )

    disc_img = disc_df.groupby(["method", "image_type"], as_index=False).agg(
        n_features=("feature", "count"),
        n_signif_fdr_005=("p_fdr", lambda s: int((s < 0.05).sum())),
        prop_signif_fdr_005=("p_fdr", lambda s: float((s < 0.05).mean())),
        median_p_fdr=("p_fdr", safe_median),
        median_abs_effect=("abs_effect_size", safe_median),
    )

    primary_img = disc_img.merge(rob_img, on="image_type", how="left")
    primary_img.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_integrated_summary_by_method_image_type.csv", index=False)

    # -------- Composite rankings (do not over-trust blindly) --------
    disc_best = (
        disc_df.groupby("feature", as_index=False)
        .agg(
            best_p_fdr=("p_fdr", "min"),
            best_abs_effect_size=("abs_effect_size", "max"),
            n_methods_signif=("p_fdr", lambda s: int((s < 0.05).sum())),
        )
    )

    disc_best["best_discrimination_score"] = robust_minmax(
        -np.log10(pd.to_numeric(disc_best["best_p_fdr"], errors="coerce").clip(lower=1e-300)),
        higher_is_better=True
    )

    primary_rank = rob_df.merge(disc_best, on="feature", how="left")
    primary_rank = add_composite_score(primary_rank)
    primary_rank = primary_rank.sort_values("composite_score", ascending=False)
    primary_rank.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_composite_feature_ranking.csv", index=False)

    primary_texture = primary_rank[primary_rank["is_texture"] == True].copy()
    primary_texture.to_csv(out_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_texture_feature_ranking.csv", index=False)


# ============================================================
# Full-arm feature-level exports
# ============================================================
def run_full_discrimination(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path
):
    rows = []

    for (A, B, C), df_arm in df_all.groupby(["A", "B", "C"]):
        for feature in feature_cols:
            g1 = df_arm[df_arm["bio_group"] == "benign"][feature].dropna().values
            g2 = df_arm[df_arm["bio_group"] == "malignant"][feature].dropna().values
            res = run_binary_inference(g1, g2)
            rows.append({
                "A": A, "B": B, "C": C,
                "feature": feature,
                **res
            })

    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan

    for (A, B, C), idx in out.groupby(["A", "B", "C"]).groups.items():
        idx = list(idx)
        out.loc[idx, "p_fdr"] = bh_correction(out.loc[idx, "p"].values)

    out["abs_effect_size"] = np.abs(out["effect_size"])
    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "FULL_discrimination_all_arms.csv", index=False)

    summarize_discrimination_by_group(out, out_dir / "FULL_discrimination_summary_by_arm.csv", ["A", "B", "C"])
    summarize_discrimination_by_group(out, out_dir / "FULL_discrimination_summary_by_arm_family.csv", ["A", "B", "C", "feature_family"])
    summarize_discrimination_by_group(out, out_dir / "FULL_discrimination_summary_by_arm_image_type.csv", ["A", "B", "C", "image_type"])

    tmp = out.copy()
    tmp["disc_score"] = robust_minmax(
        -np.log10(pd.to_numeric(tmp["p_fdr"], errors="coerce").clip(lower=1e-300)),
        higher_is_better=True
    )

    best_rows = []
    for feature, sub in tmp.groupby("feature"):
        sub = sub.sort_values(["disc_score", "abs_effect_size"], ascending=[False, False])
        best = sub.iloc[0]
        best_rows.append({
            "feature": feature,
            "best_A": best["A"],
            "best_B": best["B"],
            "best_C": best["C"],
            "best_test": best["test"],
            "best_p_fdr": best["p_fdr"],
            "best_effect_size": best["effect_size"],
            "best_abs_effect_size": best["abs_effect_size"],
            "feature_family": best["feature_family"],
            "image_type": best["image_type"],
            "is_texture": best["is_texture"],
        })

    pd.DataFrame(best_rows).sort_values(
        ["best_p_fdr", "best_abs_effect_size"],
        ascending=[True, False]
    ).to_csv(out_dir / "FULL_discrimination_best_arm_per_feature.csv", index=False)


def run_full_within_group_stability(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    min_n_group: int = 5,
):
    rows = []
    for (A, B, C), df_arm in df_all.groupby(["A", "B", "C"]):
        for feature in feature_cols:
            x_all = df_arm[feature].dropna().values
            x_b = df_arm[df_arm["bio_group"] == "benign"][feature].dropna().values
            x_m = df_arm[df_arm["bio_group"] == "malignant"][feature].dropna().values

            rows.append({
                "A": A, "B": B, "C": C,
                "feature": feature,
                "n_all": len(x_all),
                "n_benign": len(x_b),
                "n_malignant": len(x_m),
                "cv_all": group_cv(x_all) if len(x_all) >= min_n_group else np.nan,
                "rmad_all": group_rmad(x_all) if len(x_all) >= min_n_group else np.nan,
                "cv_benign": group_cv(x_b) if len(x_b) >= min_n_group else np.nan,
                "rmad_benign": group_rmad(x_b) if len(x_b) >= min_n_group else np.nan,
                "cv_malignant": group_cv(x_m) if len(x_m) >= min_n_group else np.nan,
                "rmad_malignant": group_rmad(x_m) if len(x_m) >= min_n_group else np.nan,
            })

    out = pd.DataFrame(rows)
    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "FULL_within_group_stability_all_arms.csv", index=False)

    summarize_within_group_stability_by_group(
        out,
        out_dir / "FULL_within_group_stability_summary_by_arm.csv",
        ["A", "B", "C"]
    )
    summarize_within_group_stability_by_group(
        out,
        out_dir / "FULL_within_group_stability_summary_by_arm_family.csv",
        ["A", "B", "C", "feature_family"]
    )
    summarize_within_group_stability_by_group(
        out,
        out_dir / "FULL_within_group_stability_summary_by_arm_image_type.csv",
        ["A", "B", "C", "image_type"]
    )

# ============================================================
# Axis B
# ============================================================
def run_axisB_ablation(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    modality: str,
    min_complete: int
):
    if modality == "CT":
        comparisons = [
            ("no_density", "density", "no_density_vs_density"),
            ("density", "density_norm", "density_vs_density_norm"),
            ("no_density", "density_norm", "no_density_vs_density_norm"),
        ]
    else:
        comparisons = [
            ("no_density_norm", "density", "no_density_norm_vs_density"),
        ]

    rows = []

    for A in A_LEVELS:
        for C in C_LEVELS:
            for b1, b2, tag in comparisons:
                df1 = df_all[(df_all["A"] == A) & (df_all["B"] == b1) & (df_all["C"] == C)].copy()
                df2 = df_all[(df_all["A"] == A) & (df_all["B"] == b2) & (df_all["C"] == C)].copy()

                for feature in feature_cols:
                    if df1.empty or df2.empty:
                        rows.append({
                            "A": A, "C": C,
                            "comparison": tag, "B1": b1, "B2": b2,
                            "feature": feature,
                            "n": 0, "test": "NA", "stat": np.nan, "p": np.nan,
                            "median_delta": np.nan, "median_abs_delta": np.nan,
                        })
                        continue

                    s1 = df1.set_index("ID")[feature]
                    s2 = df2.set_index("ID")[feature]
                    joined = pd.concat([s1, s2], axis=1, keys=[b1, b2]).dropna()

                    n_complete = len(joined)
                    if n_complete < min_complete:
                        rows.append({
                            "A": A, "C": C,
                            "comparison": tag, "B1": b1, "B2": b2,
                            "feature": feature,
                            "n": n_complete, "test": "NA", "stat": np.nan, "p": np.nan,
                            "median_delta": np.nan, "median_abs_delta": np.nan,
                        })
                        continue

                    stat, p = wilcoxon_paired(joined[b1].values, joined[b2].values)
                    delta = joined[b2].values - joined[b1].values
                    rows.append({
                        "A": A, "C": C,
                        "comparison": tag, "B1": b1, "B2": b2,
                        "feature": feature,
                        "n": n_complete,
                        "test": "Wilcoxon(paired)",
                        "stat": stat,
                        "p": p,
                        "median_delta": float(np.median(delta)),
                        "median_abs_delta": float(np.median(np.abs(delta))),
                    })

    out = pd.DataFrame(rows)



    if out.empty:
        out.to_csv(out_dir / "AXIS_B_paired_ablation.csv", index=False)
        print("[WARN] AXIS_B_paired_ablation produced no rows. Skipping summaries.")
        return

    out["p_fdr"] = np.nan

    for (A, C, tag), idx in out.groupby(["A", "C", "comparison"]).groups.items():
        idx = list(idx)
        out.loc[idx, "p_fdr"] = bh_correction(out.loc[idx, "p"].values)

    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "AXIS_B_paired_ablation.csv", index=False)

    summarize_shift_by_group(
        out,
        out_dir / "AXIS_B_paired_ablation_summary_by_family.csv",
        ["A", "C", "comparison", "feature_family"]
    )
    summarize_shift_by_group(
        out,
        out_dir / "AXIS_B_paired_ablation_summary_by_image_type.csv",
        ["A", "C", "comparison", "image_type"]
    )


def run_axisB_paired_robustness(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    modality: str,
    min_complete: int
):
    if modality == "CT":
        comparisons = [
            ("no_density", "density", "no_density_vs_density"),
            ("density", "density_norm", "density_vs_density_norm"),
            ("no_density", "density_norm", "no_density_vs_density_norm"),
        ]
    else:
        comparisons = [
            ("no_density_norm", "density", "no_density_norm_vs_density"),
        ]

    rows = []

    for A in A_LEVELS:
        for C in C_LEVELS:
            for b1, b2, tag in comparisons:
                df1 = df_all[(df_all["A"] == A) & (df_all["B"] == b1) & (df_all["C"] == C)].copy()
                df2 = df_all[(df_all["A"] == A) & (df_all["B"] == b2) & (df_all["C"] == C)].copy()

                if df1.empty or df2.empty:
                    continue

                id_to_group = df1.drop_duplicates("ID").set_index("ID")["bio_group"].to_dict()
                if not id_to_group:
                    id_to_group = df2.drop_duplicates("ID").set_index("ID")["bio_group"].to_dict()

                for feature in feature_cols:
                    s1 = df1.set_index("ID")[feature]
                    s2 = df2.set_index("ID")[feature]
                    joined = pd.concat([s1, s2], axis=1, keys=[b1, b2]).dropna()

                    stats_dict = paired_robustness_from_joined(joined, id_to_group, min_complete)
                    rows.append({
                        "A": A,
                        "C": C,
                        "comparison": tag,
                        "B1": b1,
                        "B2": b2,
                        "feature": feature,
                        **stats_dict
                    })

    out = pd.DataFrame(rows)
    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "AXIS_B_paired_robustness.csv", index=False)


    if out.empty:
        print("[WARN] AXIS_B_paired_robustness produced no rows. Skipping summaries.")
        return

    summarize_robustness_by_group(
        out,
        out_dir / "AXIS_B_paired_robustness_summary_by_family.csv",
        ["A", "C", "comparison", "feature_family"]
    )
    summarize_robustness_by_group(
        out,
        out_dir / "AXIS_B_paired_robustness_summary_by_image_type.csv",
        ["A", "C", "comparison", "image_type"]
    )


# ============================================================
# Axis C
# ============================================================
def run_axisC_combat(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    min_complete: int
):
    rows = []

    for (A, B), df_sub in df_all.groupby(["A", "B"]):
        df_off = df_sub[df_sub["C"] == "OFF"].copy()
        df_cb = df_sub[df_sub["C"] == "COMBAT"].copy()

        for feature in feature_cols:
            if df_off.empty or df_cb.empty:
                rows.append({
                    "A": A, "B": B,
                    "feature": feature,
                    "n": 0, "test": "NA", "stat": np.nan, "p": np.nan,
                    "median_delta": np.nan, "median_abs_delta": np.nan,
                })
                continue

            s_off = df_off.set_index("ID")[feature]
            s_cb = df_cb.set_index("ID")[feature]
            joined = pd.concat([s_off, s_cb], axis=1, keys=["OFF", "COMBAT"]).dropna()

            n_complete = len(joined)
            if n_complete < min_complete:
                rows.append({
                    "A": A, "B": B,
                    "feature": feature,
                    "n": n_complete, "test": "NA", "stat": np.nan, "p": np.nan,
                    "median_delta": np.nan, "median_abs_delta": np.nan,
                })
                continue

            stat, p = wilcoxon_paired(joined["OFF"].values, joined["COMBAT"].values)
            delta = joined["COMBAT"].values - joined["OFF"].values
            rows.append({
                "A": A, "B": B,
                "feature": feature,
                "n": n_complete,
                "test": "Wilcoxon(paired)",
                "stat": stat,
                "p": p,
                "median_delta": float(np.median(delta)),
                "median_abs_delta": float(np.median(np.abs(delta))),
            })

    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan

    for (A, B), idx in out.groupby(["A", "B"]).groups.items():
        idx = list(idx)
        out.loc[idx, "p_fdr"] = bh_correction(out.loc[idx, "p"].values)

    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "AXIS_C_OFF_vs_COMBAT_paired.csv", index=False)

    summarize_shift_by_group(
        out,
        out_dir / "AXIS_C_OFF_vs_COMBAT_summary_by_family.csv",
        ["A", "B", "feature_family"]
    )
    summarize_shift_by_group(
        out,
        out_dir / "AXIS_C_OFF_vs_COMBAT_summary_by_image_type.csv",
        ["A", "B", "image_type"]
    )


def run_axisC_combat_robustness(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    min_complete: int
):
    rows = []

    for (A, B), df_sub in df_all.groupby(["A", "B"]):
        df_off = df_sub[df_sub["C"] == "OFF"].copy()
        df_cb = df_sub[df_sub["C"] == "COMBAT"].copy()

        if df_off.empty or df_cb.empty:
            continue

        id_to_group = df_off.drop_duplicates("ID").set_index("ID")["bio_group"].to_dict()
        if not id_to_group:
            id_to_group = df_cb.drop_duplicates("ID").set_index("ID")["bio_group"].to_dict()

        for feature in feature_cols:
            s_off = df_off.set_index("ID")[feature]
            s_cb = df_cb.set_index("ID")[feature]
            joined = pd.concat([s_off, s_cb], axis=1, keys=["OFF", "COMBAT"]).dropna()

            stats_dict = paired_robustness_from_joined(joined, id_to_group, min_complete)
            rows.append({
                "A": A,
                "B": B,
                "feature": feature,
                **stats_dict
            })

    out = pd.DataFrame(rows)
    out = attach_feature_metadata(out, feature_meta)
    out.to_csv(out_dir / "AXIS_C_OFF_vs_COMBAT_robustness.csv", index=False)

    summarize_robustness_by_group(
        out,
        out_dir / "AXIS_C_OFF_vs_COMBAT_robustness_summary_by_family.csv",
        ["A", "B", "feature_family"]
    )
    summarize_robustness_by_group(
        out,
        out_dir / "AXIS_C_OFF_vs_COMBAT_robustness_summary_by_image_type.csv",
        ["A", "B", "image_type"]
    )


# ============================================================
# Texture focus
# ============================================================
def run_texture_focus_block(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    modality: str,
    min_complete: int
):
    texture_features = feature_meta[feature_meta["is_texture"] == True]["feature"].tolist()
    if not texture_features:
        print("\n⚠️ No texture features detected. Skipping texture focus block.")
        return

    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI
    methods = A_LEVELS

    df_main = df_all[(df_all["B"] == baseline_B) & (df_all["C"] == "OFF")].copy()
    df_main["method"] = df_main["A"]

    rob_rows = []
    for feature in texture_features:
        wide = aligned_wide(df_main, feature, methods)
        n_complete = len(wide)

        if n_complete < min_complete:
            rob_rows.append({
                "feature": feature,
                "n": n_complete,
                "icc_a1_overall": np.nan,
                "wcv_median_overall": np.nan,
                "friedman_p": np.nan,
            })
            continue

        try:
            _, p = stats.friedmanchisquare(*[wide[m].values for m in methods])
        except Exception:
            p = np.nan

        rob_rows.append({
            "feature": feature,
            "n": n_complete,
            "icc_a1_overall": icc_a1_absolute(wide.values),
            "wcv_median_overall": np.nanmedian([within_subject_cv(r) for r in wide.values]),
            "friedman_p": p,
        })

    tex_rob = pd.DataFrame(rob_rows)
    tex_rob["friedman_p_fdr"] = bh_correction(tex_rob["friedman_p"].values)
    tex_rob = attach_feature_metadata(tex_rob, feature_meta)

    tex_rob["score_icc"] = robust_minmax(tex_rob["icc_a1_overall"], higher_is_better=True)
    tex_rob["score_wcv"] = robust_minmax(tex_rob["wcv_median_overall"], higher_is_better=False)
    tex_rob["score_friedman"] = robust_minmax(
        -np.log10(pd.to_numeric(tex_rob["friedman_p_fdr"], errors="coerce").clip(lower=1e-300)),
        higher_is_better=True
    )
    tex_rob["texture_robustness_score"] = tex_rob[["score_icc", "score_wcv", "score_friedman"]].mean(axis=1, skipna=True)
    tex_rob = tex_rob.sort_values("texture_robustness_score", ascending=False)
    tex_rob.to_csv(out_dir / "TEXTURE_primary_baseline_robustness_ranking.csv", index=False)

    full_disc = pd.read_csv(out_dir / "FULL_discrimination_all_arms.csv")
    full_disc = full_disc[full_disc["is_texture"] == True].copy()
    full_disc["disc_score"] = robust_minmax(
        -np.log10(pd.to_numeric(full_disc["p_fdr"], errors="coerce").clip(lower=1e-300)),
        higher_is_better=True
    )

    best_rows = []
    for feature, sub in full_disc.groupby("feature"):
        sub = sub.sort_values(["disc_score", "abs_effect_size"], ascending=[False, False])
        best = sub.iloc[0]
        best_rows.append({
            "feature": feature,
            "best_A": best["A"],
            "best_B": best["B"],
            "best_C": best["C"],
            "best_p_fdr": best["p_fdr"],
            "best_effect_size": best["effect_size"],
            "best_abs_effect_size": best["abs_effect_size"],
            "feature_family": best["feature_family"],
            "image_type": best["image_type"],
        })

    tex_best = pd.DataFrame(best_rows).sort_values(["best_p_fdr", "best_abs_effect_size"], ascending=[True, False])
    tex_best.to_csv(out_dir / "TEXTURE_best_discrimination_arm_per_feature.csv", index=False)

    summarize_discrimination_by_group(
        full_disc,
        out_dir / "TEXTURE_discrimination_summary_by_family_arm.csv",
        ["A", "B", "C", "feature_family"]
    )

    tex_final = tex_rob.merge(
        tex_best[["feature", "best_p_fdr", "best_abs_effect_size", "best_A", "best_B", "best_C"]],
        on="feature", how="left"
    )
    tex_final["score_discrimination"] = robust_minmax(
        -np.log10(pd.to_numeric(tex_final["best_p_fdr"], errors="coerce").clip(lower=1e-300)),
        higher_is_better=True
    )
    tex_final["score_effect"] = robust_minmax(tex_final["best_abs_effect_size"], higher_is_better=True)
    tex_final["texture_final_score"] = tex_final[
        ["score_icc", "score_wcv", "score_discrimination", "score_effect"]
    ].mean(axis=1, skipna=True)
    tex_final = tex_final.sort_values("texture_final_score", ascending=False)
    tex_final.to_csv(out_dir / "TEXTURE_final_integrated_ranking.csv", index=False)


# ============================================================
# Curated builders
# ============================================================
def build_axisA_primary_feature_level(full_dir: Path, curated_dir: Path, modality: str):
    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI

    disc = pd.read_csv(full_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_discrimination.csv")
    stab = pd.read_csv(full_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_within_group_stability_by_method.csv")
    rob = pd.read_csv(full_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_robustness.csv")

    keep_disc = [
        "method", "feature", "test", "p", "p_fdr",
        "effect_size", "abs_effect_size",
        "feature_family", "image_type", "is_texture"
    ]
    keep_stab = [
        "method", "feature",
        "cv_benign", "rmad_benign",
        "cv_malignant", "rmad_malignant",
        "cv_all", "rmad_all"
    ]
    keep_rob = [
        "feature",
        "icc_a1_overall", "icc_c1_overall",
        "wcv_median_overall", "rmad_median_overall",
        "icc_a1_benign", "icc_a1_malignant"
    ]

    disc = disc[[c for c in keep_disc if c in disc.columns]].copy()
    stab = stab[[c for c in keep_stab if c in stab.columns]].copy()
    rob = rob[[c for c in keep_rob if c in rob.columns]].copy()

    out = disc.merge(stab, on=["method", "feature"], how="left")
    out = out.merge(rob, on="feature", how="left")
    out.to_csv(curated_dir / "axisA_primary_feature_level.csv", index=False)
    return out


def build_axisA_curated_summaries(axisA_feat: pd.DataFrame, out_dir: Path):
    method_rows = []
    for method, sub in axisA_feat.groupby("method"):
        method_rows.append({
            "method": method,
            "n_features": len(sub),
            "n_signif_fdr_005": int((sub["p_fdr"] < 0.05).sum()),
            "prop_signif_fdr_005": float((sub["p_fdr"] < 0.05).mean()),
            "median_abs_effect": safe_median(sub["abs_effect_size"]),
            "median_cv_benign": safe_median(sub["cv_benign"]),
            "median_rmad_benign": safe_median(sub["rmad_benign"]),
            "median_cv_malignant": safe_median(sub["cv_malignant"]),
            "median_rmad_malignant": safe_median(sub["rmad_malignant"]),
        })
    pd.DataFrame(method_rows).to_csv(out_dir / "axisA_primary_method_summary.csv", index=False)

    fam_rows = []
    for (method, fam), sub in axisA_feat.groupby(["method", "feature_family"]):
        fam_rows.append({
            "method": method,
            "feature_family": fam,
            "n_features": len(sub),
            "n_signif_fdr_005": int((sub["p_fdr"] < 0.05).sum()),
            "prop_signif_fdr_005": float((sub["p_fdr"] < 0.05).mean()),
            "median_abs_effect": safe_median(sub["abs_effect_size"]),
            "median_cv_benign": safe_median(sub["cv_benign"]),
            "median_rmad_benign": safe_median(sub["rmad_benign"]),
            "median_cv_malignant": safe_median(sub["cv_malignant"]),
            "median_rmad_malignant": safe_median(sub["rmad_malignant"]),
            "median_icc_a1_overall": safe_median(sub["icc_a1_overall"]),
            "median_icc_c1_overall": safe_median(sub["icc_c1_overall"]),
            "median_wcv_overall": safe_median(sub["wcv_median_overall"]),
            "median_rmad_overall": safe_median(sub["rmad_median_overall"]),
        })
    pd.DataFrame(fam_rows).to_csv(out_dir / "axisA_primary_family_summary.csv", index=False)

    img_rows = []
    for (method, img), sub in axisA_feat.groupby(["method", "image_type"]):
        img_rows.append({
            "method": method,
            "image_type": img,
            "n_features": len(sub),
            "n_signif_fdr_005": int((sub["p_fdr"] < 0.05).sum()),
            "prop_signif_fdr_005": float((sub["p_fdr"] < 0.05).mean()),
            "median_abs_effect": safe_median(sub["abs_effect_size"]),
            "median_cv_benign": safe_median(sub["cv_benign"]),
            "median_rmad_benign": safe_median(sub["rmad_benign"]),
            "median_cv_malignant": safe_median(sub["cv_malignant"]),
            "median_rmad_malignant": safe_median(sub["rmad_malignant"]),
            "median_icc_a1_overall": safe_median(sub["icc_a1_overall"]),
            "median_icc_c1_overall": safe_median(sub["icc_c1_overall"]),
            "median_wcv_overall": safe_median(sub["wcv_median_overall"]),
            "median_rmad_overall": safe_median(sub["rmad_median_overall"]),
        })
    pd.DataFrame(img_rows).to_csv(out_dir / "axisA_primary_image_summary.csv", index=False)

    tex = axisA_feat[axisA_feat["is_texture"] == True].copy()
    tex_rows = []
    for (method, fam), sub in tex.groupby(["method", "feature_family"]):
        tex_rows.append({
            "method": method,
            "feature_family": fam,
            "n_features": len(sub),
            "n_signif_fdr_005": int((sub["p_fdr"] < 0.05).sum()),
            "prop_signif_fdr_005": float((sub["p_fdr"] < 0.05).mean()),
            "median_abs_effect": safe_median(sub["abs_effect_size"]),
            "median_cv_benign": safe_median(sub["cv_benign"]),
            "median_rmad_benign": safe_median(sub["rmad_benign"]),
            "median_cv_malignant": safe_median(sub["cv_malignant"]),
            "median_rmad_malignant": safe_median(sub["rmad_malignant"]),
            "median_icc_a1_overall": safe_median(sub["icc_a1_overall"]),
            "median_icc_c1_overall": safe_median(sub["icc_c1_overall"]),
            "median_wcv_overall": safe_median(sub["wcv_median_overall"]),
            "median_rmad_overall": safe_median(sub["rmad_median_overall"]),
        })
    pd.DataFrame(tex_rows).to_csv(out_dir / "texture_primary_family_summary.csv", index=False)


def build_axisA_primary_pairwise_summary(full_dir: Path, curated_dir: Path, modality: str):
    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI
    pairs = pd.read_csv(full_dir / f"PRIMARY_A_baseline_{baseline_B}_OFF_pairwise_wilcoxon.csv")

    keep = []
    for m1, m2 in PAIRWISE_CORE:
        sub = pairs[(pairs["m1"] == m1) & (pairs["m2"] == m2)].copy()
        if sub.empty:
            sub = pairs[(pairs["m1"] == m2) & (pairs["m2"] == m1)].copy()
            if not sub.empty:
                sub["m1"] = m1
                sub["m2"] = m2
        if not sub.empty:
            keep.append(sub)

    if not keep:
        return

    pairs = pd.concat(keep, axis=0, ignore_index=True)

    rows = []
    for (m1, m2, fam), sub in pairs.groupby(["m1", "m2", "feature_family"]):
        rows.append({
            "m1": m1,
            "m2": m2,
            "feature_family": fam,
            "n_features": len(sub),
            "n_changed_fdr_005": int((sub["p_fdr"] < 0.05).sum()),
            "prop_changed_fdr_005": float((sub["p_fdr"] < 0.05).mean()),
            "median_abs_delta": safe_median(sub["median_abs_delta"]),
            "median_signed_delta": safe_median(sub["median_delta"]),
        })

    pd.DataFrame(rows).to_csv(curated_dir / "axisA_primary_pairwise_summary.csv", index=False)


def build_texture_candidate_features(out_dir: Path):
    axisA = pd.read_csv(out_dir / "axisA_primary_feature_level.csv")
    tex = axisA[axisA["is_texture"] == True].copy()
    vs = tex[tex["method"] == "VS"].copy()

    if vs.empty:
        pd.DataFrame().to_csv(out_dir / "texture_candidate_features.csv", index=False)
        return

    vs["candidate_score"] = 0.0
    vs["candidate_score"] += robust_minmax(-np.log10(pd.to_numeric(vs["p_fdr"], errors="coerce").clip(lower=1e-300)), True).fillna(0)
    vs["candidate_score"] += robust_minmax(vs["abs_effect_size"], True).fillna(0)
    vs["candidate_score"] += robust_minmax(vs["cv_benign"], False).fillna(0)
    vs["candidate_score"] += robust_minmax(vs["cv_malignant"], False).fillna(0)
    vs["candidate_score"] += robust_minmax(vs["icc_a1_overall"], True).fillna(0)
    vs["candidate_score"] = vs["candidate_score"] / 5.0

    out = vs.sort_values(["candidate_score", "abs_effect_size"], ascending=[False, False]).copy()
    cols = [
        "feature", "feature_family", "image_type",
        "p_fdr", "abs_effect_size",
        "cv_benign", "cv_malignant",
        "rmad_benign", "rmad_malignant",
        "icc_a1_overall", "icc_c1_overall",
        "wcv_median_overall", "rmad_median_overall",
        "candidate_score"
    ]
    cols = [c for c in cols if c in out.columns]
    out[cols].head(50).to_csv(out_dir / "texture_candidate_features.csv", index=False)


def build_statistical_recommendation_summary(curated_dir: Path, full_dir: Path, modality: str):
    fam = pd.read_csv(curated_dir / "axisA_primary_family_summary.csv")
    full_arm = pd.read_csv(full_dir / "FULL_discrimination_summary_by_arm.csv")

    baseline_B = PRIMARY_BASELINE_LABEL_CT if modality == "CT" else PRIMARY_BASELINE_LABEL_MRI

    rows = []

    baseline = full_arm[(full_arm["B"] == baseline_B) & (full_arm["C"] == "OFF")].copy()
    if not baseline.empty:
        baseline = baseline.sort_values(["prop_signif_fdr_005", "median_abs_effect"], ascending=[False, False])
        top = baseline.iloc[0]
        rows.append({
            "category": "baseline_best_discrimination",
            "value": f"A={top['A']}|B={top['B']}|C={top['C']}",
            "note": "Best aggregate baseline discrimination"
        })

    for fam_name in ["glcm", "gldm", "glrlm", "glszm", "ngtdm"]:
        sub = fam[fam["feature_family"] == fam_name].copy()
        if sub.empty:
            continue

        vs = sub[sub["method"] == "VS"]
        nr = sub[sub["method"] == "NR"]
        rs = sub[sub["method"] == "RS"]

        if vs.empty or nr.empty or rs.empty:
            continue

        v = vs.iloc[0]
        n = nr.iloc[0]
        r = rs.iloc[0]

        note = []
        if v["prop_signif_fdr_005"] >= min(n["prop_signif_fdr_005"], r["prop_signif_fdr_005"]):
            note.append("VS preserves discrimination")
        if (v["median_cv_benign"] <= n["median_cv_benign"]) and (v["median_cv_malignant"] <= n["median_cv_malignant"]):
            note.append("VS improves intragroup stability vs NR")

        rows.append({
            "category": f"family_{fam_name}",
            "value": "VS",
            "note": "; ".join(note) if note else "No clear VS advantage"
        })

    rows.append({
        "category": "audit_glszm",
        "value": "YES",
        "note": "GLSZM behaves discordantly and should be technically audited before strong conclusions"
    })

    rows.append({
        "category": "fk_control",
        "value": "artifact-risk",
        "note": "FK useful as negative control, not as recommended extraction strategy"
    })

    pd.DataFrame(rows).to_csv(curated_dir / "statistical_recommendation_summary.csv", index=False)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Radiomics factorial statistics pipeline (full + curated).")
    ap.add_argument("--root_dir", type=str, required=True, help="Root dir containing merged_homogeneous/ and combat_transformed/")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--modality", type=str, required=True, choices=["CT", "MRI"], help="Modality")
    ap.add_argument("--min_complete", type=int, default=5, help="Minimum complete paired samples")
    ap.add_argument("--strict_arms", action="store_true", help="Fail if expected arms are missing")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    modality = args.modality.upper()

    out_dir.mkdir(parents=True, exist_ok=True)

    full_dir = out_dir / "full_exports"
    curated_dir = out_dir / "curated_exports"

    full_dir.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = [
        root_dir / "merged_homogeneous",
        root_dir / "combat_transformed",
    ]

    csv_paths = []
    for d in input_dirs:
        if d.exists():
            csv_paths.extend(sorted(d.glob("*.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {root_dir}")

    records = []
    arm_inventory = []

    for fp in csv_paths:
        A = parse_A_from_name(fp.name)
        B = parse_B_from_name(fp.name, modality=modality)
        C = parse_C_from_parent(fp.parent.name)

        df = pd.read_csv(fp)

        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{fp.name} missing required columns: {missing_cols}")

        df = ensure_case_id(df)
        df = df.copy()
        
        if modality == "CT":
            df["malignancy_score"] = pd.to_numeric(df["malignancy_score"], errors="coerce").astype("Int64")
            df["bio_group"] = df["malignancy_score"].apply(group_label)
            df = df[df["bio_group"].isin(["benign", "malignant"])].copy()

        elif modality == "MRI":
            if "response_label" in df.columns:
                df["malignancy_label"] = pd.to_numeric(df["response_label"], errors="coerce")
            elif "pCR" in df.columns:
                df["malignancy_label"] = pd.to_numeric(df["pCR"], errors="coerce")
            elif "malignancy_label" in df.columns:
                df["malignancy_label"] = pd.to_numeric(df["malignancy_label"], errors="coerce")
            else:
                raise ValueError(f"{fp.name} missing MRI outcome column: expected response_label, pCR, or malignancy_label")

            df = df[df["malignancy_label"].isin([0, 1])].copy()
            df["malignancy_label"] = df["malignancy_label"].astype(int)
            df["bio_group"] = df["malignancy_label"].map({0: "benign", 1: "malignant"})

        df["A"] = A
        df["B"] = B
        df["C"] = C
        df["arm"] = f"A={A}|B={B}|C={C}"

        records.append(df)
        arm_inventory.append({
            "file": str(fp),
            "parent_dir": fp.parent.name,
            "A": A,
            "B": B,
            "C": C,
            "n_rows": len(df),
        })

    df_all = pd.concat(records, axis=0, ignore_index=True)
    feature_cols = get_feature_cols(df_all)
    feature_meta = build_feature_metadata(feature_cols)

    pd.DataFrame(arm_inventory).sort_values(["C", "A", "B", "file"]).to_csv(out_dir / "arm_inventory.csv", index=False)
    feature_meta.to_csv(out_dir / "feature_metadata.csv", index=False)

    found_arms = set(df_all["arm"].unique().tolist())
    check_arm_completeness(found_arms, modality=modality, strict=args.strict_arms)

    print(f"\nLoaded files: {len(csv_paths)}")
    print(f"Detected numeric feature columns: {len(feature_cols)}")
    print(f"Total stacked rows: {len(df_all)}")
    print(f"Modality mode: {modality}")

    summary_rows = []
    for (A, B, C), df_sub in df_all.groupby(["A", "B", "C"]):
        summary_rows.append({
            "A": A,
            "B": B,
            "C": C,
            "n_rows": len(df_sub),
            "n_ids": df_sub["ID"].nunique(),
            "n_benign": int((df_sub["bio_group"] == "benign").sum()),
            "n_malignant": int((df_sub["bio_group"] == "malignant").sum()),
        })
    pd.DataFrame(summary_rows).sort_values(["C", "A", "B"]).to_csv(out_dir / "dataset_summary_by_arm.csv", index=False)

    # Full / raw exports
    run_primary_axisA(df_all, feature_cols, feature_meta, full_dir, modality, args.min_complete)
    run_primary_axisA_within_group_stability(df_all, feature_cols, feature_meta, full_dir, modality, args.min_complete)

    run_full_discrimination(df_all, feature_cols, feature_meta, full_dir)
    run_full_within_group_stability(df_all, feature_cols, feature_meta, full_dir, args.min_complete)

    run_axisB_ablation(df_all, feature_cols, feature_meta, full_dir, modality, args.min_complete)
    run_axisB_paired_robustness(df_all, feature_cols, feature_meta, full_dir, modality, args.min_complete)

    if (df_all["C"] == "COMBAT").any():
        run_axisC_combat(df_all, feature_cols, feature_meta, full_dir, args.min_complete)
        run_axisC_combat_robustness(df_all, feature_cols, feature_meta, full_dir, args.min_complete)
    else:
        print("\n⚠️ No C=COMBAT arms found. Skipping Axis C ComBat analyses.")

    run_texture_focus_block(df_all, feature_cols, feature_meta, full_dir, modality, args.min_complete)

    # Curated exports
    axisA_feat = build_axisA_primary_feature_level(full_dir, curated_dir, modality)
    build_axisA_curated_summaries(axisA_feat, curated_dir)
    build_axisA_primary_pairwise_summary(full_dir, curated_dir, modality)
    build_texture_candidate_features(curated_dir)
    build_statistical_recommendation_summary(curated_dir, full_dir, modality)

    print("\n✅ Done. Main outputs written to:")
    print(out_dir)
    print("\nMost important files:")
    print(" - feature_metadata.csv")
    print(" - dataset_summary_by_arm.csv")
    print(" - PRIMARY_A_baseline_*_OFF_discrimination.csv")
    print(" - PRIMARY_A_baseline_*_OFF_robustness.csv")
    print(" - PRIMARY_A_baseline_*_OFF_within_group_stability_by_method.csv")
    print(" - FULL_discrimination_all_arms.csv")
    print(" - FULL_within_group_stability_all_arms.csv")
    print(" - AXIS_B_paired_ablation.csv")
    print(" - AXIS_B_paired_robustness.csv")
    print(" - AXIS_C_OFF_vs_COMBAT_paired.csv")
    print(" - AXIS_C_OFF_vs_COMBAT_robustness.csv")
    print(" - TEXTURE_final_integrated_ranking.csv")
    print(" - axisA_primary_feature_level.csv")
    print(" - statistical_recommendation_summary.csv")


if __name__ == "__main__":
    main()

    """python <REPO_PARENT>/LIDL_lung/scripts/analisis/estadistico.py \
  --root_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --out_dir <REPO_PARENT>/LIDL_lung/study_results/estadisticos \
  --modality CT \
  --strict_arms

  python study_statistics.py \
  --root_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --out_dir <REPO_PARENT>/LIDL_lung/study_results \
  --modality MRI \
  --strict_arms
  """
