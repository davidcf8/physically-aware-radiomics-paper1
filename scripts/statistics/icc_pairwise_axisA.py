#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise ICC analysis for Axis A radiomics comparisons.

Goal
-----
Compute feature-wise ICC for PAIRS of methods instead of a single global ICC
across NR/RS/VS/FK all together.

This is the correct approach if you want to answer questions like:
- Is VS closer to NR than RS is?
- Does VS preserve native features better than resampling?
- Is FK behaving as a negative control?

Design
------
Axis A:
    NR = no resample
    RS = isotropic resampling
    VS = voxel-spacing-aware
    FK = fake isotropic control

Axis B:
    CT:
        no_density / density / density_norm
    MRI:
        no_density_norm / density_norm

Axis C:
    OFF / COMBAT

Default main analysis:
    baseline B + OFF
    CT  -> B = no_density
    MRI -> B = no_density_norm

Outputs
-------
Feature-level:
    - pairwise_icc_feature_level.csv

Summaries:
    - pairwise_icc_summary_by_pair.csv
    - pairwise_icc_summary_by_pair_family.csv
    - pairwise_icc_summary_by_pair_image_type.csv
    - pairwise_icc_summary_by_pair_family_group.csv

Focused summaries:
    - pairwise_icc_NR_reference_summary.csv
    - pairwise_icc_texture_NR_reference_summary.csv

Author
------
OpenAI + user-guided adaptation
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================
A_LEVELS = ["NR", "RS", "VS", "FK"]
PAIRWISE_CORE = [
    ("NR", "VS"),
    ("NR", "RS"),
    ("NR", "FK"),
    ("RS", "VS"),
    ("FK", "VS"),
    ("FK", "RS"),
]

BENIGN_LABELS = {1, 2}
MALIGN_LABELS = {4, 5}

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

TEXTURE_FAMILIES = {"glcm", "glrlm", "glszm", "gldm", "ngtdm"}


# ============================================================
# Helpers
# ============================================================
def safe_median(x) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    return float(np.median(vals)) if len(vals) else np.nan


def safe_mean(x) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    return float(np.mean(vals)) if len(vals) else np.nan


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

    raise ValueError("Could not build ID column.")


def group_label(m):
    if pd.isna(m):
        return np.nan
    m = int(m)
    if m in BENIGN_LABELS:
        return "benign"
    if m in MALIGN_LABELS:
        return "malignant"
    return np.nan

def ensure_binary_outcome(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    df = df.copy()

    if modality == "CT":
        if "malignancy_score" not in df.columns:
            raise ValueError("CT mode requires malignancy_score.")
        df["malignancy_score"] = pd.to_numeric(df["malignancy_score"], errors="coerce").astype("Int64")
        df["bio_group"] = df["malignancy_score"].apply(group_label)
        df = df[df["bio_group"].isin(["benign", "malignant"])].copy()
        return df

    # MRI: pCR / response endpoint
    for col in ["pCR", "response_label", "malignancy_label"]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            df["response_label"] = y
            df = df[df["response_label"].isin([0, 1])].copy()
            df["response_label"] = df["response_label"].astype(int)
            df["bio_group"] = df["response_label"].map({0: "non_pCR", 1: "pCR"})
            return df

    raise ValueError("MRI mode requires one of: pCR, response_label, malignancy_label.")

def parse_A_from_name(name: str) -> str:
    lower = name.lower()

    if (
        "fake" in lower
        or "_fk_" in lower
        or "_fake_" in lower
        or lower.startswith("radiomics_fk")
        or lower.startswith("radiomics_mri_fk")
        or lower.startswith("radiomics_mri_nr_fake")
    ):
        return "FK"

    if (
        "no_resample" in lower
        or "_nr_" in lower
        or lower.startswith("radiomics_mri_nr")
        or lower.startswith("radiomics_nr")
    ):
        return "NR"

    if (
        "_vs_" in lower
        or "radiomics_vs_" in lower
        or lower.startswith("radiomics_vs")
        or lower.startswith("radiomics_mri_vs")
        or "voxelspacing" in lower
        or "voxel_spacing" in lower
    ):
        return "VS"

    if (
        ("resample" in lower and "no_resample" not in lower)
        or "_rs_" in lower
        or lower.startswith("radiomics_mri_rs")
        or lower.startswith("radiomics_rs")
    ):
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


# ============================================================
# ICC functions
# ============================================================
def icc_a1_absolute(X: np.ndarray) -> float:
    """
    ICC(A,1): two-way mixed, absolute agreement, single measure.
    For pairwise comparisons, X shape = (n_subjects, 2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 2:
        return np.nan
    if np.isnan(X).any():
        return np.nan

    n, k = X.shape

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
    """
    ICC(C,1): two-way mixed, consistency, single measure.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 2:
        return np.nan
    if np.isnan(X).any():
        return np.nan

    n, k = X.shape

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


def mean_abs_diff(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[mask] - x[mask])))


def median_abs_diff(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.median(np.abs(y[mask] - x[mask])))


def mean_signed_diff(y: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y[mask] - x[mask]))


# ============================================================
# Summaries
# ============================================================
def summarize_icc(df: pd.DataFrame, out_path: Path, group_cols: List[str]):
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_features"] = len(sub)
        row["median_icc_a1"] = safe_median(sub["icc_a1"])
        row["median_icc_c1"] = safe_median(sub["icc_c1"])
        row["mean_icc_a1"] = safe_mean(sub["icc_a1"])
        row["mean_icc_c1"] = safe_mean(sub["icc_c1"])
        row["median_mean_abs_diff"] = safe_median(sub["mean_abs_diff"])
        row["median_median_abs_diff"] = safe_median(sub["median_abs_diff"])
        row["median_bias_signed"] = safe_median(sub["mean_signed_diff"])
        row["n_icc_a1_ge_090"] = int((pd.to_numeric(sub["icc_a1"], errors="coerce") >= 0.90).sum())
        row["prop_icc_a1_ge_090"] = float((pd.to_numeric(sub["icc_a1"], errors="coerce") >= 0.90).mean())
        row["n_icc_a1_ge_075"] = int((pd.to_numeric(sub["icc_a1"], errors="coerce") >= 0.75).sum())
        row["prop_icc_a1_ge_075"] = float((pd.to_numeric(sub["icc_a1"], errors="coerce") >= 0.75).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================
# Main pairwise ICC computation
# ============================================================
def compute_pairwise_icc(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    feature_meta: pd.DataFrame,
    out_dir: Path,
    B_value: str,
    C_value: str,
    min_complete: int = 5,
):
    df = df_all[(df_all["B"] == B_value) & (df_all["C"] == C_value)].copy()
    if df.empty:
        raise RuntimeError(f"No data found for B={B_value}, C={C_value}")

    rows = []

    for m1, m2 in PAIRWISE_CORE:
        df1 = df[df["A"] == m1].copy()
        df2 = df[df["A"] == m2].copy()

        if df1.empty or df2.empty:
            continue


        if "benign" in set(df["bio_group"]):
            group0, group1 = "benign", "malignant"
        else:
            group0, group1 = "non_pCR", "pCR"

        ids_b = sorted(set(df1.loc[df1["bio_group"] == group0, "ID"]) & set(df2.loc[df2["bio_group"] == group0, "ID"]))
        ids_m = sorted(set(df1.loc[df1["bio_group"] == group1, "ID"]) & set(df2.loc[df2["bio_group"] == group1, "ID"]))

        ids_all = sorted(set(df1["ID"]) & set(df2["ID"]))

        for feature in feature_cols:
            s1 = df1.set_index("ID")[feature]
            s2 = df2.set_index("ID")[feature]

            joined = pd.concat([s1, s2], axis=1, keys=[m1, m2]).dropna()
            joined_all = joined.loc[joined.index.intersection(ids_all)]
            joined_b = joined.loc[joined.index.intersection(ids_b)]
            joined_m = joined.loc[joined.index.intersection(ids_m)]

            def get_metrics(j: pd.DataFrame):
                n = len(j)
                if n < min_complete:
                    return {
                        "n": n,
                        "icc_a1": np.nan,
                        "icc_c1": np.nan,
                        "mean_abs_diff": np.nan,
                        "median_abs_diff": np.nan,
                        "mean_signed_diff": np.nan,
                    }
                X = j[[m1, m2]].values.astype(float)
                return {
                    "n": n,
                    "icc_a1": icc_a1_absolute(X),
                    "icc_c1": icc_c1_consistency(X),
                    "mean_abs_diff": mean_abs_diff(j[m1].values, j[m2].values),
                    "median_abs_diff": median_abs_diff(j[m1].values, j[m2].values),
                    "mean_signed_diff": mean_signed_diff(j[m2].values, j[m1].values),
                }

            metr_all = get_metrics(joined_all)
            metr_b = get_metrics(joined_b)
            metr_m = get_metrics(joined_m)

            rows.append({
                "m1": m1,
                "m2": m2,
                "feature": feature,

                "n_all": metr_all["n"],
                "icc_a1": metr_all["icc_a1"],
                "icc_c1": metr_all["icc_c1"],
                "mean_abs_diff": metr_all["mean_abs_diff"],
                "median_abs_diff": metr_all["median_abs_diff"],
                "mean_signed_diff": metr_all["mean_signed_diff"],

                "n_benign": metr_b["n"],
                "icc_a1_benign": metr_b["icc_a1"],
                "icc_c1_benign": metr_b["icc_c1"],
                "mean_abs_diff_benign": metr_b["mean_abs_diff"],
                "median_abs_diff_benign": metr_b["median_abs_diff"],
                "mean_signed_diff_benign": metr_b["mean_signed_diff"],

                "n_malignant": metr_m["n"],
                "icc_a1_malignant": metr_m["icc_a1"],
                "icc_c1_malignant": metr_m["icc_c1"],
                "mean_abs_diff_malignant": metr_m["mean_abs_diff"],
                "median_abs_diff_malignant": metr_m["median_abs_diff"],
                "mean_signed_diff_malignant": metr_m["mean_signed_diff"],
            })

    out = pd.DataFrame(rows)
    out = attach_feature_metadata(out, feature_meta)

    # Useful ranking score
    out["score_icc"] = robust_minmax(out["icc_a1"], higher_is_better=True)
    out["score_absdiff"] = robust_minmax(out["median_abs_diff"], higher_is_better=False)
    out["pair_preservation_score"] = out[["score_icc", "score_absdiff"]].mean(axis=1, skipna=True)

    out.to_csv(out_dir / "pairwise_icc_feature_level.csv", index=False)

    summarize_icc(out, out_dir / "pairwise_icc_summary_by_pair.csv", ["m1", "m2"])
    summarize_icc(out, out_dir / "pairwise_icc_summary_by_pair_family.csv", ["m1", "m2", "feature_family"])
    summarize_icc(out, out_dir / "pairwise_icc_summary_by_pair_image_type.csv", ["m1", "m2", "image_type"])

    # By family and group
    fam_group_rows = []
    for (m1, m2, fam), sub in out.groupby(["m1", "m2", "feature_family"]):
        fam_group_rows.append({
            "m1": m1,
            "m2": m2,
            "feature_family": fam,

            "n_features": len(sub),

            "median_icc_a1_all": safe_median(sub["icc_a1"]),
            "median_icc_a1_benign": safe_median(sub["icc_a1_benign"]),
            "median_icc_a1_malignant": safe_median(sub["icc_a1_malignant"]),

            "median_icc_c1_all": safe_median(sub["icc_c1"]),
            "median_icc_c1_benign": safe_median(sub["icc_c1_benign"]),
            "median_icc_c1_malignant": safe_median(sub["icc_c1_malignant"]),

            "median_absdiff_all": safe_median(sub["median_abs_diff"]),
            "median_absdiff_benign": safe_median(sub["median_abs_diff_benign"]),
            "median_absdiff_malignant": safe_median(sub["median_abs_diff_malignant"]),
        })
    pd.DataFrame(fam_group_rows).to_csv(out_dir / "pairwise_icc_summary_by_pair_family_group.csv", index=False)

    # Focused summaries using NR as reference
    nr_ref = out[out["m1"] == "NR"].copy()
    if not nr_ref.empty:
        summarize_icc(nr_ref, out_dir / "pairwise_icc_NR_reference_summary.csv", ["m1", "m2", "feature_family"])

        nr_ref_tex = nr_ref[nr_ref["is_texture"] == True].copy()
        if not nr_ref_tex.empty:
            summarize_icc(
                nr_ref_tex,
                out_dir / "pairwise_icc_texture_NR_reference_summary.csv",
                ["m1", "m2", "feature_family"]
            )

    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Pairwise ICC for Axis A radiomics comparisons")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Root dir containing merged_homogeneous/ and combat_transformed/")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory")
    ap.add_argument("--modality", type=str, required=True, choices=["CT", "MRI"],
                    help="CT or MRI")
    ap.add_argument("--B", type=str, default=None,
                    help="Axis B level to analyze. Default: CT=no_density, MRI=no_density_norm")
    ap.add_argument("--C", type=str, default="OFF", choices=["OFF", "COMBAT"],
                    help="Axis C level to analyze")
    ap.add_argument("--min_complete", type=int, default=5,
                    help="Minimum number of complete paired subjects required")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    modality = args.modality.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.B is not None:
        B_value = args.B
    else:
        B_value = "no_density" if modality == "CT" else "no_density_norm"

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
    inventory = []

    for fp in csv_paths:
        A = parse_A_from_name(fp.name)
        B = parse_B_from_name(fp.name, modality=modality)
        C = parse_C_from_parent(fp.parent.name)

        df = pd.read_csv(fp)
        df = ensure_case_id(df)
        df = ensure_binary_outcome(df, modality=modality)

        df["A"] = A
        df["B"] = B
        df["C"] = C
        df["arm"] = f"A={A}|B={B}|C={C}"

        records.append(df)
        inventory.append({
            "file": str(fp),
            "A": A,
            "B": B,
            "C": C,
            "n_rows": len(df),
            "n_ids": df["ID"].nunique(),
        })

    df_all = pd.concat(records, axis=0, ignore_index=True)
    feature_cols = get_feature_cols(df_all)
    feature_meta = build_feature_metadata(feature_cols)

    pd.DataFrame(inventory).sort_values(["C", "A", "B", "file"]).to_csv(out_dir / "input_inventory.csv", index=False)
    feature_meta.to_csv(out_dir / "feature_metadata.csv", index=False)

    print(f"\nLoaded CSVs: {len(csv_paths)}")
    print(f"Detected feature columns: {len(feature_cols)}")
    print(f"Requested analysis: modality={modality}, B={B_value}, C={args.C}")

    compute_pairwise_icc(
        df_all=df_all,
        feature_cols=feature_cols,
        feature_meta=feature_meta,
        out_dir=out_dir,
        B_value=B_value,
        C_value=args.C,
        min_complete=args.min_complete,
    )

    print("\n✅ Done.")
    print(f"Outputs written to: {out_dir}")
    print("Main files:")
    print(" - pairwise_icc_feature_level.csv")
    print(" - pairwise_icc_summary_by_pair.csv")
    print(" - pairwise_icc_summary_by_pair_family.csv")
    print(" - pairwise_icc_summary_by_pair_image_type.csv")
    print(" - pairwise_icc_summary_by_pair_family_group.csv")
    print(" - pairwise_icc_NR_reference_summary.csv")
    print(" - pairwise_icc_texture_NR_reference_summary.csv")


if __name__ == "__main__":
    main()