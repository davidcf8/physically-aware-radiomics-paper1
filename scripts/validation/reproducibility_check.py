#!/usr/bin/env python3
"""Lightweight release checks for the Paper 1 Axis A reproducibility package.

This script does not retrain models. It validates the packaged derived feature
CSVs and authoritative reference outputs included in the public release.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "derived_features"
REF = REPO / "results" / "reference_outputs"
ARMS = ("NR", "RS", "VS", "FK")


def status(name: str, ok: bool, detail: str = "") -> bool:
    label = "PASS" if ok else "FAIL"
    print(f"{label}\t{name}\t{detail}")
    return ok


def read_feature_csv(modality: str, cohort: str, arm: str) -> pd.DataFrame:
    return pd.read_csv(DATA / modality / cohort / "merged_homogeneous" / f"radiomics_{modality}_{arm}.csv")


def check_cohorts() -> bool:
    ok = True
    expected = {
        ("CT", "internal"): (673, {"train": 538, "test": 135}),
        ("MRI", "internal"): (208, {"train": 163, "test": 45}),
        ("CT", "external"): (46, {"external": 46}),
        ("MRI", "external"): (63, {"external": 63}),
    }
    for modality, cohort in expected:
        counts = []
        split_counts = []
        feature_counts = []
        for arm in ARMS:
            df = read_feature_csv(modality, cohort, arm)
            counts.append(df["patient_id"].nunique())
            split_counts.append(df["split"].value_counts().to_dict())
            feature_counts.append(sum(c.startswith(("original_", "gradient_")) for c in df.columns))
        n_expected, split_expected = expected[(modality, cohort)]
        ok &= status(
            f"{modality} {cohort} cohort",
            set(counts) == {n_expected} and all(s == split_expected for s in split_counts),
            f"n={counts}; splits={split_counts[0]}; radiomic_cols={feature_counts}",
        )
    return ok


def check_predictive_features() -> bool:
    ok = True
    expected = {
        "CT": {"NR": 5, "RS": 8, "VS": 12, "FK": 9},
        "MRI": {"NR": 16, "RS": 13, "VS": 12, "FK": 10},
    }
    for modality, exp in expected.items():
        run = REF / "modeling" / f"{modality}_axisA_per_arm_ensemble" / "selected_features"
        got = {}
        for arm in ARMS:
            p = run / f"selected_features__A-{arm}__B-OFF__N-OFF__C-OFF__PER_ARM_ENSEMBLE.csv"
            got[arm] = len(pd.read_csv(p))
        ok &= status(f"{modality} final per-arm predictive feature counts", got == exp, str(got))
    return ok


def check_recurrence() -> bool:
    ok = True
    checks = [
        ("CT relaxed recurrence", "ensemble_axisA_CT_relaxed", {"NR": 7, "RS": 11, "VS": 6, "FK": 7}),
        ("MRI relaxed recurrence", "ensemble_MRI_relax", {"NR": 21, "RS": 26, "VS": 20, "FK": 22}),
        ("CT strict recurrence", "ensemble_axisA_CT", {"NR": 0, "RS": 1, "VS": 0, "FK": 2}),
        ("MRI strict recurrence", "ensemble_axisA_MRI", {"NR": 0, "RS": 9, "VS": 1, "FK": 3}),
    ]
    for label, dname, exp in checks:
        df = pd.read_csv(REF / "feature_selection" / dname / "master_arm_summary.csv")
        got = {r["A"]: int(r["n_stable_features"]) for _, r in df.iterrows()}
        ok &= status(label, got == exp, str(got))
    return ok


def check_bootstrap() -> bool:
    ok = True
    ci = pd.read_csv(REF / "bootstrap" / "axisA_auc_bootstrap_ci.csv")
    ok &= status("bootstrap CI rows", len(ci) == 48, f"rows={len(ci)}")
    ok &= status("bootstrap CI n", set(ci["n_bootstrap"]) == {10000}, f"n_bootstrap={sorted(ci['n_bootstrap'].unique())}")
    expected_auc = {
        ("CT", "internal held-out", "NR", "LR"): 0.762888888889,
        ("CT", "external", "RS", "XGB"): 0.284499054820,
        ("MRI", "internal held-out", "VS", "XGB"): 0.380184331797,
        ("MRI", "external", "FK", "MLP"): 0.674747474747,
    }
    for key, exp in expected_auc.items():
        modality, validation, arm, model = key
        row = ci[
            (ci["modality"] == modality)
            & (ci["validation"] == validation)
            & (ci["arm"] == arm)
            & (ci["model"] == model)
        ]
        got = float(row.iloc[0]["auc"]) if len(row) else float("nan")
        ok &= status(f"AUC spot-check {key}", abs(got - exp) < 1e-10, f"{got:.12f}")

    paired = pd.read_csv(REF / "bootstrap" / "axisA_auc_paired_differences.csv")
    internal = paired[paired["validation"] == "internal held-out"].copy()
    excludes = ~((internal["ci95_low"] <= 0) & (internal["ci95_high"] >= 0))
    ok &= status("internal paired comparison count", len(internal) == 24, f"rows={len(internal)}")
    ok &= status("internal paired CIs include/exclude zero", int((~excludes).sum()) == 19 and int(excludes.sum()) == 5,
                 f"include_zero={int((~excludes).sum())}; exclude_zero={int(excludes.sum())}")
    return ok


def main() -> int:
    checks = [
        check_cohorts(),
        check_predictive_features(),
        check_recurrence(),
        check_bootstrap(),
    ]
    if all(checks):
        print("PASS\tall_release_checks\tPaper 1 Axis A release checks passed")
        return 0
    print("FAIL\tall_release_checks\tOne or more checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
