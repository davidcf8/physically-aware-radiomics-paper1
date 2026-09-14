#!/usr/bin/env bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="<REPO_PARENT>/LIDL_lung/scripts/analisis/PAPER1/modeling.py"
OUTDIR="<REPO_PARENT>/LIDL_lung/study_results/NUEVOS/MODELING_SEQ_20260622_2310"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

FAILED=()

run_step() {
  local label="$1"
  shift
  local log="${LOGDIR}/${label}.log"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] START ${label}"
  echo "Command: python3 -u ${SCRIPT} $*" > "${log}"

  python3 -u "${SCRIPT}" "$@" >> "${log}" 2>&1
  local rc=$?

  if [[ ${rc} -eq 0 ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] OK    ${label}"
  else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] FAIL  ${label} rc=${rc}"
    FAILED+=("${label}:${rc}")
  fi

  sleep 5
}

run_step "01_CT_axisA_common_neutral" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir "${OUTDIR}" \
  --analysis_label CT_axisA_common_neutral \
  --modality CT \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

run_step "02_CT_axisA_per_arm_ensemble" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir "${OUTDIR}" \
  --analysis_label CT_axisA_per_arm_ensemble \
  --modality CT \
  --axisA_only \
  --drop_shape \
  --feature_mode PER_ARM_ENSEMBLE \
  --per_arm_top_k 30 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

run_step "03_CT_axisA_full_features" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir "${OUTDIR}" \
  --analysis_label CT_axisA_full_features \
  --modality CT \
  --axisA_only \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --inner_cv_splits 5 \
  --seed 1337

run_step "04_CT_axisA_common_neutral_mlp" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/ready \
  --outdir "${OUTDIR}" \
  --analysis_label CT_axisA_common_neutral_mlp \
  --modality CT \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --include_mlp \
  --seed 1337

run_step "05_MRI_axisA_common_neutral" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir "${OUTDIR}" \
  --analysis_label MRI_axisA_common_neutral \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

run_step "06_MRI_axisA_per_arm_ensemble" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir "${OUTDIR}" \
  --analysis_label MRI_axisA_per_arm_ensemble \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode PER_ARM_ENSEMBLE \
  --per_arm_top_k 30 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --seed 1337

run_step "07_MRI_axisA_full_features" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir "${OUTDIR}" \
  --analysis_label MRI_axisA_full_features \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode FULL_FEATURES \
  --inner_cv_splits 5 \
  --seed 1337

run_step "08_MRI_axisA_common_neutral_mlp" \
  --ready_dir <REPO_PARENT>/LIDL_lung/results/MRIready \
  --outdir "${OUTDIR}" \
  --analysis_label MRI_axisA_common_neutral_mlp \
  --modality MRI \
  --target_col pCR \
  --axisA_only \
  --drop_shape \
  --feature_mode COMMON_NEUTRAL \
  --common_top_k 30 \
  --per_arm_top_k 30 \
  --min_common_arm_agreement 4 \
  --ensemble_min_votes 4 \
  --inner_cv_splits 5 \
  --include_mlp \
  --seed 1337

summary="${LOGDIR}/summary.txt"
{
  echo "Finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All modeling steps completed successfully."
  else
    echo "Failed steps:"
    printf '%s\n' "${FAILED[@]}"
  fi
} | tee "${summary}"

if [[ ${#FAILED[@]} -eq 0 ]]; then
  exit 0
fi
exit 1
