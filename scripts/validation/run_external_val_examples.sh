#!/usr/bin/env bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="<REPO_PARENT>/LIDL_lung/scripts/analisis/PAPER1/external_val.py"
MODEL_ROOT="<REPO_PARENT>/LIDL_lung/study_results/NUEVOS/MODELING"
CT_EXTERNAL="<REPO_PARENT>/LIDL_lung/results/CTexvalready"
MRI_EXTERNAL="<REPO_PARENT>/LIDL_lung/results/MRIexvalready"
OUTROOT="<REPO_PARENT>/LIDL_lung/study_results/NUEVOS/extvalidation"
LOGDIR="${OUTROOT}/logs"
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

  sleep 3
}

run_step "01_CT_common_neutral" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${CT_EXTERNAL}" \
  --outdir "${OUTROOT}/CT_common_neutral" \
  --run_glob "CT_axisA_common_neutral*"

run_step "02_CT_per_arm_ensemble" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${CT_EXTERNAL}" \
  --outdir "${OUTROOT}/CT_per_arm_ensemble" \
  --run_glob "CT_axisA_per_arm_ensemble"

run_step "03_CT_MLP_common_neutral" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${CT_EXTERNAL}" \
  --outdir "${OUTROOT}/CT_MLP_common_neutral" \
  --run_glob "CT_axisA_common_neutral_mlp"

run_step "04_MRI_common_neutral" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${MRI_EXTERNAL}" \
  --outdir "${OUTROOT}/MRI_common_neutral" \
  --target_col pathological_complete_response \
  --run_glob "MRI_axisA_common_neutral*"

run_step "05_MRI_per_arm_ensemble" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${MRI_EXTERNAL}" \
  --outdir "${OUTROOT}/MRI_per_arm_ensemble" \
  --target_col pathological_complete_response \
  --run_glob "MRI_axisA_per_arm_ensemble"

run_step "06_MRI_MLP_common_neutral" \
  --model_root "${MODEL_ROOT}" \
  --external_ready "${MRI_EXTERNAL}" \
  --outdir "${OUTROOT}/MRI_MLP_common_neutral" \
  --target_col pathological_complete_response \
  --run_glob "MRI_axisA_common_neutral_mlp"

summary="${LOGDIR}/summary.txt"
{
  echo "Finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All external validation steps completed successfully."
  else
    echo "Failed steps:"
    printf '%s\n' "${FAILED[@]}"
  fi
} | tee "${summary}"

if [[ ${#FAILED[@]} -eq 0 ]]; then
  exit 0
fi
exit 1
