#!/usr/bin/env bash
# Unified entrypoint for the SVDD data preprocessing pipeline.
#
# Usage:
#   ./scripts/preprocessing/run.sh sonics features
#   ./scripts/preprocessing/run.sh sonics pool
#   ./scripts/preprocessing/run.sh sonics merge
#   ./scripts/preprocessing/run.sh mom features
#   ./scripts/preprocessing/run.sh m6 features
#
# Full documentation: docs/preprocessing_pipeline.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATHS_FILE="${PATHS_FILE:-${SCRIPT_DIR}/paths.env}"
if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
else
  echo "Warning: ${PATHS_FILE} not found. Copy paths.env.example → paths.env"
  SCRATCH="${SCRATCH:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data}"
  SONICS_FEATURES_CLEAN="${SCRATCH}/Sonics/preprocessed_16k_mono"
  SONICS_POOLED_CLEAN="${SCRATCH}/Sonics/preprocessed_pooled_fp16"
  SONICS_FEATURES_AUG="${SCRATCH}/Sonics/preprocessed_aug_v0"
  SONICS_POOLED_AUG="${SCRATCH}/Sonics/preprocessed_pooled_aug_v0"
  SONICS_POOLED_MERGED="${SCRATCH}/Sonics/preprocessed_pooled_merged"
  MOM_FEATURES="${SCRATCH}/MoM/mom_preprocessed"
  MOM_POOLED="${SCRATCH}/MoM/preprocessed_pooled_fp16"
  M6_FEATURES="${SCRATCH}/M6/m6_preprocessed"
  M6_POOLED="${SCRATCH}/M6/preprocessed_pooled_fp16"
fi

VENV="${VENV:-${PROJECT_ROOT}/.venv/bin/activate}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <dataset> <stage> [options]

Datasets:  sonics | mom | m6

Stages:
  audio          Stage 0 — resample/codec to 16k mono (Sonics only, CPU)
  features       Stage 1 — W2V + MERT extraction (GPU, preprocess.py)
  features-aug   Stage 1b — augmented train features (GPU, preprocess.py + sonics_aug)
  pool           Stage 2 — concat + AvgPool → fp16 [1500,2048] (CPU)
  pool-aug       Stage 2b — pool augmented features (CPU)
  merge          Stage 3 — merge clean + aug index.json for training
  aug-all        Stage 1b+2b for variants 0,1,2 then merge (Sonics train aug)

Examples:
  $(basename "$0") sonics features
  $(basename "$0") sonics pool
  $(basename "$0") sonics merge --aug ${SONICS_POOLED_AUG}/index.json
  $(basename "$0") mom pool

Docs: docs/preprocessing_pipeline.md
EOF
}

activate_venv() {
  if [[ -f "${VENV}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}"
  else
    echo "Warning: venv not found at ${VENV}"
  fi
}

run_features() {
  local dataset="$1"
  local extra_args=("${@:2}")
  activate_venv
  cd "${PROJECT_ROOT}/src"
  case "${dataset}" in
    sonics)
      python preprocess.py paths.output_dir="${SONICS_FEATURES_CLEAN}" "${extra_args[@]}"
      ;;
    mom)
      python preprocess.py \
        preprocessing=mom \
        paths.output_dir="${MOM_FEATURES}" \
        "${extra_args[@]}"
      ;;
    m6)
      python preprocess.py \
        preprocessing=m6 \
        paths.output_dir="${M6_FEATURES}" \
        "${extra_args[@]}"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

run_features_aug() {
  local variant="${1:-0}"
  activate_venv
  cd "${PROJECT_ROOT}/src"
  local out_dir="${SONICS_FEATURES_AUG/_v0/_v${variant}}"
  echo "Stage 1b (aug): variant=${variant} → ${out_dir}"
  python preprocess.py \
    preprocessing=sonics_aug \
    paths.output_dir="${out_dir}" \
    preprocessing.aug_variant="${variant}"
}

run_aug_variant() {
  local variant="$1"
  run_features_aug "${variant}"
  local feat_dir="${SONICS_FEATURES_AUG/_v0/_v${variant}}"
  local pool_dir="${SONICS_POOLED_AUG/_v0/_v${variant}}"
  run_pool "${feat_dir}/index.json" "${pool_dir}"
}

run_aug_all() {
  local variants=(0 1 2)
  local aug_indexes=()
  for v in "${variants[@]}"; do
    run_aug_variant "${v}"
    aug_indexes+=("${SONICS_POOLED_AUG/_v0/_v${v}}/index.json")
  done
  run_merge --aug "${aug_indexes[@]}"
}

run_pool() {
  local index="$1"
  local output="$2"
  activate_venv
  cd "${PROJECT_ROOT}"
  python scripts/processing/precompute_pooled.py \
    --index "${index}" \
    --output_dir "${output}" \
    --workers "${WORKERS:-16}"
}

run_merge() {
  local clean_index="${SONICS_POOLED_CLEAN}/index.json"
  local out_index="${SONICS_POOLED_MERGED}/index.json"
  local aug_indexes=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --clean) clean_index="$2"; shift 2 ;;
      --out)   out_index="$2"; shift 2 ;;
      --aug)   aug_indexes+=("$2"); shift 2 ;;
      *) echo "Unknown merge arg: $1" >&2; exit 1 ;;
    esac
  done

  if [[ ${#aug_indexes[@]} -eq 0 ]]; then
    aug_indexes=("${SONICS_POOLED_AUG}/index.json")
  fi

  activate_venv
  cd "${PROJECT_ROOT}"
  python scripts/processing/merge_pooled_index.py \
    --clean "${clean_index}" \
    --aug "${aug_indexes[@]}" \
    --out "${out_index}"
  echo "Merged index: ${out_index}"
  echo "Training: set data.data_dir to ${SONICS_POOLED_MERGED} (directory containing merged index)"
}

# --- main ---

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

DATASET="$1"
STAGE="$2"
shift 2

case "${STAGE}" in
  audio)
    [[ "${DATASET}" == "sonics" ]] || { echo "audio stage only for sonics"; exit 1; }
    bash "${PROJECT_ROOT}/scripts/bash/sonics_mp3_codec.sh"
    ;;
  features)
    run_features "${DATASET}" "$@"
    ;;
  features-aug)
    [[ "${DATASET}" == "sonics" ]] || { echo "features-aug only for sonics"; exit 1; }
    variant="${1:-0}"
    run_features_aug "${variant}"
    ;;
  pool)
    case "${DATASET}" in
      sonics) run_pool "${SONICS_FEATURES_CLEAN}/index.json" "${SONICS_POOLED_CLEAN}" ;;
      mom)    run_pool "${MOM_FEATURES}/index.json" "${MOM_POOLED}" ;;
      m6)     run_pool "${M6_FEATURES}/index.json" "${M6_POOLED}" ;;
      *) echo "Unknown dataset"; exit 1 ;;
    esac
    ;;
  pool-aug)
    [[ "${DATASET}" == "sonics" ]] || { echo "pool-aug only for sonics"; exit 1; }
    variant="${1:-0}"
    feat_dir="${SONICS_FEATURES_AUG/_v0/_v${variant}}"
    pool_dir="${SONICS_POOLED_AUG/_v0/_v${variant}}"
    run_pool "${feat_dir}/index.json" "${pool_dir}"
    ;;
  aug-all)
    [[ "${DATASET}" == "sonics" ]] || { echo "aug-all only for sonics"; exit 1; }
    run_aug_all
    ;;
  merge)
    [[ "${DATASET}" == "sonics" ]] || { echo "merge only for sonics"; exit 1; }
    run_merge "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
