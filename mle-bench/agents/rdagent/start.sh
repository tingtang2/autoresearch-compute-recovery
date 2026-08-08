#!/bin/bash
#
# RD-Agent start script — clean integration using the official `rdagent` CLI.
#
# 1. Activate conda env
# 2. Detect hardware
# 3. Set up data directory (writable copy)
# 4. Prepend MLE-bench instructions to description.md
# 5. Run `rdagent data_science` CLI
# 6. Find best submission.csv and copy to /home/submission/

set -euo pipefail
set -x

cd "${AGENT_DIR}"

# ── 1. Activate conda environment ──────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate agent

# ── 2. Detect hardware ─────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ' -)
else
  HARDWARE="CPU"
fi
export HARDWARE

python -c "import torch; print('PyTorch GPU:', torch.cuda.is_available())" || true

# ── 3. Compute time/step display ──────────────────────────────────────────
TIME_LIMIT_SECS=${TIME_LIMIT_SECS:-0}
STEP_LIMIT=${STEP_LIMIT:-50}

if [[ "${TIME_LIMIT_SECS}" =~ ^[0-9]+$ && "${TIME_LIMIT_SECS}" -gt 0 ]]; then
  TIME_LIMIT_HOURS=$(( (TIME_LIMIT_SECS + 3599) / 3600 ))
  DISPLAY_TIME="${TIME_LIMIT_SECS}s"
else
  TIME_LIMIT_HOURS=24
  DISPLAY_TIME="unbounded"
fi

echo "====================================="
echo "RD-Agent Starting (clean CLI mode)"
echo "Competition: ${COMPETITION_ID}"
echo "Time limit:  ${DISPLAY_TIME}"
echo "Step limit:  ${STEP_LIMIT}"
echo "Hardware:    ${HARDWARE}"
echo "====================================="

# ── 4. Set up writable data directory ─────────────────────────────────────
# RD-Agent expects data at ${DS_LOCAL_DATA_PATH}/${COMPETITION_ID}/
# /home/data is read-only, so copy to a writable location.
DATA_ROOT="${AGENT_DIR}/data"
COMP_DATA="${DATA_ROOT}/${COMPETITION_ID}"
mkdir -p "${COMP_DATA}"
cp -a /home/data/. "${COMP_DATA}/"

export DS_LOCAL_DATA_PATH="${DATA_ROOT}"

# KaggleScen.download_data() checks for zip_files/{competition} — if missing, it
# tries Docker-based download. Data is already mounted by MLE-bench, so skip it.
mkdir -p "${DATA_ROOT}/zip_files/${COMPETITION_ID}"

# ── 5. Prepend MLE-bench instructions to description.md ──────────────────
DESCRIPTION_FILE="${COMP_DATA}/description.md"
INSTRUCTIONS_FILE="/home/instructions.txt"

if [[ -f "${INSTRUCTIONS_FILE}" && -f "${DESCRIPTION_FILE}" ]]; then
  ORIGINAL_DESC=$(cat "${DESCRIPTION_FILE}")
  {
    cat "${INSTRUCTIONS_FILE}"
    echo ""
    echo "ADDITIONAL NOTES"
    echo "------"
    echo ""
    echo "- **Compute**: You have access to ${HARDWARE} with the appropriate drivers installed."
    echo "- **Total Runtime**: You have a maximum of ${TIME_LIMIT_HOURS} hours to attempt this task."
    echo "- **Total Steps**: You have a maximum of ${STEP_LIMIT} steps to submit your solution."
    echo "- **Use as much of the available time as possible to refine and optimize your submission.**"
    echo ""
    echo "COMPETITION INSTRUCTIONS"
    echo "------"
    echo ""
    echo "${ORIGINAL_DESC}"
  } > "${DESCRIPTION_FILE}"
  echo "Prepended MLE-bench instructions to description.md"
fi

# ── 6. Run rdagent CLI ────────────────────────────────────────────────────
# The rdagent data_science command uses env vars for configuration (DS_ prefix).
# Key env vars are set via config.yaml env_vars and exported by MLE-bench.
RDAGENT_EXIT=0
if [[ "${TIME_LIMIT_SECS}" =~ ^[0-9]+$ && "${TIME_LIMIT_SECS}" -gt 0 ]]; then
  timeout "${TIME_LIMIT_SECS}" rdagent data_science \
    --competition "${COMPETITION_ID}" \
    --step-n "${STEP_LIMIT}" \
    --timeout "${TIME_LIMIT_SECS}" || RDAGENT_EXIT=$?
else
  rdagent data_science \
    --competition "${COMPETITION_ID}" \
    --step-n "${STEP_LIMIT}" || RDAGENT_EXIT=$?
fi

if [[ "${RDAGENT_EXIT}" -eq 124 ]]; then
  echo "RD-Agent timed out after ${TIME_LIMIT_SECS}s (expected)"
elif [[ "${RDAGENT_EXIT}" -ne 0 ]]; then
  echo "RD-Agent exited with status ${RDAGENT_EXIT}"
fi

# ── 7. Find and copy best submission ──────────────────────────────────────
SUBMISSION_PATH="${SUBMISSION_DIR}/submission.csv"

if [[ -f "${SUBMISSION_PATH}" ]]; then
  echo "Submission already at ${SUBMISSION_PATH}"
else
  echo "Searching for submission.csv in RD-Agent workspaces..."

  # Search RD-Agent log/workspace directories for the most recent submission
  BEST_SUBMISSION=""
  BEST_MTIME=0

  while IFS= read -r -d '' candidate; do
    MTIME=$(stat -c '%Y' "${candidate}" 2>/dev/null || echo 0)
    if [[ "${MTIME}" -gt "${BEST_MTIME}" ]]; then
      BEST_MTIME="${MTIME}"
      BEST_SUBMISSION="${candidate}"
    fi
  done < <(find /home -name "submission.csv" -type f -print0 2>/dev/null)

  if [[ -n "${BEST_SUBMISSION}" ]]; then
    echo "Found submission at ${BEST_SUBMISSION}, copying to ${SUBMISSION_PATH}"
    mkdir -p "${SUBMISSION_DIR}"
    cp "${BEST_SUBMISSION}" "${SUBMISSION_PATH}"
  else
    # Last resort: copy sample_submission.csv if available
    if [[ -f "${COMP_DATA}/sample_submission.csv" ]]; then
      echo "WARNING: No submission produced. Copying sample_submission.csv as fallback."
      mkdir -p "${SUBMISSION_DIR}"
      cp "${COMP_DATA}/sample_submission.csv" "${SUBMISSION_PATH}"
    else
      echo "ERROR: No submission.csv found anywhere."
      find /home -name "*.csv" -type f 2>/dev/null | head -20 || true
      exit 1
    fi
  fi
fi

# ── 8. Copy trace logs ───────────────────────────────────────────────
echo "Copying RD-Agent trace logs to /home/logs..."
cp -a /home/agent/log/. /home/logs/rdagent/ 2>/dev/null || true

echo "RD-Agent completed. Submission at ${SUBMISSION_PATH}"
