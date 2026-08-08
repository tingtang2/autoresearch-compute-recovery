#!/bin/bash
set -x  # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)"  # Make conda available to the shell
conda activate agent

# Determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# Check GPU availability in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
# Check GPU availability in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# Convert TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# Start a new file to store the full instructions, starting with general instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# Update instructions: replace /home/ paths to make paths relative
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt

# Append agent-specific instructions with environment variable substitution
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

# Append competition instructions
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# Create workspace directories (AIDE will create indexed subdirs like 0-exp/)
# Need to ensure they're writable since entrypoint excludes /home/agent from chmod
mkdir -p ${AGENT_DIR}/workspaces
mkdir -p ${AGENT_DIR}/logs
chmod -R a+rw ${AGENT_DIR}/workspaces ${AGENT_DIR}/logs 2>/dev/null || true

# Log debug consultant status
echo "============================================"
echo "AIDE ML Agent Starting"
echo "============================================"
echo "Bug Consultant: ${BUG_CONSULTANT:-not set}"
echo "Time Limit: ${TIME_LIMIT}"
echo "Step Limit: ${STEP_LIMIT:-not set}"
echo "Model: gpt-5-mini"
echo "============================================"

# Real-time sync function - copies AIDE outputs to mlebench directories every 60 seconds
sync_outputs() {
  while true; do
    sleep 60
    # Find AIDE log directory
    SYNC_LOG_DIR=$(find ${AGENT_DIR}/logs -maxdepth 1 -type d -name "*-exp" 2>/dev/null | sort -V | tail -1)
    if [ -n "$SYNC_LOG_DIR" ] && [ -d "$SYNC_LOG_DIR" ]; then
      # Sync solutions directory (submissions)
      if [ -d "$SYNC_LOG_DIR/solutions" ]; then
        mkdir -p "${LOGS_DIR}/solutions"
        cp -u "$SYNC_LOG_DIR/solutions"/*.csv "${LOGS_DIR}/solutions/" 2>/dev/null || true
        cp -u "$SYNC_LOG_DIR/solutions"/*.py "${LOGS_DIR}/solutions/" 2>/dev/null || true
      fi
      # Sync journal and other key files
      cp -u "$SYNC_LOG_DIR/journal.json" "${LOGS_DIR}/" 2>/dev/null || true
      cp -u "$SYNC_LOG_DIR/tree_plot.html" "${LOGS_DIR}/" 2>/dev/null || true
      cp -u "$SYNC_LOG_DIR/config.yaml" "${LOGS_DIR}/" 2>/dev/null || true
      # Sync bug consultant directory
      if [ -d "$SYNC_LOG_DIR/bug_consultant" ]; then
        mkdir -p "${LOGS_DIR}/bug_consultant"
        cp -u "$SYNC_LOG_DIR/bug_consultant"/* "${LOGS_DIR}/bug_consultant/" 2>/dev/null || true
      fi
      # Sync per_step_grading directory
      if [ -d "$SYNC_LOG_DIR/per_step_grading" ]; then
        mkdir -p "${LOGS_DIR}/per_step_grading"
        cp -u "$SYNC_LOG_DIR/per_step_grading"/* "${LOGS_DIR}/per_step_grading/" 2>/dev/null || true
      fi
      # Copy latest submission to submission dir for real-time grading
      LATEST_SUB=$(ls -t "$SYNC_LOG_DIR/solutions"/submission_node_*.csv 2>/dev/null | head -1)
      if [ -n "$LATEST_SUB" ] && [ -f "$LATEST_SUB" ]; then
        cp -u "$LATEST_SUB" "${SUBMISSION_DIR}/submission.csv" 2>/dev/null || true
      fi
      echo "[SYNC] Synced outputs at $(date '+%H:%M:%S')"
    fi
  done
}

# Start background sync process
sync_outputs &
SYNC_PID=$!
echo "Started background sync process (PID: $SYNC_PID)"

# Run AIDE with timeout, forwarding bash arguments for OmegaConf overrides
# AIDE creates directories like logs/0-exp/ and workspaces/0-exp/
timeout $TIME_LIMIT_SECS aide \
  data_dir="/home/data/" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="exp" \
  log_dir="${AGENT_DIR}/logs" \
  workspace_dir="${AGENT_DIR}/workspaces" \
  $@  # Forward the bash arguments to aide

EXIT_CODE=$?

# Stop background sync process
if [ -n "$SYNC_PID" ]; then
  kill $SYNC_PID 2>/dev/null || true
  echo "Stopped background sync process"
fi

if [ $EXIT_CODE -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

echo "============================================"
echo "Extracting AIDE outputs for mlebench..."
echo "============================================"

# Find the actual AIDE log directory (will be named like 0-exp, 1-exp, etc.)
AIDE_LOG_DIR=$(find ${AGENT_DIR}/logs -maxdepth 1 -type d -name "*-exp" | sort -V | tail -1)
AIDE_WORKSPACE_DIR=$(find ${AGENT_DIR}/workspaces -maxdepth 1 -type d -name "*-exp" | sort -V | tail -1)

echo "AIDE Log Dir: ${AIDE_LOG_DIR}"
echo "AIDE Workspace Dir: ${AIDE_WORKSPACE_DIR}"

# Copy logs to mlebench logs directory
if [ -d "${AIDE_LOG_DIR}" ]; then
  echo "Copying logs from ${AIDE_LOG_DIR} to ${LOGS_DIR}/"
  cp -r "${AIDE_LOG_DIR}"/* "${LOGS_DIR}/" 2>/dev/null || true

  # Copy best_solution.py to CODE_DIR
  if [ -f "${AIDE_LOG_DIR}/best_solution.py" ]; then
    echo "Copying best_solution.py to ${CODE_DIR}/"
    cp "${AIDE_LOG_DIR}/best_solution.py" "${CODE_DIR}/"
  fi
fi

# Extract the best submission for mlebench grading
# Priority order:
# 1. submission_post_search.csv (best selection strategy)
# 2. submission_raw.csv (best by validation metric)
# 3. Latest submission_node_*.csv from solutions/
# 4. submission.csv from workspace working directory

SUBMISSION_FOUND=false

# Try post_search submission first (uses configured selection strategy)
if [ -f "${AIDE_LOG_DIR}/submission_post_search.csv" ]; then
  echo "Using submission_post_search.csv"
  cp "${AIDE_LOG_DIR}/submission_post_search.csv" "${SUBMISSION_DIR}/submission.csv"
  SUBMISSION_FOUND=true
elif [ -f "${AIDE_LOG_DIR}/submission_raw.csv" ]; then
  echo "Using submission_raw.csv (best validation metric)"
  cp "${AIDE_LOG_DIR}/submission_raw.csv" "${SUBMISSION_DIR}/submission.csv"
  SUBMISSION_FOUND=true
fi

# Try solutions directory
if [ "$SUBMISSION_FOUND" = false ] && [ -d "${AIDE_LOG_DIR}/solutions" ]; then
  LATEST_SUBMISSION=$(ls -t "${AIDE_LOG_DIR}/solutions"/submission_node_*.csv 2>/dev/null | head -1)
  if [ -n "$LATEST_SUBMISSION" ] && [ -f "$LATEST_SUBMISSION" ]; then
    echo "Using latest node submission: $LATEST_SUBMISSION"
    cp "$LATEST_SUBMISSION" "${SUBMISSION_DIR}/submission.csv"
    SUBMISSION_FOUND=true
  fi
fi

# Try workspace working directory
if [ "$SUBMISSION_FOUND" = false ] && [ -f "${AIDE_WORKSPACE_DIR}/working/submission.csv" ]; then
  echo "Using workspace submission.csv"
  cp "${AIDE_WORKSPACE_DIR}/working/submission.csv" "${SUBMISSION_DIR}/submission.csv"
  SUBMISSION_FOUND=true
fi

# Last resort: find any submission CSV in AIDE directories
if [ "$SUBMISSION_FOUND" = false ]; then
  echo "WARNING: No standard submission found, searching for any CSV..."
  FOUND_CSV=$(find "${AGENT_DIR}/logs" "${AGENT_DIR}/workspaces" -name "submission*.csv" -type f 2>/dev/null | head -1)
  if [ -n "$FOUND_CSV" ] && [ -f "$FOUND_CSV" ]; then
    echo "Found submission: $FOUND_CSV"
    cp "$FOUND_CSV" "${SUBMISSION_DIR}/submission.csv"
    SUBMISSION_FOUND=true
  fi
fi

# Final check
if [ -f "${SUBMISSION_DIR}/submission.csv" ]; then
  echo "SUCCESS: Submission created at ${SUBMISSION_DIR}/submission.csv"
  wc -l "${SUBMISSION_DIR}/submission.csv"
else
  echo "ERROR: No submission.csv could be created!"
fi

echo "============================================"
echo "AIDE ML Agent Completed"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
