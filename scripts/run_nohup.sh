#!/bin/bash
# Start training in background with nohup (survives SSH disconnect).
#
# Usage:
#   bash scripts/run_nohup.sh scripts/train_partnet_unified.py --config configs/partnet_pt_v2_unified.yaml
#   bash scripts/run_nohup.sh scripts/train_partnet_unified.py --config configs/partnet_pt_v3_unified.yaml
#   bash scripts/run_nohup.sh scripts/train_partnet_unified.py --config ... --resume path/to/best.pth
#
# Logs: experiments/<exp_name>/nohup.log
# Status: tail -f experiments/<exp_name>/nohup.log

SCRIPT=$1
shift  # remaining args go to the Python script

# Extract experiment name from args
EXP_NAME=""
for arg in "$@"; do
    if [[ "$arg" == *"unified"* ]]; then
        EXP_NAME=$(echo "$arg" | grep -oP 'partnet_pt_[a-z0-9_]+(?=\.yaml)')
    fi
done
EXP_NAME=${EXP_NAME:-training}

EXP_DIR="experiments/exp_pt_${EXP_NAME##*_}_unified"  # best-effort guess
LOG_FILE="${EXP_DIR}/nohup.log"

echo "Starting: python $SCRIPT $@"
echo "Log file: $LOG_FILE"
echo "Run 'tail -f $LOG_FILE' to monitor"
echo ""

mkdir -p "$EXP_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python "$SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID: $PID"

# Save PID for later cleanup
echo "$PID" > "${EXP_DIR}/train.pid"
echo "Started. Wait a few seconds then check:"
echo "  tail -5 $LOG_FILE"
