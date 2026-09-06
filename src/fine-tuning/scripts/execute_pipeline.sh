#!/bin/bash
# ==============================================================================
# Entrypoint: execute_pipeline.sh
# Purpose: Master CLI interface to dispatch the modular stages.
# Usage: ./execute_pipeline.sh --stage 1
# ==============================================================================

# Ensure environment variables are loaded
source "$(dirname "$0")/core/environment.sh"
validate_environment

source "$(dirname "$0")/modules/task_adapter.sh"
source "$(dirname "$0")/modules/monolithic.sh"
source "$(dirname "$0")/modules/language_adapter.sh"

STAGE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$STAGE" ]]; then
    echo "Usage: $0 --stage <1|2|3|4|all>"
    exit 1
fi

echo "======================================================================"
echo "  [VoQA Pipeline Execution Engine] "
echo "======================================================================"

case $STAGE in
    1) run_stage1 ;;
    2) run_stage2 ;;
    3) run_stage3 ;;
    4) run_stage4 ;;
    all)
        run_stage1
        run_stage2
        run_stage3
        run_stage4
        ;;
    *)
        echo "Invalid stage specified. Use 1, 2, 3, 4, or all."
        exit 1
        ;;
esac

echo "[*] Exiting Master Orchestrator."
