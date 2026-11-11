#!/usr/bin/env bash
# Train and evaluate one seed of one experiment on a group of GPUs.
#   bash scripts/run.sh configs/<model>/<method>.yaml <seed> <gpus> [port] [stage: all|train|evaluate]
# <gpus> is a comma-separated list, e.g. "0,1,2,3": training runs one process per device and evaluation shards its
# datasets over the same devices.  Completed stages are skipped, so re-running is safe.  Each judged bank gets a
# score cache (scores.npz + summary.json) with every judge label the paper reports.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P); cd "${ROOT}"
CFG=$1; SEED=$2; GPUS=$3; PORT=${4:-29500}; STAGE=${5:-all}
export CUDA_VISIBLE_DEVICES=${GPUS}
python -m training.run "${CFG}" --seed "${SEED}" --stage "${STAGE}" --port "${PORT}"
# A finished checkpoint keeps its weights only; evaluation never reads the optimizer state.
python -m tools.finalize_checkpoints --apply | tail -n 3
