#!/usr/bin/env bash
#
# Build the C++ benchmark and profile it with perf.
#
# Usage:
#   ./scripts/profile.sh [workload] [mode] [repeat]
#
#   workload:  mixed | exact | approx | nofb | big | gen | all   (default: mixed)
#   mode:      stat | record | both                              (default: both)
#
# Examples:
#   ./scripts/profile.sh                 # build, perf stat + perf record on 'mixed'
#   ./scripts/profile.sh exact stat      # counters for the all-exact workload
#   ./scripts/profile.sh mixed record    # call graphs only
#
# perf stat counters cover ONLY the measured sampling loop (dataset
# generation, model construction and warmup are excluded) via perf's
# control FIFO. Falls back to a plain run if perf is unavailable.

set -euo pipefail

WORKLOAD="${1:-mixed}"
MODE="${2:-both}"
REPEAT="${3:-50}"   # measured-loop multiplier: keeps counters stable (~seconds)

cd "$(dirname "$0")/.."

# ---- build ----------------------------------------------------------------
cmake -B build \
      -DBUILD_BENCHMARK=ON \
      -DBUILD_PYTHON_MODULE=OFF \
      -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j > /dev/null

BENCH=./build/ft_bench
echo "Built ${BENCH}"

# ---- no perf? just run ------------------------------------------------------
if ! command -v perf > /dev/null 2>&1; then
    echo "perf not found -- running benchmark without profiling:"
    exec "${BENCH}" "${WORKLOAD}" "${REPEAT}"
fi

# ---- perf stat (counters over the measured region only) --------------------
if [[ "${MODE}" == "stat" || "${MODE}" == "both" ]]; then
    echo
    echo "=== perf stat (${WORKLOAD}, measured region only) ==="

    if perf stat --help 2>&1 | grep -q -- '--control'; then
        FIFO_DIR="$(mktemp -d)"
        trap 'rm -rf "${FIFO_DIR}"' EXIT
        mkfifo "${FIFO_DIR}/ctl" "${FIFO_DIR}/ack"

        PERF_CTL_FIFO="${FIFO_DIR}/ctl" \
        PERF_ACK_FIFO="${FIFO_DIR}/ack" \
        perf stat -d --delay=-1 \
            --control "fifo:${FIFO_DIR}/ctl,${FIFO_DIR}/ack" \
            -- "${BENCH}" "${WORKLOAD}" "${REPEAT}"
    else
        echo "(old perf without --control; falling back to -D 500 delay)"
        perf stat -d -D 500 -- "${BENCH}" "${WORKLOAD}" "${REPEAT}"
    fi
fi

# ---- perf record (call graphs) ----------------------------------------------
if [[ "${MODE}" == "record" || "${MODE}" == "both" ]]; then
    echo
    echo "=== perf record (${WORKLOAD}) ==="
    perf record -g -o build/perf.data -- "${BENCH}" "${WORKLOAD}" "${REPEAT}"
    echo
    echo "Top functions:"
    perf report -i build/perf.data --stdio --percent-limit 2 | head -30
    echo
    echo "Interactive view:  perf report -i build/perf.data"
fi