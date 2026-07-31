#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: bash $0" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "$SCRIPT_DIR/config.env"

[[ ! -e "$RUNTIME_ROOT" ]] \
    || k3_die "RUNTIME_ROOT already exists: $RUNTIME_ROOT"

for required_file in \
    "$SCRIPT_DIR/install.pbs" \
    "$SCRIPT_DIR/install-worker.sh" \
    "$SCRIPT_DIR/runtime.py"
do
    if [[ ! -f "$required_file" ]]; then
        k3_die "required installer file not found: $required_file"
    fi
done

cd "$SCRIPT_DIR"
job_id=$(
    qsub \
        -P "$ABCI_PROJECT" \
        -q rt_HC \
        -l "select=1:mpiprocs=1" \
        -l "walltime=06:00:00" \
        -v "K3_CONFIG=$K3_CONFIG" \
        "$SCRIPT_DIR/install.pbs"
)

echo "job_id=$job_id"
