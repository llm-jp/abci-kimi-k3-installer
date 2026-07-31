#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: bash $0 REPLICA_COUNT" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "$SCRIPT_DIR/config.env"

readonly REPLICA_COUNT="$1"
k3_require_positive_integer REPLICA_COUNT "$REPLICA_COUNT"
readonly NODE_COUNT=$((REPLICA_COUNT * 2))

for required_file in \
    "$SCRIPT_DIR/server.pbs" \
    "$SCRIPT_DIR/server-node.sh" \
    "$SCRIPT_DIR/runtime.py"
do
    if [[ ! -f "$required_file" ]]; then
        k3_die "required server file not found: $required_file"
    fi
done

cd "$SCRIPT_DIR"
job_id=$(
    qsub \
        -P "$ABCI_PROJECT" \
        -q rt_HF \
        -l "select=${NODE_COUNT}:mpiprocs=1" \
        -l "walltime=02:00:00" \
        -v "USE_SSH=1,REPLICA_COUNT=$REPLICA_COUNT,K3_CONFIG=$K3_CONFIG" \
        "$SCRIPT_DIR/server.pbs"
)

echo "job_id=$job_id"
echo "replica_count=$REPLICA_COUNT"
echo "node_count=$NODE_COUNT"
echo "gpu_count=$((NODE_COUNT * 8))"
