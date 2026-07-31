#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: bash $0 ROUTER_HOST REQUEST_SCRIPT" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "$SCRIPT_DIR/config.env"

readonly ROUTER_HOST="$1"
if [[ ! "$ROUTER_HOST" =~ ^[A-Za-z0-9._-]+$ ||
      "$ROUTER_HOST" == -* ]]; then
    k3_die "invalid Router host: $ROUTER_HOST"
fi

[[ -f "$2" ]] || k3_die "request script not found: $2"
readonly REQUEST_SCRIPT="$(realpath "$2")"
[[ "$REQUEST_SCRIPT" != *","* ]] \
    || k3_die "request script path must not contain a comma: $REQUEST_SCRIPT"

[[ -f "$SCRIPT_DIR/request.pbs" ]] \
    || k3_die "required request file not found: $SCRIPT_DIR/request.pbs"

cd "$SCRIPT_DIR"
job_id=$(
    qsub \
        -P "$ABCI_PROJECT" \
        -q rt_HC \
        -l "select=1:mpiprocs=1" \
        -l "walltime=00:10:00" \
        -v "USE_SSH=1,ROUTER_HOST=$ROUTER_HOST,K3_CONFIG=$K3_CONFIG,REQUEST_SCRIPT=$REQUEST_SCRIPT" \
        "$SCRIPT_DIR/request.pbs"
)

echo "job_id=$job_id"
echo "router_host=$ROUTER_HOST"
echo "request_script=$REQUEST_SCRIPT"
