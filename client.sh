#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage:" >&2
    echo "  bash $0 smoke ROUTER_HOST" >&2
    echo "  bash $0 capacity ROUTER_HOST REPLICA_COUNT" >&2
    echo "  bash $0 benchmark ROUTER_HOST REPLICA_COUNT CONCURRENCY_PER_SERVER WAVES INPUT_TOKENS OUTPUT_TOKENS SEED" >&2
}

if (( $# < 2 )); then
    usage
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "$SCRIPT_DIR/config.env"

readonly COMMAND="$1"
readonly ROUTER_HOST="$2"
readonly ROUTER_URL="http://127.0.0.1:31000"
readonly PYTHON="$RUNTIME_ROOT/venv/bin/python"

if [[ ! "$ROUTER_HOST" =~ ^[A-Za-z0-9._-]+$ ||
      "$ROUTER_HOST" == -* ]]; then
    k3_die "invalid Router host: $ROUTER_HOST"
fi
for required_path in "$PYTHON" "$SCRIPT_DIR/runtime.py"; do
    [[ -e "$required_path" ]] || k3_die "required path not found: $required_path"
done

declare -a REMOTE_ARGS
case "$COMMAND" in
    smoke)
        (( $# == 2 )) || k3_die "smoke does not take additional arguments"
        REMOTE_ARGS=(
            "$PYTHON"
            "$SCRIPT_DIR/runtime.py"
            smoke
            --base-url "$ROUTER_URL/v1"
        )
        ;;
    capacity)
        (( $# == 3 )) || k3_die "capacity requires REPLICA_COUNT"
        k3_require_positive_integer REPLICA_COUNT "$3"
        REMOTE_ARGS=(
            "$PYTHON"
            "$SCRIPT_DIR/runtime.py"
            capacity
            --router-url "$ROUTER_URL"
            --model-dir "$MODEL_DIR"
            --expected-workers "$3"
        )
        ;;
    benchmark)
        (( $# == 8 )) || {
            usage
            k3_die "benchmark requires all workload arguments"
        }
        for argument in "${@:3:5}"; do
            k3_require_positive_integer "benchmark argument" "$argument"
        done
        [[ "$8" =~ ^[0-9]+$ ]] || k3_die "SEED must be a non-negative integer: $8"
        mkdir -p "$SCRIPT_DIR/results"
        RESULT_JSON="$SCRIPT_DIR/results/k3-server-scaling-$(date '+%Y%m%d-%H%M%S').json"
        REMOTE_ARGS=(
            "$PYTHON"
            "$SCRIPT_DIR/client.py"
            --router-url "$ROUTER_URL"
            --model-dir "$MODEL_DIR"
            --expected-workers "$3"
            --concurrency-per-server "$4"
            --waves "$5"
            --input-tokens "$6"
            --output-tokens "$7"
            --seed "$8"
            --output-json "$RESULT_JSON"
        )
        echo "result_json=$RESULT_JSON"
        ;;
    *)
        k3_die "unknown client command: $COMMAND"
        ;;
esac

printf -v REMOTE_EXEC '%q ' "${REMOTE_ARGS[@]}"
REMOTE_COMMAND=". /etc/profile.d/modules.sh && module purge && module load python/3.12/3.12.9 && exec $REMOTE_EXEC"

echo "command=$COMMAND"
echo "router_host=$ROUTER_HOST"
exec ssh \
    -T \
    -o BatchMode=yes \
    -o ConnectTimeout=15 \
    "$ROUTER_HOST" \
    "$REMOTE_COMMAND"
