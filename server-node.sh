#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != "discover" && "$1" != "launch" ]]; then
    echo "Usage: bash $0 discover|launch" >&2
    exit 2
fi

readonly MODE="$1"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "${K3_CONFIG:?K3_CONFIG is not set}"

readonly HOSTNAME_SHORT="$(hostname -s)"
readonly GLOBAL_RANK="${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is not set}"
readonly WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?OMPI_COMM_WORLD_SIZE is not set}"

LOCAL_IP=$(
    ip -o -4 addr show dev ibn1 |
        awk '{sub(/\/.*/, "", $4); print $4; exit}'
)
if [[ -z "$LOCAL_IP" ]]; then
    k3_die "could not determine IPv4 address of ibn1 on $HOSTNAME_SHORT"
fi

if [[ "$MODE" == "discover" ]]; then
    readonly INFO_DIR="${K3_NODE_INFO_DIR:?K3_NODE_INFO_DIR is not set}"
    readonly EXPECTED_RANKS="${EXPECTED_DISCOVERY_RANKS:?EXPECTED_DISCOVERY_RANKS is not set}"
    if (( WORLD_SIZE != EXPECTED_RANKS ||
          GLOBAL_RANK < 0 ||
          GLOBAL_RANK >= WORLD_SIZE )); then
        k3_die "invalid discovery rank/world: $GLOBAL_RANK/$WORLD_SIZE"
    fi
    readonly INFO_FILE="$INFO_DIR/rank-$GLOBAL_RANK.tsv"
    readonly TEMP_FILE="$INFO_FILE.tmp.$HOSTNAME_SHORT.$$"
    printf '%s\t%s\n' "$HOSTNAME_SHORT" "$LOCAL_IP" >"$TEMP_FILE"
    mv "$TEMP_FILE" "$INFO_FILE"
    echo "node_info rank=$GLOBAL_RANK host=$HOSTNAME_SHORT ip=$LOCAL_IP"
    exit 0
fi

if [[ -z ${PBS_JOBID:-} ]]; then
    k3_die "launch mode requires a PBS job"
fi

readonly REPLICA_INDEX="${REPLICA_INDEX:?REPLICA_INDEX is not set}"
readonly TOTAL_REPLICAS="${REPLICA_COUNT:?REPLICA_COUNT is not set}"
readonly NODE_RANK="$GLOBAL_RANK"
readonly LOG_DIR="${SGLANG_LOG_DIR:?SGLANG_LOG_DIR is not set}"
readonly DIST_ADDR="${DIST_INIT_ADDR:?DIST_INIT_ADDR is not set}"
readonly JOB_NUMBER="${PBS_JOBID%%.*}"
readonly NODE_LOG="$LOG_DIR/node-rank-$NODE_RANK-$HOSTNAME_SHORT.log"
readonly PYTHON="$RUNTIME_ROOT/venv/bin/python"
readonly RUNTIME="$SCRIPT_DIR/runtime.py"

exec >>"$NODE_LOG" 2>&1

if (( WORLD_SIZE != 2 ||
      NODE_RANK < 0 ||
      NODE_RANK >= 2 )); then
    k3_die "invalid replica launcher rank/world: $NODE_RANK/$WORLD_SIZE"
fi
if (( TOTAL_REPLICAS < 1 ||
      REPLICA_INDEX < 0 ||
      REPLICA_INDEX >= TOTAL_REPLICAS )); then
    k3_die "invalid replica index/count: $REPLICA_INDEX/$TOTAL_REPLICAS"
fi

for required_path in \
    "$PYTHON" \
    "$RUNTIME" \
    "$MODEL_DIR/config.json" \
    "$MODEL_DIR/model.safetensors.index.json"
do
    [[ -e "$required_path" ]] || k3_die "required path not found: $required_path"
done

NCCL_LINKER_LIBRARY=$(c++ -print-file-name=libnccl.so)
readonly NCCL_LINKER_LIBRARY
if [[ "$NCCL_LINKER_LIBRARY" == "libnccl.so" ||
      ! -f "$NCCL_LINKER_LIBRARY" ]]; then
    k3_die "NCCL module did not provide libnccl.so"
fi

readonly CACHE_DIR="$RUNTIME_ROOT/cache/k3-h200-tp16/$HOSTNAME_SHORT"
readonly HF_CACHE_DIR="$RUNTIME_ROOT/cache/transformers/$JOB_NUMBER/$HOSTNAME_SHORT"
mkdir -p \
    "$CACHE_DIR/cuda" \
    "$CACHE_DIR/triton" \
    "$CACHE_DIR/torch-extensions" \
    "$HF_CACHE_DIR/modules" \
    "$HF_CACHE_DIR/xdg"

export PATH="$RUNTIME_ROOT/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export CUDA_CACHE_PATH="$CACHE_DIR/cuda"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch-extensions"
export HF_HOME="$HF_CACHE_DIR/huggingface"
export HF_MODULES_CACHE="$HF_CACHE_DIR/modules"
export XDG_CACHE_HOME="$HF_CACHE_DIR/xdg"

export SGLANG_HOST_IP="$LOCAL_IP"
export NCCL_SOCKET_IFNAME=ibn1
export GLOO_SOCKET_IFNAME=ibn1
export NCCL_IB_DISABLE=0
export NCCL_NET=IB

SCHEDULER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
unset CUDA_VISIBLE_DEVICES

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
if (( ${#GPU_NAMES[@]} != 8 )); then
    k3_die "expected 8 GPUs on $HOSTNAME_SHORT; found ${#GPU_NAMES[@]}"
fi
for gpu_name in "${GPU_NAMES[@]}"; do
    if [[ "$gpu_name" != NVIDIA\ H200* ]]; then
        k3_die "expected NVIDIA H200 on $HOSTNAME_SHORT; found $gpu_name"
    fi
done

echo "replica_index=$REPLICA_INDEX"
echo "replica_count=$TOTAL_REPLICAS"
echo "node_rank=$NODE_RANK"
echo "host=$HOSTNAME_SHORT"
echo "local_ip=$LOCAL_IP"
echo "dist_init_addr=$DIST_ADDR"
echo "server_port=30000"
echo "scheduler_cuda_visible_devices=$SCHEDULER_CUDA_VISIBLE_DEVICES"
echo "gpu_count=${#GPU_NAMES[@]}"
echo "gpu_name=${GPU_NAMES[0]}"
echo "nccl_linker_library=$NCCL_LINKER_LIBRARY"
echo "cache_dir=$CACHE_DIR"
echo "hf_cache_dir=$HF_CACHE_DIR"

"$PYTHON" "$RUNTIME" prewarm --model-dir "$MODEL_DIR"

exec "$PYTHON" -m sglang.launch_server \
    --trust-remote-code \
    --model-path "$MODEL_DIR" \
    --tp-size 16 \
    --ep-size 16 \
    --moe-runner-backend marlin \
    --decode-attention-backend flashmla \
    --enable-symm-mem \
    --mem-fraction-static 0.85 \
    --mamba-full-memory-ratio 0.45 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --nnodes 2 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_ADDR" \
    --host 0.0.0.0 \
    --port 30000
