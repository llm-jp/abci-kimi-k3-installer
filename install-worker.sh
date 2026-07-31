#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
k3_load_config "${K3_CONFIG:?K3_CONFIG is not set}"

readonly SGLANG_COMMIT="9cb03516b2baa9b42a418de98deea491a9ab8eb9"
readonly SGLANG_REPOSITORY="https://github.com/sgl-project/sglang.git"
readonly RUST_VERSION="1.90.0"
readonly ROUTER_VERSION="0.3.2"
readonly TORCH_VERSION="2.11.0+cu130"
readonly TORCHVISION_VERSION="0.26.0+cu130"
readonly TORCHAUDIO_VERSION="2.11.0+cu130"
readonly PYPI_INDEX_URL="https://pypi.org/simple"
readonly PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"

readonly SOURCE_DIR="$RUNTIME_ROOT/src/sglang"
readonly VENV_DIR="$RUNTIME_ROOT/venv"
readonly PYTHON="$VENV_DIR/bin/python"
readonly INSTALL_MARKER="$RUNTIME_ROOT/.k3-install-${SGLANG_COMMIT:0:8}.complete"

[[ -n ${PBS_JOBID:-} ]] || k3_die "install-worker.sh must run inside a PBS job"
[[ ! -e "$RUNTIME_ROOT" ]] \
    || k3_die "RUNTIME_ROOT already exists: $RUNTIME_ROOT"
for required_path in \
    "$MODEL_DIR/config.json" \
    "$MODEL_DIR/model.safetensors.index.json" \
    "$SCRIPT_DIR/runtime.py"
do
    [[ -e "$required_path" ]] || k3_die "required path not found: $required_path"
done
for command_name in curl git python3; do
    command -v "$command_name" >/dev/null 2>&1 \
        || k3_die "required command not found: $command_name"
done
[[ "$(python3 --version 2>&1)" == "Python 3.12."* ]] \
    || k3_die "Python 3.12 module is required: $(python3 --version 2>&1)"

python3 "$SCRIPT_DIR/runtime.py" verify-model --model-dir "$MODEL_DIR"

mkdir -p "$RUNTIME_ROOT/src"

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/k3-install.XXXXXXXX")
cleanup() {
    [[ ! -d ${WORK_DIR:-} ]] || rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

echo "=== Install Rust $RUST_VERSION ==="
curl \
    --proto '=https' \
    --tlsv1.2 \
    --fail \
    --show-error \
    --location \
    --output "$WORK_DIR/rustup-init.sh" \
    https://sh.rustup.rs
RUSTUP_HOME="$RUNTIME_ROOT/rustup" \
CARGO_HOME="$RUNTIME_ROOT/cargo" \
    sh "$WORK_DIR/rustup-init.sh" \
        -y \
        --no-modify-path \
        --profile minimal \
        --default-toolchain "$RUST_VERSION"
export RUSTUP_HOME="$RUNTIME_ROOT/rustup"
export CARGO_HOME="$RUNTIME_ROOT/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
[[ "$(rustc --version)" == "rustc $RUST_VERSION "* ]] \
    || k3_die "unexpected Rust toolchain: $(rustc --version)"

echo "=== Fetch SGLang source ==="
mkdir "$SOURCE_DIR"
git -C "$SOURCE_DIR" init
git -C "$SOURCE_DIR" remote add origin "$SGLANG_REPOSITORY"
git -C "$SOURCE_DIR" fetch --depth 1 origin "$SGLANG_COMMIT"
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
ACTUAL_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
[[ "$ACTUAL_COMMIT" == "$SGLANG_COMMIT" ]] \
    || k3_die "source commit mismatch: $ACTUAL_COMMIT"

echo "=== Create Python environment ==="
python3 -m venv "$VENV_DIR"
export PATH="$VENV_DIR/bin:$CARGO_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$PYTHON" -m pip install \
    --index-url "$PYPI_INDEX_URL" \
    --upgrade \
    pip

echo "=== Install PyTorch CUDA 13.0 ==="
"$PYTHON" -m pip install \
    --index-url "$PYTORCH_INDEX_URL" \
    --extra-index-url "$PYPI_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

echo "=== Install SGLang ==="
"$PYTHON" -m pip install \
    --index-url "$PYPI_INDEX_URL" \
    --extra-index-url "$PYTORCH_INDEX_URL" \
    --editable "$SOURCE_DIR/python"

echo "=== Install SGLang Router ==="
"$PYTHON" -m pip install \
    --index-url "$PYPI_INDEX_URL" \
    "sglang-router==$ROUTER_VERSION"

echo "=== Verify installation ==="
"$PYTHON" -m pip check
"$PYTHON" "$SCRIPT_DIR/runtime.py" \
    verify-install \
    --source-dir "$SOURCE_DIR" \
    --expected-commit "$SGLANG_COMMIT" \
    --router-version "$ROUTER_VERSION"
"$PYTHON" -m pip freeze >"$RUNTIME_ROOT/environment.freeze.txt"

marker_temp="$INSTALL_MARKER.tmp.$$"
{
    echo "SGLANG_COMMIT=$SGLANG_COMMIT"
    echo "RUST_VERSION=$RUST_VERSION"
    echo "ROUTER_VERSION=$ROUTER_VERSION"
    echo "PBS_JOBID=$PBS_JOBID"
    echo "COMPLETED_AT=$(date --iso-8601=seconds)"
} >"$marker_temp"
mv "$marker_temp" "$INSTALL_MARKER"

echo "runtime_root=$RUNTIME_ROOT"
echo "source_commit=$ACTUAL_COMMIT"
echo "python_freeze=$RUNTIME_ROOT/environment.freeze.txt"
echo "K3_INSTALL_OK"
