#!/usr/bin/env bash

k3_die() {
    echo "ERROR: $*" >&2
    exit 1
}

k3_require_positive_integer() {
    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        k3_die "$1 must be a positive integer: $2"
    fi
}

k3_load_config() {
    local config_path="$1"
    if [[ ! -f "$config_path" ]]; then
        k3_die "config file not found: $config_path"
    fi
    config_path=$(realpath "$config_path")

    # shellcheck source=/dev/null
    source "$config_path"

    local variable_name
    for variable_name in ABCI_PROJECT RUNTIME_ROOT MODEL_DIR; do
        if [[ -z ${!variable_name:-} || ${!variable_name:-} == *CHANGE_ME* ]]; then
            k3_die "set a valid value for $variable_name in $config_path"
        fi
    done
    for variable_name in RUNTIME_ROOT MODEL_DIR; do
        if [[ ${!variable_name} != /* ]]; then
            k3_die "$variable_name must be an absolute path: ${!variable_name}"
        fi
    done
    if [[ "$ABCI_PROJECT" == *","* || "$config_path" == *","* ]]; then
        k3_die "ABCI_PROJECT and config path must not contain a comma"
    fi

    K3_CONFIG="$config_path"
    export K3_CONFIG
}

k3_load_build_modules() {
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
    module purge
    module load gcc/13.2.0
    module load cuda/13.0/13.0.1
    module load python/3.12/3.12.9
}

k3_load_server_modules() {
    k3_load_build_modules
    module load nccl/2.28/2.28.3-1
    module load hpcx/2.26
}
