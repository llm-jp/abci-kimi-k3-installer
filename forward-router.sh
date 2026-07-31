#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: bash $0 ROUTER_HOST" >&2
    exit 2
fi

exec ssh \
    -N \
    -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:31000:127.0.0.1:31000 \
    "$1"
