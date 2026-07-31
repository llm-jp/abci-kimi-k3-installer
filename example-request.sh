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

curl \
    --fail-with-body \
    --silent \
    --show-error \
    --header "Content-Type: application/json" \
    --data-binary @- \
    http://127.0.0.1:31000/v1/chat/completions <<JSON
{
  "model": "$MODEL_DIR",
  "messages": [
    {
      "role": "user",
      "content": "二次方程式の解の公式を導出して下さい。"
    }
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": false
}
JSON
printf '\n'
