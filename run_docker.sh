#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: ./run_docker.sh [COMMAND [ARG...]]

Open a Bash shell in the DUET-DINO container, or run COMMAND.

Environment variables:
  DOCKER_IMAGE     Image to run (default: duet-dino:latest)
  CHECKPOINT_DIR   Host checkpoint directory (default: <repository>/checkpoints)
  OUTPUT_DIR       Host output directory (default: <repository>/output)
  HOST_PORT        Host port mapped to container port 8000 (default: 8000)
  GPUS             Docker GPU selection (default: all)
HELP
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is required but was not found in PATH." >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${DOCKER_IMAGE:-duet-dino:latest}"
checkpoint_dir="${CHECKPOINT_DIR:-${repo_dir}/checkpoints}"
output_dir="${OUTPUT_DIR:-${repo_dir}/output}"

mkdir -p -- "$checkpoint_dir" "$output_dir"
checkpoint_dir="$(cd -- "$checkpoint_dir" && pwd)"
output_dir="$(cd -- "$output_dir" && pwd)"

# Allocate a terminal only when launched from an interactive terminal.
interactive_args=()
if [[ -t 0 && -t 1 ]]; then
    interactive_args=(-it)
fi

if [[ $# -eq 0 ]]; then
    set -- /bin/bash
fi

exec docker run --rm "${interactive_args[@]}" \
    --gpus "${GPUS:-all}" \
    -p "${HOST_PORT:-8000}:8000" \
    --mount "type=bind,source=${repo_dir},target=/workspace/duet-dino" \
    --mount "type=bind,source=${checkpoint_dir},target=/workspace/duet-dino/checkpoints" \
    --mount "type=bind,source=${output_dir},target=/workspace/duet-dino/output" \
    --workdir /workspace/duet-dino \
    "$image" "$@"
