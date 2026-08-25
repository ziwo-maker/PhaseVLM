#!/usr/bin/env bash
# Launch an LLMLight development/inference container with a balanced vLLM profile.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./dockerCityflow/run_llmlight_vllm_optimized.sh [command ...]

The default command is "sleep infinity", allowing project commands to be run
later with docker exec. Resource settings can be overridden with environment
variables:

  LLMLIGHT_CONTAINER_NAME       Container name (default: LLMlight-vllm-optimized)
  LLMLIGHT_IMAGE                Image name (default: llmlight:v1)
  LLMLIGHT_GPU_IDS              GPU IDs to expose (default: 0,1)
  LLMLIGHT_CPUSET               CPUs to pin (default: 0-15,32-47)
  LLMLIGHT_MEMORY               Cgroup memory limit (default: 192g)
  LLMLIGHT_MEMORY_RESERVATION   Cgroup memory soft reservation (default: 144g)
  LLMLIGHT_MIN_AVAILABLE_GIB    Required host available memory (default: 128)

Examples:
  ./dockerCityflow/run_llmlight_vllm_optimized.sh
  ./dockerCityflow/run_llmlight_vllm_optimized.sh bash
  ./dockerCityflow/run_llmlight_vllm_optimized.sh python run_open_LLM_with_vllm.py
  LLMLIGHT_GPU_IDS=0,1,2 LLMLIGHT_CPUSET=0-63 \
    ./dockerCityflow/run_llmlight_vllm_optimized.sh python run_open_LLM_with_vllm.py
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

readonly PROJECT_ROOT="/home/code/LLMSTSC"
readonly MODEL_ROOT="/home/data/model"
readonly RECORDED_ROOT="/data/datacode/recorded"
readonly RECORDS_ROOT="/data/records"

: "${LLMLIGHT_CONTAINER_NAME:=LLMlight-vllm-optimized}"
: "${LLMLIGHT_IMAGE:=llmlight:v1}"
: "${LLMLIGHT_GPU_IDS:=0,1}"
: "${LLMLIGHT_CPUSET:=0-15,32-47}"
: "${LLMLIGHT_MEMORY:=192g}"
: "${LLMLIGHT_MEMORY_RESERVATION:=144g}"
: "${LLMLIGHT_MIN_AVAILABLE_GIB:=128}"

if [[ $# -eq 0 ]]; then
    set -- sleep infinity
fi

for mount_path in "$PROJECT_ROOT" "$MODEL_ROOT"; do
    if [[ ! -d "$mount_path" ]]; then
        printf 'Required mount path does not exist: %s\n' "$mount_path" >&2
        exit 1
    fi
done

mount_args=(
    --mount "type=bind,src=$PROJECT_ROOT,dst=$PROJECT_ROOT"
    --mount "type=bind,src=$MODEL_ROOT,dst=$MODEL_ROOT,readonly"
)
for mount_path in "$RECORDED_ROOT" "$RECORDS_ROOT"; do
    if [[ -d "$mount_path" ]]; then
        mount_args+=(--mount "type=bind,src=$mount_path,dst=$mount_path")
    else
        printf 'Optional data mount is unavailable and will be skipped: %s\n' "$mount_path" >&2
    fi
done

if ! docker image inspect "$LLMLIGHT_IMAGE" >/dev/null 2>&1; then
    printf 'Docker image is unavailable: %s\n' "$LLMLIGHT_IMAGE" >&2
    exit 1
fi

if docker container inspect "$LLMLIGHT_CONTAINER_NAME" >/dev/null 2>&1; then
    printf 'Container already exists: %s\n' "$LLMLIGHT_CONTAINER_NAME" >&2
    printf 'Choose another LLMLIGHT_CONTAINER_NAME; this script never removes containers.\n' >&2
    exit 1
fi

if ! [[ "$LLMLIGHT_MIN_AVAILABLE_GIB" =~ ^[0-9]+$ ]]; then
    printf 'LLMLIGHT_MIN_AVAILABLE_GIB must be an integer.\n' >&2
    exit 1
fi

available_kib="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo)"
required_kib=$((LLMLIGHT_MIN_AVAILABLE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    printf 'Only %.1f GiB host memory is available; need at least %s GiB.\n' \
        "$((available_kib * 10 / 1024 / 1024))e-1" "$LLMLIGHT_MIN_AVAILABLE_GIB" >&2
    exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
IFS=',' read -r -a requested_gpu_ids <<< "$LLMLIGHT_GPU_IDS"
for gpu_id in "${requested_gpu_ids[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]] || (( gpu_id >= gpu_count )); then
        printf 'Invalid GPU ID %s; host has %s GPU(s).\n' "$gpu_id" "$gpu_count" >&2
        exit 1
    fi
done

exec docker run -d \
    --name "$LLMLIGHT_CONTAINER_NAME" \
    --init \
    --gpus "\"device=$LLMLIGHT_GPU_IDS\"" \
    --network host \
    --ipc host \
    --memory "$LLMLIGHT_MEMORY" \
    --memory-reservation "$LLMLIGHT_MEMORY_RESERVATION" \
    --memory-swap "$LLMLIGHT_MEMORY" \
    --cpuset-cpus "$LLMLIGHT_CPUSET" \
    --pids-limit 1024 \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=65535:65535 \
    --ulimit stack=67108864:67108864 \
    --log-driver local \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    --restart unless-stopped \
    --workdir "$PROJECT_ROOT" \
    --env "NVIDIA_VISIBLE_DEVICES=$LLMLIGHT_GPU_IDS" \
    --env "CUDA_VISIBLE_DEVICES=$LLMLIGHT_GPU_IDS" \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env VLLM_LOGGING_LEVEL=INFO \
    --env TOKENIZERS_PARALLELISM=false \
    --env OMP_NUM_THREADS=16 \
    --env MALLOC_ARENA_MAX=2 \
    "${mount_args[@]}" \
    "$LLMLIGHT_IMAGE" \
    "$@"
