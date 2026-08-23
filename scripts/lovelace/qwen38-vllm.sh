#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${QWEN_CONTAINER_NAME:-ptx-bench-qwen38-27b-fp8}"
IMAGE="${QWEN_VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}"
MODEL_REVISION="${QWEN_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.8-27b-fp8}"
GPU_INDEX="${QWEN_GPU_INDEX:-2}"
HOST_BIND="${QWEN_HOST_BIND:-0.0.0.0}"
HOST_PORT="${QWEN_HOST_PORT:-8022}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
CACHE_VOLUME="${QWEN_CACHE_VOLUME:-ptx-bench-qwen38-hf-cache}"
VLLM_CACHE_VOLUME="${QWEN_VLLM_CACHE_VOLUME:-ptx-bench-qwen38-vllm-cache}"

usage() {
    cat <<'EOF'
Usage: qwen38-vllm.sh {start|stop|status|logs|health}

Environment overrides:
  QWEN_GPU_INDEX                 GPU index, default: 2
  QWEN_HOST_BIND                 bind address, default: 0.0.0.0
  QWEN_HOST_PORT                 host port, default: 8022
  QWEN_MAX_MODEL_LEN             context window, default: 131072
  QWEN_MAX_NUM_SEQS              concurrent sequences, default: 1
  QWEN_GPU_MEMORY_UTILIZATION    vLLM GPU fraction, default: 0.90
  QWEN_ENABLE_THINKING           default chat-template thinking mode, default: false
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

container_exists() {
    docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
    [ "$(docker container inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" = "true" ]
}

check_gpu_free() {
    local memory_used
    local compute_pids

    memory_used="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    compute_pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"

    if [ -n "${compute_pids}" ] || [ "${memory_used}" -gt 1024 ]; then
        echo "GPU ${GPU_INDEX} is not free: memory.used=${memory_used} MiB pids=${compute_pids:-none}" >&2
        exit 1
    fi
}

check_port_free() {
    if ss -H -ltn "sport = :${HOST_PORT}" | grep -q .; then
        echo "TCP port ${HOST_PORT} is already in use" >&2
        exit 1
    fi
}

start_server() {
    require_command docker
    require_command nvidia-smi
    require_command ss

    if container_running; then
        echo "${CONTAINER_NAME} is already running"
        return
    fi

    check_gpu_free
    check_port_free

    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        docker pull "${IMAGE}"
    fi

    docker volume create "${CACHE_VOLUME}" >/dev/null
    docker volume create "${VLLM_CACHE_VOLUME}" >/dev/null

    if container_exists; then
        docker start "${CONTAINER_NAME}" >/dev/null
    else
        docker run -d \
            --name "${CONTAINER_NAME}" \
            --gpus "device=${GPU_INDEX}" \
            --ipc=host \
            --shm-size=16g \
            -p "${HOST_BIND}:${HOST_PORT}:8000" \
            -v "${CACHE_VOLUME}:/root/.cache/huggingface" \
            -v "${VLLM_CACHE_VOLUME}:/root/.cache/vllm" \
            "${IMAGE}" \
            "${MODEL}" \
            --revision "${MODEL_REVISION}" \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --host 0.0.0.0 \
            --port 8000 \
            --language-model-only \
            --trust-remote-code \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens 8192 \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --default-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_xml >/dev/null
    fi

    echo "Started ${CONTAINER_NAME} on GPU ${GPU_INDEX}, port ${HOST_PORT}"
    echo "First start downloads the model and may take several minutes."
    echo "Follow progress with: $0 logs"
}

stop_server() {
    require_command docker
    if container_running; then
        docker stop --time 30 "${CONTAINER_NAME}" >/dev/null
        echo "Stopped ${CONTAINER_NAME}; model files remain in Docker volume ${CACHE_VOLUME}."
    else
        echo "${CONTAINER_NAME} is not running"
    fi
}

show_status() {
    require_command docker
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}'
    nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
}

show_logs() {
    require_command docker
    docker logs --tail 200 -f "${CONTAINER_NAME}" 2>&1
}

health_check() {
    require_command curl
    curl --fail --silent --show-error \
        "http://127.0.0.1:${HOST_PORT}/v1/models"
    printf '\n'
}

case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    health)
        health_check
        ;;
    *)
        usage
        exit 2
        ;;
esac
