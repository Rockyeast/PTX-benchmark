#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${QWEN_CONTAINER_NAME:-ptx-bench-qwen38-27b-fp8}"
IMAGE="${QWEN_VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}"
MODEL_REVISION="${QWEN_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.8-27b-fp8}"
GPU_INDEX="${QWEN_GPU_INDEX:-2}"
HOST_PORT="${QWEN_HOST_PORT:-8022}"
ACCESS_MODE="${QWEN_ACCESS_MODE:-docker-bridge}"
DOCKER_BRIDGE_NETWORK="${QWEN_DOCKER_BRIDGE_NETWORK:-bridge}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"
CACHE_VOLUME="${QWEN_CACHE_VOLUME:-ptx-bench-qwen38-hf-cache}"
VLLM_CACHE_VOLUME="${QWEN_VLLM_CACHE_VOLUME:-ptx-bench-qwen38-vllm-cache}"
API_KEY_FILE="${QWEN_API_KEY_FILE:-${HOME}/.config/ptx-bench/qwen38-api-key}"
HARBOR_ENV_FILE="${QWEN_HARBOR_ENV_FILE:-${HOME}/.config/ptx-bench/qwen38-harbor.env}"

usage() {
    cat <<'EOF'
Usage: qwen38-vllm.sh {start|stop|status|logs|health|audit|rotate-key}

Environment overrides:
  QWEN_GPU_INDEX                 GPU index, default: 2
  QWEN_HOST_PORT                 host port, default: 8022
  QWEN_ACCESS_MODE               docker-bridge (default) or loopback
  QWEN_DOCKER_BRIDGE_NETWORK     Docker network used by Harbor egress, default: bridge
  QWEN_MAX_MODEL_LEN             context window, default: 131072
  QWEN_MAX_NUM_SEQS              concurrent sequences, default: 1
  QWEN_GPU_MEMORY_UTILIZATION    vLLM GPU fraction, default: 0.90
  QWEN_ENABLE_THINKING           default chat-template thinking mode, default: false
  QWEN_API_KEY_FILE              server API key file, default: ~/.config/ptx-bench/qwen38-api-key
  QWEN_HARBOR_ENV_FILE           Harbor client env file, default: ~/.config/ptx-bench/qwen38-harbor.env
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

docker_bridge_gateway() {
    docker network inspect "${DOCKER_BRIDGE_NETWORK}" \
        -f '{{(index .IPAM.Config 0).Gateway}}'
}

resolve_host_bind() {
    case "${ACCESS_MODE}" in
        docker-bridge)
            docker_bridge_gateway
            ;;
        loopback)
            printf '127.0.0.1\n'
            ;;
        *)
            echo "Invalid QWEN_ACCESS_MODE=${ACCESS_MODE}; expected docker-bridge or loopback" >&2
            exit 1
            ;;
    esac
}

ensure_api_key() {
    require_command openssl

    local key_dir
    key_dir="$(dirname "${API_KEY_FILE}")"
    mkdir -p "${key_dir}"
    chmod 700 "${key_dir}"

    if [ ! -s "${API_KEY_FILE}" ]; then
        umask 077
        openssl rand -hex 32 > "${API_KEY_FILE}"
        echo "Created API key file: ${API_KEY_FILE}"
    fi
    chmod 600 "${API_KEY_FILE}"

    if [ -z "$(tr -d '\r\n' < "${API_KEY_FILE}")" ]; then
        echo "API key file is empty: ${API_KEY_FILE}" >&2
        exit 1
    fi
}

write_harbor_env_file() {
    local host_bind
    local env_dir
    local temporary_env
    host_bind="$(resolve_host_bind)"
    env_dir="$(dirname "${HARBOR_ENV_FILE}")"
    mkdir -p "${env_dir}"
    chmod 700 "${env_dir}"
    temporary_env="$(mktemp "${HARBOR_ENV_FILE}.tmp.XXXXXX")"
    chmod 600 "${temporary_env}"
    {
        printf 'OPENAI_API_KEY='
        tr -d '\r\n' < "${API_KEY_FILE}"
        printf '\nOPENAI_BASE_URL=http://%s:%s/v1\n' "${host_bind}" "${HOST_PORT}"
    } > "${temporary_env}"
    mv "${temporary_env}" "${HARBOR_ENV_FILE}"
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

    local host_bind
    host_bind="$(resolve_host_bind)"

    if container_running; then
        audit_server
        return
    fi

    check_gpu_free
    check_port_free
    ensure_api_key
    write_harbor_env_file

    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        docker pull "${IMAGE}"
    fi

    docker volume create "${CACHE_VOLUME}" >/dev/null
    docker volume create "${VLLM_CACHE_VOLUME}" >/dev/null

    # A stopped container may retain an old public binding or an obsolete key.
    # Always recreate it; the named cache volumes are preserved.
    if container_exists; then
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi

    # vLLM listens on all interfaces only inside its container namespace. The
    # host-side publish remains restricted to loopback or the bridge gateway.
    docker run -d \
            --name "${CONTAINER_NAME}" \
            --restart=no \
            --network "${DOCKER_BRIDGE_NETWORK}" \
            --gpus "device=${GPU_INDEX}" \
            --security-opt no-new-privileges:true \
            --shm-size=16g \
            -p "${host_bind}:${HOST_PORT}:8000" \
            -v "${CACHE_VOLUME}:/root/.cache/huggingface" \
            -v "${VLLM_CACHE_VOLUME}:/root/.cache/vllm" \
            -v "${API_KEY_FILE}:/run/secrets/qwen38-api-key:ro" \
            --label "ptx-bench.access-mode=${ACCESS_MODE}" \
            --label "ptx-bench.host-bind=${host_bind}" \
            --entrypoint /bin/sh \
            "${IMAGE}" \
            -c 'export VLLM_API_KEY="$(tr -d "\r\n" < /run/secrets/qwen38-api-key)"; exec vllm serve "$@"' \
            qwen-vllm \
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

    echo "Started ${CONTAINER_NAME} on GPU ${GPU_INDEX}, ${host_bind}:${HOST_PORT}"
    echo "Harbor client credentials: ${HARBOR_ENV_FILE} (mode 0600)"
    echo "First start downloads the model and may take several minutes."
    echo "Follow progress with: $0 logs"
}

stop_server() {
    require_command docker
    if container_running; then
        docker stop --timeout 30 "${CONTAINER_NAME}" >/dev/null
        echo "Stopped ${CONTAINER_NAME}; model files remain in Docker volume ${CACHE_VOLUME}."
    else
        echo "${CONTAINER_NAME} is not running"
    fi
}

show_status() {
    require_command docker
    local host_bind
    host_bind="$(resolve_host_bind)"
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}'
    printf 'Expected private binding: %s:%s (%s)\n' \
        "${host_bind}" "${HOST_PORT}" "${ACCESS_MODE}"
    if [ -e "${API_KEY_FILE}" ]; then
        stat -c 'API key: %n mode=%a owner=%U' "${API_KEY_FILE}"
    fi
    nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
}

show_logs() {
    require_command docker
    docker logs --tail 200 -f "${CONTAINER_NAME}" 2>&1 \
        | sed -E 's/[0-9a-f]{64}/[REDACTED_API_KEY]/g'
}

health_check() {
    require_command curl
    ensure_api_key
    local host_bind
    host_bind="$(resolve_host_bind)"
    curl --fail --silent --show-error \
        -H "Authorization: Bearer $(tr -d '\r\n' < "${API_KEY_FILE}")" \
        "http://${host_bind}:${HOST_PORT}/v1/models"
    printf '\n'
}

audit_server() {
    require_command curl
    require_command docker

    if ! container_running; then
        echo "${CONTAINER_NAME} is not running" >&2
        exit 1
    fi

    local host_bind
    local expected_binding
    local actual_binding
    local unauthenticated_status
    local key
    local args
    host_bind="$(resolve_host_bind)"
    expected_binding="${host_bind}:${HOST_PORT}"
    actual_binding="$(docker port "${CONTAINER_NAME}" 8000/tcp)"
    if [ "${actual_binding}" != "${expected_binding}" ]; then
        echo "Unsafe or unexpected binding: ${actual_binding}; expected ${expected_binding}" >&2
        exit 1
    fi

    unauthenticated_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
        "http://${host_bind}:${HOST_PORT}/v1/models")"
    if [ "${unauthenticated_status}" != "401" ] && [ "${unauthenticated_status}" != "403" ]; then
        echo "Unauthenticated API request returned ${unauthenticated_status}; expected 401 or 403" >&2
        exit 1
    fi

    health_check >/dev/null
    key="$(tr -d '\r\n' < "${API_KEY_FILE}")"
    args="$(docker inspect -f '{{json .Args}}' "${CONTAINER_NAME}")"
    if grep -Fq "${key}" <<< "${args}"; then
        echo "API key leaked into container command arguments" >&2
        exit 1
    fi

    echo "Security audit passed: private bind, API authentication, no key in container argv."
}

rotate_api_key() {
    require_command openssl
    local key_dir
    local temporary_key

    if container_running; then
        docker stop --timeout 30 "${CONTAINER_NAME}" >/dev/null
    fi
    if container_exists; then
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi

    key_dir="$(dirname "${API_KEY_FILE}")"
    mkdir -p "${key_dir}"
    chmod 700 "${key_dir}"
    temporary_key="$(mktemp "${API_KEY_FILE}.tmp.XXXXXX")"
    chmod 600 "${temporary_key}"
    openssl rand -hex 32 > "${temporary_key}"
    mv "${temporary_key}" "${API_KEY_FILE}"
    echo "Rotated API key and removed the old container configuration."
    start_server
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
    audit)
        audit_server
        ;;
    rotate-key)
        rotate_api_key
        ;;
    *)
        usage
        exit 2
        ;;
esac
