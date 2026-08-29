#!/bin/bash

# Wait for one genuinely idle B200 and launch exactly one Codex/high trial.
# The caller must apply an outer timeout below Catalyst's 2.5-hour process cap.
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s TASK_NAME\n' "$0" >&2
  exit 2
fi

task=$1
case "$task" in
  ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128|\
  ptx-flash-attention-backward-bf16-s128-d128|\
  ptx-block-sparse-attention-forward-bf16-s1024-d128-b16|\
  ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16|\
  ptx-gemm-rowwise-fp8e4m3-m128-n128-k128)
    ;;
  *)
    printf 'Task is not in the overnight calibration allowlist: %s\n' "$task" >&2
    exit 2
    ;;
esac

harbor_root=${HARBOR_ROOT:-/home/zhuominc/rockyeast-work/harbor-ptx}
benchmark_root=${PTX_BENCH_ROOT:-/home/zhuominc/rockyeast-work/PTX-benchmark}
sif_path=${PTX_BENCH_SIF:-/home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v2.sif}
poll_seconds=${PTX_BENCH_POLL_SECONDS:-45}
memory_threshold_mb=${PTX_BENCH_IDLE_MEMORY_MB:-256}

choose_gpu() {
  local index used util confirmed_used confirmed_util
  while true; do
    index=0
    while IFS=',' read -r used util; do
      used=${used//[[:space:]]/}
      util=${util//[[:space:]]/}
      if [[ $used =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] \
          && (( used < memory_threshold_mb && util == 0 )); then
        sleep 20
        confirmed_used=$(nvidia-smi --id="$index" --query-gpu=memory.used \
          --format=csv,noheader,nounits | tr -d '[:space:]')
        confirmed_util=$(nvidia-smi --id="$index" --query-gpu=utilization.gpu \
          --format=csv,noheader,nounits | tr -d '[:space:]')
        if [[ $confirmed_used =~ ^[0-9]+$ && $confirmed_util =~ ^[0-9]+$ ]] \
            && (( confirmed_used < memory_threshold_mb && confirmed_util == 0 )); then
          printf '%s\n' "$index"
          return 0
        fi
        break
      fi
      index=$((index + 1))
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)
    printf '%s no stable idle B200 for Codex trial; retrying in %s seconds\n' \
      "$(date --iso-8601=seconds)" "$poll_seconds" >&2
    sleep "$poll_seconds"
  done
}

while true; do
  gpu=$(choose_gpu)
  stamp=$(date +%Y%m%d-%H%M%S)
  job_name="${task}-b200-codex-gpt56-high-online-docs-${stamp}"
  printf '%s starting Codex/high/online-docs %s on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$task" "$gpu"
  if (
    cd "$harbor_root"
    .venv/bin/harbor run \
      --path "$benchmark_root/tasks/$task" \
      --agent codex \
      --model gpt-5.6-sol \
      --agent-env CODEX_FORCE_AUTH_JSON=true \
      --agent-kwarg reasoning_effort=high \
      --agent-kwarg web_search=live \
      --allow-agent-host chatgpt.com \
      --allow-agent-host '*.chatgpt.com' \
      --env singularity \
      --override-gpus 1 \
      --environment-kwarg singularity_nv=true \
      --environment-kwarg "singularity_gpu_ids=[$gpu]" \
      --environment-kwarg "singularity_gpu_memory_threshold_mb=$memory_threshold_mb" \
      --environment-kwarg "singularity_image_path=$sif_path" \
      --n-attempts 1 \
      --n-concurrent 1 \
      --jobs-dir "$benchmark_root/jobs" \
      --job-name "$job_name" \
      --yes
  ); then
    exit 0
  fi
  printf '%s GPU %s was claimed before Harbor allocation; returning to wait\n' \
    "$(date --iso-8601=seconds)" "$gpu" >&2
  sleep "$poll_seconds"
done
