#!/bin/bash

# Wait for a genuinely idle B200, then run the provisional Direct-PTX Oracle
# queue serially. The caller should wrap this script in a timeout below the
# server's 2.5-hour per-process limit.
set -euo pipefail

harbor_root=${HARBOR_ROOT:-/home/zhuominc/rockyeast-work/harbor-ptx}
benchmark_root=${PTX_BENCH_ROOT:-/home/zhuominc/rockyeast-work/PTX-benchmark}
sif_path=${PTX_BENCH_SIF:-/home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v2.sif}
poll_seconds=${PTX_BENCH_POLL_SECONDS:-45}
memory_threshold_mb=${PTX_BENCH_IDLE_MEMORY_MB:-256}

if (( $# > 0 )); then
  tasks=("$@")
else
  tasks=(
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128
    ptx-flash-attention-backward-bf16-s128-d128
    ptx-block-sparse-attention-forward-bf16-s1024-d128-b16
    ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16
    ptx-gemm-rowwise-fp8e4m3-m128-n128-k128
  )
fi

choose_gpu() {
  local index used confirmed
  while true; do
    index=0
    while IFS= read -r used; do
      used=${used//[[:space:]]/}
      if [[ $used =~ ^[0-9]+$ ]] && (( used < memory_threshold_mb )); then
        # Tuning programs may leave a sub-second gap between workers. Require
        # the same device to remain below the threshold before handing it to
        # Harbor; the allocator still performs the authoritative final check.
        sleep 20
        confirmed=$(nvidia-smi --id="$index" --query-gpu=memory.used \
          --format=csv,noheader,nounits)
        confirmed=${confirmed//[[:space:]]/}
        if [[ $confirmed =~ ^[0-9]+$ ]] && (( confirmed < memory_threshold_mb )); then
          printf '%s\n' "$index"
          return 0
        fi
        break
      fi
      index=$((index + 1))
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    printf '%s no B200 below %s MiB; retrying in %s seconds\n' \
      "$(date --iso-8601=seconds)" "$memory_threshold_mb" "$poll_seconds" >&2
    sleep "$poll_seconds"
  done
}

for task in "${tasks[@]}"; do
  while true; do
    gpu=$(choose_gpu)
    stamp=$(date +%Y%m%d-%H%M%S)
    job_name="${task}-b200-oracle-${stamp}"
    printf '%s starting %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$task" "$gpu"
    if (
      cd "$harbor_root"
      timeout 1800 .venv/bin/harbor run \
        --path "$benchmark_root/tasks/$task" \
        --agent oracle \
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
      break
    fi
    printf '%s GPU %s was claimed before Harbor allocation; returning to wait\n' \
      "$(date --iso-8601=seconds)" "$gpu" >&2
    sleep "$poll_seconds"
  done
done
