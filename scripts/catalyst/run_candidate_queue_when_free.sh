#!/bin/bash

# Score one already-generated kernel.ptx with a canonical task's hidden
# verifier on a genuinely idle B200. This does not run an Agent and does not
# edit the canonical task. The caller must apply an outer timeout below the
# Catalyst 2.5-hour per-process limit.
set -euo pipefail

if (( $# != 3 )); then
  printf 'Usage: %s TASK_NAME CANDIDATE_PTX EVIDENCE_LABEL\n' "$0" >&2
  exit 2
fi

task=$1
candidate=$2
label=$3
benchmark_root=${PTX_BENCH_ROOT:-/home/zhuominc/rockyeast-work/PTX-benchmark}
sif_path=${PTX_BENCH_SIF:-/home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v2.sif}
poll_seconds=${PTX_BENCH_POLL_SECONDS:-45}
memory_threshold_mb=${PTX_BENCH_IDLE_MEMORY_MB:-256}
stable_seconds=${PTX_BENCH_IDLE_STABLE_SECONDS:-20}
lock_dir=${PTX_BENCH_GPU_LOCK_DIR:-/tmp}
task_dir="$benchmark_root/tasks/$task"
stamp=${PTX_BENCH_CANDIDATE_RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
evidence_dir="$benchmark_root/jobs/candidate-verification-$label-$stamp"
selected_gpu=

if [[ ! -f $candidate ]]; then
  printf 'Candidate PTX does not exist: %s\n' "$candidate" >&2
  exit 2
fi
if [[ ! -d $task_dir/tests || ! -d $task_dir/environment/files ]]; then
  printf 'Canonical task is incomplete: %s\n' "$task_dir" >&2
  exit 2
fi

gpu_memory_mb() {
  nvidia-smi --id="$1" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]'
}

gpu_utilization() {
  nvidia-smi --id="$1" --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits | tr -d '[:space:]'
}

choose_and_lock_gpu() {
  local index used util confirmed_used confirmed_util lock_file
  while true; do
    index=0
    while IFS=',' read -r used util; do
      used=${used//[[:space:]]/}
      util=${util//[[:space:]]/}
      if [[ $used =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] \
          && (( used < memory_threshold_mb && util == 0 )); then
        lock_file="$lock_dir/harbor-singularity-gpu-$index.lock"
        exec 9>"$lock_file"
        if ! flock -n 9; then
          exec 9>&-
          index=$((index + 1))
          continue
        fi
        sleep "$stable_seconds"
        confirmed_used=$(gpu_memory_mb "$index")
        confirmed_util=$(gpu_utilization "$index")
        if [[ $confirmed_used =~ ^[0-9]+$ && $confirmed_util =~ ^[0-9]+$ ]] \
            && (( confirmed_used < memory_threshold_mb && confirmed_util == 0 )); then
          selected_gpu=$index
          return 0
        fi
        flock -u 9
        exec 9>&-
        break
      fi
      index=$((index + 1))
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)
    printf '%s no stable idle B200 for candidate verification; retrying in %s seconds\n' \
      "$(date --iso-8601=seconds)" "$poll_seconds" >&2
    sleep "$poll_seconds"
  done
}

choose_and_lock_gpu
gpu=$selected_gpu
run_dir=$(mktemp -d "/tmp/ptx-candidate-${task}.XXXXXX")
app_dir="$run_dir/app"
logs_dir="$run_dir/logs"
mkdir -p "$app_dir" "$logs_dir" "$evidence_dir"
install -m 0644 "$candidate" "$app_dir/kernel.ptx"

printf '%s scoring %s with canonical task %s on physical GPU %s\n' \
  "$(date --iso-8601=seconds)" "$candidate" "$task" "$gpu"
status=0
CUDA_VISIBLE_DEVICES="$gpu" \
SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu" \
APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" \
timeout 1200 singularity exec --nv \
  --no-mount home,tmp,bind-paths \
  --pwd / \
  --writable-tmpfs \
  --fakeroot \
  --containall \
  --bind "$task_dir/environment/files:/harbor-staging:ro" \
  --bind "$task_dir/tests:/tests:ro" \
  --bind "$app_dir:/app" \
  --bind "$logs_dir:/logs" \
  "$sif_path" bash -lc \
    'export HARBOR_STAGING=/harbor-staging; source "$HARBOR_STAGING/setup.sh"; bash /tests/test.sh' \
  || status=$?

install -m 0644 "$candidate" "$evidence_dir/kernel.ptx"
cp -a "$logs_dir/." "$evidence_dir/logs"
candidate_sha=$(sha256sum "$candidate" | awk '{print $1}')
task_snapshot_sha=$(cd "$task_dir" && find . -type f -not -path '*/__pycache__/*' -not -name '*.pyc' \
  -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')

python3 - "$evidence_dir/outcome.json" "$task" "$candidate" "$candidate_sha" \
  "$task_snapshot_sha" "$gpu" "$status" "$logs_dir/verifier/reward.json" <<'PY'
import json
import os
import sys

path, task, candidate, candidate_sha, task_snapshot_sha, gpu, status, reward_path = sys.argv[1:]
payload = {
    "task": task,
    "candidate": candidate,
    "candidate_sha256": candidate_sha,
    # This is only a local snapshot digest. It is deliberately not named a
    # Harbor checksum because Harbor 0.13.2 uses a different checksum recipe.
    "task_snapshot_sha256": task_snapshot_sha,
    "physical_gpu": int(gpu),
    "process_status": int(status),
}
if os.path.exists(reward_path):
    with open(reward_path, encoding="utf-8") as handle:
        reward = json.load(handle)
    payload["verifier"] = reward
    payload["outcome"] = "scored"
else:
    payload["outcome"] = "infrastructure_failure"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

flock -u 9 || true
exec 9>&-
printf 'Candidate evidence: %s\n' "$evidence_dir"
if (( status != 0 )) || [[ ! -f $logs_dir/verifier/reward.json ]]; then
  exit 3
fi
