#!/bin/bash

# Run deliberately incorrect PTX fixtures through the hidden verifier on one
# genuinely idle B200. A fixture passes this meta-test only when the verifier
# reports reward=0 and correctness=0 without an infrastructure failure.
#
# The caller must apply an outer timeout below Catalyst's 2.5-hour per-process
# limit. This script is intentionally separate from the Oracle queue so the
# canonical task directories never need to be edited or copied.
set -euo pipefail

benchmark_root=${PTX_BENCH_ROOT:-/home/zhuominc/rockyeast-work/PTX-benchmark}
sif_path=${PTX_BENCH_SIF:-/home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v2.sif}
poll_seconds=${PTX_BENCH_POLL_SECONDS:-45}
memory_threshold_mb=${PTX_BENCH_IDLE_MEMORY_MB:-256}
stable_seconds=${PTX_BENCH_IDLE_STABLE_SECONDS:-20}
lock_dir=${PTX_BENCH_GPU_LOCK_DIR:-/tmp}
selected_gpu=
run_stamp=${PTX_BENCH_NEGATIVE_RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
evidence_root=${PTX_BENCH_NEGATIVE_EVIDENCE_ROOT:-$benchmark_root/jobs/direct-ptx-negative-fixtures-$run_stamp}
mkdir -p "$evidence_root"
printf 'Negative-fixture evidence root: %s\n' "$evidence_root"

# task:relative-fixture-path. Explicit arguments run only the requested
# fixtures; no arguments preserve the historical overnight regression queue.
if (( $# > 0 )); then
  fixtures=("$@")
else
  fixtures=(
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128:tests/reference_t1.ptx
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128:tests/reference_t6.ptx
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128:tests/fixtures/final-checkpoint-only.ptx
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128:tests/fixtures/wrong-accepted.ptx
    ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128:tests/fixtures/wrong-head-map.ptx
    ptx-flash-attention-backward-bf16-s128-d128:tests/fixtures/only-dq.ptx
    ptx-flash-attention-backward-bf16-s128-d128:tests/fixtures/depends-on-prezero.ptx
    ptx-block-sparse-attention-forward-bf16-s1024-d128-b16:tests/fixtures/first-block-only.ptx
    ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16:tests/fixtures/ignore-page-table.ptx
    ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16:tests/fixtures/non-causal.ptx
    ptx-gemm-rowwise-fp8e4m3-m128-n128-k128:tests/fixtures/wrong-scale.ptx
  )
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
        mkdir -p "$lock_dir"
        # Keep descriptor 9 open until release_gpu is called. Harbor uses the
        # same advisory lock path, preventing two PTX-Bench runs from colliding.
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
    printf '%s no stable idle B200 for negative fixtures; retrying in %s seconds\n' \
      "$(date --iso-8601=seconds)" "$poll_seconds" >&2
    sleep "$poll_seconds"
  done
}

release_gpu() {
  flock -u 9 || true
  exec 9>&-
}

run_fixture() {
  local gpu=$1 task=$2 relative_fixture=$3
  local task_dir fixture run_dir app_dir logs_dir status reward correctness
  local fixture_slug evidence_dir

  task_dir="$benchmark_root/tasks/$task"
  fixture="$task_dir/$relative_fixture"
  if [[ ! -f $fixture ]]; then
    printf 'Missing negative fixture: %s\n' "$fixture" >&2
    return 2
  fi

  run_dir=$(mktemp -d "/tmp/ptx-negative-${task}.XXXXXX")
  app_dir="$run_dir/app"
  logs_dir="$run_dir/logs"
  fixture_slug=${relative_fixture//\//__}
  evidence_dir="$evidence_root/$task/$fixture_slug"
  mkdir -p "$app_dir" "$logs_dir"
  if [[ $task == ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128 \
      && $relative_fixture == tests/fixtures/* ]]; then
    # These semantic fixtures intentionally contain only the bad T6 entry.
    # Prepend the valid T1 entry at run time so the combined-task verifier
    # reaches the T6 semantic checks instead of rejecting a missing T1 symbol.
    python3 - "$task_dir/tests/reference_t1.ptx" "$fixture" \
      "$app_dir/kernel.ptx" <<'PY'
from pathlib import Path
import sys


def body(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    marker = ".address_size 64"
    _, separator, remainder = text.partition(marker)
    if not separator:
        raise SystemExit(f"missing PTX header marker in {path}")
    return remainder.lstrip("\n")


t1_path, bad_t6_path, output_path = sys.argv[1:]
merged = (
    ".version 9.2\n"
    ".target sm_100a\n"
    ".address_size 64\n\n"
    + body(t1_path)
    + "\n"
    + body(bad_t6_path)
)
Path(output_path).write_text(merged, encoding="utf-8")
PY
  else
    install -m 0644 "$fixture" "$app_dir/kernel.ptx"
  fi

  status=0
  # Match Harbor's Singularity lifecycle: writable fakeroot overlay, contained
  # mounts, task environment staging, and setup.sh before verifier execution.
  # A bare SIF intentionally does not contain the task-specific manifest.
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

  if (( status != 0 )); then
    mkdir -p "$evidence_dir/logs"
    install -m 0644 "$fixture" "$evidence_dir/kernel.ptx"
    cp -a "$logs_dir/." "$evidence_dir/logs/"
    python3 - "$evidence_dir/outcome.json" "$task" "$relative_fixture" \
      infrastructure_failure "$status" <<'PY'
import json, sys
path, task, fixture, outcome, status = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"task": task, "fixture": fixture, "outcome": outcome,
               "process_status": int(status)}, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    printf 'INFRA_FAILURE task=%s fixture=%s status=%s logs=%s\n' \
      "$task" "$relative_fixture" "$status" "$evidence_dir/logs" >&2
    return "$status"
  fi
  if [[ ! -f $logs_dir/verifier/reward.json ]]; then
    mkdir -p "$evidence_dir/logs"
    install -m 0644 "$fixture" "$evidence_dir/kernel.ptx"
    cp -a "$logs_dir/." "$evidence_dir/logs/"
    printf 'INFRA_FAILURE task=%s fixture=%s missing reward.json logs=%s\n' \
      "$task" "$relative_fixture" "$evidence_dir/logs" >&2
    return 3
  fi

  read -r reward correctness < <(python3 - "$logs_dir/verifier/reward.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("reward"), payload.get("correctness"))
PY
  )
  mkdir -p "$evidence_dir/logs"
  install -m 0644 "$fixture" "$evidence_dir/kernel.ptx"
  cp -a "$logs_dir/." "$evidence_dir/logs/"
  if [[ $reward != 0 && $reward != 0.0 ]] || [[ $correctness != 0 && $correctness != 0.0 ]]; then
    python3 - "$evidence_dir/outcome.json" "$task" "$relative_fixture" \
      false_accept "$reward" "$correctness" <<'PY'
import json, sys
path, task, fixture, outcome, reward, correctness = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"task": task, "fixture": fixture, "outcome": outcome,
               "reward": float(reward), "correctness": float(correctness)},
              handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    printf 'FALSE_ACCEPT task=%s fixture=%s reward=%s correctness=%s logs=%s\n' \
      "$task" "$relative_fixture" "$reward" "$correctness" "$evidence_dir/logs" >&2
    return 4
  fi

  python3 - "$evidence_dir/outcome.json" "$task" "$relative_fixture" \
    rejected "$reward" "$correctness" <<'PY'
import json, sys
path, task, fixture, outcome, reward, correctness = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"task": task, "fixture": fixture, "outcome": outcome,
               "reward": float(reward), "correctness": float(correctness)},
              handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

  printf 'REJECTED task=%s fixture=%s reward=%s correctness=%s\n' \
    "$task" "$relative_fixture" "$reward" "$correctness"
  rm -rf -- "$run_dir"
}

for item in "${fixtures[@]}"; do
  task=${item%%:*}
  fixture=${item#*:}
  fixture_slug=${fixture//\//__}
  existing_outcome="$evidence_root/$task/$fixture_slug/outcome.json"
  if [[ -f $existing_outcome ]] && python3 - "$existing_outcome" <<'PY'
import json
import sys

raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("outcome") == "rejected" else 1)
PY
  then
    printf 'SKIP previously rejected task=%s fixture=%s\n' "$task" "$fixture"
    continue
  fi
  while true; do
    choose_and_lock_gpu
    gpu=$selected_gpu
    printf '%s testing %s/%s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$task" "$fixture" "$gpu"
    if run_fixture "$gpu" "$task" "$fixture"; then
      release_gpu
      break
    else
      status=$?
    fi
    release_gpu
    if (( status == 4 )); then
      # A false accept is a real verifier bug. Preserve logs and stop instead
      # of repeatedly consuming GPU time.
      exit "$status"
    fi
    printf '%s fixture run hit infrastructure/race status %s; returning to wait\n' \
      "$(date --iso-8601=seconds)" "$status" >&2
    sleep "$poll_seconds"
  done
done
