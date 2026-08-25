#!/usr/bin/env bash
# In-network online test: Jinan pt8/hd64 source policy on Jinan real flow.
# Apples-to-apples counterpart of run_jinan_pt32_innetwork_20260813.sh:
# same data, same prompt_mode (queue_wait_short), same max_length=64, same 3 seeds.
# Only architecture differs: prefix_tokens=8, hidden_dim=64 (old) vs 32/256 (new).
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
cd "$REPO_ROOT"

CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
LLM_PATH="/home/data/model/qwen-vl-8B"
DATASET="jinan"
TRAFFIC_FILE="anon_3_4_jinan_real.json"
SEEDS="2024 42 123"
PY_BIN="${PY_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
OUT_ROOT="records/pipeline_runs/jinan_pt8_innetwork_20260813"
mkdir -p "${OUT_ROOT}"

exec >"${OUT_ROOT}/eval.log" 2>&1

echo "=== In-network Jinan online test: pt8/hd64 ==="
echo "started_at=$(date -Is)"
echo "ckpt=${CKPT} dataset=${DATASET} traffic=${TRAFFIC_FILE}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] seed=${seed} starting"
  "${PY_BIN}" projection_train/run_projection_online.py \
    --memo "jinan_pt8_innetwork" \
    --projection_ckpt "${CKPT}" \
    --llm_path "${LLM_PATH}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --run_counts 3600 --num_rounds 1 \
    --seed "${seed}" \
    --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 \
    --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --inference_batch_size 64 \
    --bf16 \
    --result_json "${OUT_ROOT}/result_seed${seed}.json"
  echo "[$(date -Is)] seed=${seed} done rc=$?"
done

echo "[$(date -Is)] merging summary"
python3 - <<PY
import json, os, statistics
results = {}
for seed in [2024, 42, 123]:
    path = f"${OUT_ROOT}/result_seed{seed}.json"
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        r = payload.get("results", payload)
        results[f"seed_{seed}"] = {
            "ATT": r.get("test_avg_travel_time_over"),
            "AQL": r.get("test_avg_queue_len_over"),
            "AWT": r.get("test_avg_waiting_time_over"),
        }
        print(f"seed {seed}: ATT={results[f'seed_{seed}']['ATT']} AQL={results[f'seed_{seed}']['AQL']} AWT={results[f'seed_{seed}']['AWT']}")

def agg(key):
    vals = [results[s][key] for s in results if results[s][key] is not None]
    return len(vals), (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
out = {"per_seed": results}
for key in ["ATT", "AQL", "AWT"]:
    n, (m, s) = agg(key)
    out[key] = {"n": n, "mean": m, "std": s}
    print(f"{key}: mean={m:.2f} std={s:.2f} n={n}")
out["Sum_mean"] = {"n": 0, "mean": None}
with open("${OUT_ROOT}/eval_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("[$(date -Is)] summary written to ${OUT_ROOT}/eval_summary.json")
PY

echo "completed_at=$(date -Is)"
echo "status=success"
