#!/usr/bin/env bash
# Build direct-urgency datasets for every discoverable city training source.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
BUILDER="$REPO_ROOT/scripts/build_direct_urgency_stage2_dataset.py"
RG_BIN=/root/.nvm/versions/node/v20.20.2/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path/rg
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/direct_urgency_city_coverage_20260820}"
DATA_ROOT="$RUN_ROOT/data"
COVERAGE_TSV="$RUN_ROOT/coverage.tsv"

mkdir -p "$DATA_ROOT"
printf 'source_path\tinput_sha256\tcity\tstatus\tlabelled_rows\teligible_rows\toutput_path\n' > "$COVERAGE_TSV"

city_for_path() {
    case "$1" in
        *hangzhou*) printf 'hangzhou' ;;
        *jinan*) printf 'jinan' ;;
        *newyork*|*new_york*|*new-york*) printf 'newyork' ;;
        *) printf 'unknown' ;;
    esac
}

audit_source() {
    local source="$1"
    jq -r '
        . as $record |
        if ($record.expert_action? // "") != "" then
          [
            1,
            if (
              ($record.phase_queue | type) == "object" and
              ($record.phase_wait | type) == "object" and
              ($record.phase_queue | has("ETWT") and has("NTST") and has("ELWL") and has("NLSL")) and
              ($record.phase_wait | has("ETWT") and has("NTST") and has("ELWL") and has("NLSL"))
            ) then 1 else 0 end
          ] | @tsv
        else
          [0, 0] | @tsv
        end
    ' "$source" | awk -F '\t' '
        { labelled += $1; eligible += $2 }
        END { printf "%d %d\n", labelled, eligible }
    '
}

candidate_sources() {
    "$RG_BIN" --files "$REPO_ROOT/data/processed" -g '*.jsonl' |
        "$RG_BIN" -i 'jinan|hangzhou|newyork|new_york|new-york'
    "$RG_BIN" --files "$REPO_ROOT/records/pipeline_runs/joint_coreset_stage12_all_20260818/datasets" -g 'stage1.jsonl' 2>/dev/null
    "$RG_BIN" --files "$REPO_ROOT/records/sim_expert_validation_search" -g '*.jsonl' |
        "$RG_BIN" -i 'jinan|hangzhou|newyork|new_york|new-york' |
        "$RG_BIN" '/(align|traffic_prompt)' || true
    "$RG_BIN" --files "$REPO_ROOT/records/dagger" -g '*.jsonl' |
        "$RG_BIN" -i 'newyork|new_york|new-york' || true
    "$RG_BIN" --files /mnt/data/LLMSTSC/generated -g 'traffic_prompt_series_2s.jsonl' 2>/dev/null |
        "$RG_BIN" -i 'newyork|new_york|new-york' || true
}

while IFS= read -r source; do
    [[ -f "$source" ]] || continue
    city="$(city_for_path "$source")"
    read -r labelled eligible < <(audit_source "$source")

    if [[ "$labelled" -eq 0 ]]; then
        printf '%s\t\t%s\tskipped_no_labels\t0\t0\t\n' "$source" "$city" >> "$COVERAGE_TSV"
        continue
    fi
    if [[ "$labelled" -ne "$eligible" ]]; then
        printf '%s\t\t%s\tskipped_missing_phase_queue_wait\t%s\t%s\t\n' \
            "$source" "$city" "$labelled" "$eligible" >> "$COVERAGE_TSV"
        continue
    fi

    input_sha="$(sha256sum "$source" | awk '{print $1}')"
    output_dir="$DATA_ROOT/$city"
    output_path="$output_dir/${input_sha}.jsonl"
    summary_path="$output_dir/${input_sha}.summary.json"
    mkdir -p "$output_dir"

    if [[ -s "$output_path" && -s "$summary_path" ]]; then
        status=reused_identical_input
    else
        "$PYTHON_BIN" "$BUILDER" \
            --input "$source" \
            --output "$output_path" \
            --summary "$summary_path"
        status=built
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$source" "$input_sha" "$city" "$status" "$labelled" "$eligible" "$output_path" >> "$COVERAGE_TSV"
done < <(candidate_sources | sort -u)

printf 'coverage=%s\n' "$COVERAGE_TSV"
