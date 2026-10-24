#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = {"ATT": "test_avg_travel_time_over", "AQL": "test_avg_queue_len_over", "AWT": "test_avg_waiting_time_over"}
SCENARIOS = ("Jinan-1", "Hangzhou-1")
ARMS = ("Text", "TS2I")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    train_max_length = config.get("train_max_length", config.get("max_length", 512))
    online_max_length = config.get("online_max_length", config.get("max_length", 512))
    rows = {}
    for scenario in SCENARIOS:
        rows[scenario] = {}
        for arm in ARMS:
            path = root / "online" / scenario / arm / "result.json"
            if path.exists():
                payload = json.loads(path.read_text())
                result = payload.get("results", payload)
                rows[scenario][arm] = {name: float(result[key]) for name, key in METRICS.items()}
    complete = all(len(rows[s]) == 2 for s in SCENARIOS)
    summary = {"complete": complete, "seed": 2024, "results": rows}
    if complete:
        summary["means"] = {arm: {m: sum(rows[s][arm][m] for s in SCENARIOS) / 2 for m in METRICS} for arm in ARMS}
    (root / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Information-Matched Representation Ablation", "", "This supplementary information-matched ablation uses a single seed (2024) and is designed to isolate representation modality rather than estimate run-to-run statistical uncertainty.", "", "## Protocol", "", "The paper's original text-only/TS2I-only/full representation ablation changed modality together with prompt and rule content, so it did not hold the full 16x8 history and structured cues fixed. The later repository matched-representation script added those controls but spans extra arms, flows, seeds, and max_length=1024; it is not this two-scenario, seed-2024 experiment. Here both arms use the same raw same-flow training records, stratified split, Qwen3-VL-8B frozen backbone, Stage II mixed hard/soft/ranking/KL supervision, candidate actions, compact structured cues, control guidance, 30 s decision interval, CityFlow evaluator, and validation-regret checkpoint selection. Jinan-1 and Hangzhou-1 are each trained and evaluated on their own flow, matching the paper's primary in-flow protocol, not its separate transfer analysis.", "", "Text serializes all 16 rows in oldest-to-newest order, 2 seconds apart, with NT NL ST SL ET EL WT WL columns. TS2I receives the identical tensor through the existing visual projector. Both use the same prompt aside from that history representation. Stage I alignment is visual-only and would favor TS2I, so neither arm uses it; this isolates representation under random Stage II initialization, not the benefit of the full two-stage pipeline.", "", "Training and inference set max_length=512, batch_size=1, grad_accum_steps=16 (prior matched script: 16; unchanged because both arms fit), effective batch size=16 per single-GPU worker, bf16, 15 epochs, learning rate=1e-4, seed=2024. The audit checks every training prompt with the actual tokenizer and rejects any above 512; the text policy also rejects overlong inference prompts instead of truncating. See information_match_audit.json and paired_input_examples.json. All three metrics are lower-is-better; delta = 100 x (TS2I - Text) / Text. No significance test is applied to one seed.", "", "## Results", "", "| Scenario | Text ATT | TS2I ATT | ΔATT | Text AQL | TS2I AQL | ΔAQL | Text AWT | TS2I AWT | ΔAWT |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    audit_path = root / "information_match_audit.json"
    if not audit_path.exists():
        audit_path = root.parent / "information_match_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        details = audit["scenarios"]
    lines[8:8] = [f"Full training-input audit: Jinan-1 {details['Jinan-1']['samples']} samples, max {details['Jinan-1']['maximum_tokens']} tokens, {details['Jinan-1']['over_512']} truncated; Hangzhou-1 {details['Hangzhou-1']['samples']} samples, max {details['Hangzhou-1']['maximum_tokens']} tokens, {details['Hangzhou-1']['over_512']} truncated.", ""]
    lines = [line.replace("Training and inference set max_length=512", f"Training max_length={train_max_length}; all online evaluations max_length={online_max_length}") for line in lines]
    for scenario in (*SCENARIOS, "Mean"):
        data = summary.get("means", {}) if scenario == "Mean" else rows[scenario]
        if len(data) != 2:
            lines.append(f"| {scenario} | pending | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        values = []
        for metric in METRICS:
            text = data["Text"][metric]
            ts2i = data["TS2I"][metric]
            values += [f"{text:.2f}", f"{ts2i:.2f}", f"{100 * (ts2i - text) / text:+.2f}%" if text else "n/a"]
        lines.append("| " + scenario + " | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation", ""]
    if complete:
        wins = all(rows[s]["TS2I"][m] < rows[s]["Text"][m] for s in SCENARIOS for m in METRICS)
        lines.append("TS2I improves all three metrics in both scenarios." if wins else "The results do not uniformly support the proposed TS2I superiority claim; see the signed per-metric deltas above.")
        lines.append("This is information-matched at the observation and prompt level; the modality-specific trainable encoder capacity remains an architectural difference and a single seed cannot establish run-to-run reliability.")
    else:
        lines.append("Evaluation is incomplete. Do not infer a representation effect from partial runs.")
    gpu_assignment = config.get("gpu_assignment") or {"Jinan-1/Text": 0, "Jinan-1/TS2I": 2, "Hangzhou-1/Text": 3, "Hangzhou-1/TS2I": 0}
    lines += ["", "## Artifacts and execution", "", "Code changes: `projection_train/prompt_utils.py` adds a common compact prompt and complete history serializer; `projection_train/train_projection_bc.py` adds matched text formatting and deterministic seeding; `projection_train/run_projection_online.py` uses the same format online; `scripts/information_matched/` contains preflight, launcher, monitor, and aggregator; `tests/test_information_matched.py` checks value and prompt parity.", "", f"GPU assignment: {gpu_assignment}. See `config.json`, `status/`, `logs/`, `train/`, and `online/` for commands, process state, checkpoints, and CityFlow results. The two modalities use different trainable representation interfaces (a text prefix versus TS2I/vision projectors), so exact parameter-capacity matching is not established; this limits a purely causal attribution to modality alone."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
