#!/usr/bin/env python3
"""Preflight paired inputs and tokenizer lengths for the two same-flow arms."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from projection_train.action_space import ACTION_LIST
from projection_train.grouped_dataset import DEFAULT_MOVEMENT_KEYS, GroupedTrafficActionDataset
from projection_train.prompt_utils import (
    MOVEMENT_KEYS,
    build_task_prompt,
    serialize_compact_movement_history,
)
from projection_train.train_projection_bc import split_dataset


SOURCES = {
    "Jinan-1": ("data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl", "jinan", "anon_3_4_jinan_real.json"),
    "Hangzhou-1": ("data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl", "hangzhou", "anon_4_4_hangzhou_real.json"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/mnt/data/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    assert tuple(DEFAULT_MOVEMENT_KEYS) == MOVEMENT_KEYS
    report = {"seed": 2024, "max_length": 512, "action_set": ACTION_LIST, "scenarios": {}}
    examples = {}
    for scenario, (path, dataset, flow) in SOURCES.items():
        source = GroupedTrafficActionDataset(path, seq_len=16, valid_actions=ACTION_LIST)
        train, val = split_dataset(source, val_ratio=0.1, seed=2024, split_strategy="stratified", validation_purge_samples=0)
        assert all(flow.removesuffix(".json") in item.snapshot_path for item in source.samples)
        train_indices_hash = hashlib.sha256(json.dumps(train.indices).encode()).hexdigest()
        val_indices_hash = hashlib.sha256(json.dumps(val.indices).encode()).hexdigest()
        lengths = []
        rng = random.Random(2024)
        chosen = set(rng.sample(range(len(source)), min(5, len(source))))
        paired = []
        for index, sample in enumerate(source.samples):
            sequence = sample.sequence
            assert len(sequence) == 16 and all(len(row) == 8 for row in sequence)
            assert sample.prompt_context and all(tag in sample.prompt_context for tag in ("maxQ", "maxW", "totalQ"))
            assert sample.expert_action in ACTION_LIST
            assert set(sample.candidate_action_waits).issubset(ACTION_LIST)
            history = serialize_compact_movement_history(sequence)
            decoded = [[float(value) for value in row.split()] for row in history.splitlines()[1:]]
            assert decoded == sequence
            common = build_task_prompt(sample.raw_prompt, mode="information_matched", action_tags=ACTION_LIST,
                                       traffic_context=sample.prompt_context)
            text_prompt = common + "\n" + history
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": text_prompt}], tokenize=False,
                                                     add_generation_prompt=True)
            original_tokens = len(tokenizer(rendered, add_special_tokens=True, truncation=False).input_ids)
            truncated_tokens = len(tokenizer(rendered, add_special_tokens=True, truncation=True, max_length=512).input_ids)
            lengths.append(original_tokens)
            if index in chosen:
                paired.append({
                    "index": index, "intersection_id": sample.intersection_id, "snapshot_path": sample.snapshot_path,
                    "history_16x8": sequence, "movement_order": MOVEMENT_KEYS,
                    "structured_context": sample.prompt_context, "common_prompt": common,
                    "text_input": text_prompt, "ts2i_input": {"history_16x8": sequence, "prompt": common},
                    "action_candidates": ACTION_LIST, "expert_action": sample.expert_action,
                    "candidate_action_waits": sample.candidate_action_waits,
                    "candidate_action_metrics": sample.candidate_action_metrics,
                    "original_tokens": original_tokens, "truncated_tokens": truncated_tokens,
                    "truncation": original_tokens != truncated_tokens,
                })
        scenario_report = {
            "source_path": path, "source_sha256": sha256(path), "training_flow": flow, "test_flow": flow,
            "dataset": dataset, "samples": len(source), "train_samples": len(train), "val_samples": len(val),
            "train_indices_sha256": train_indices_hash, "val_indices_sha256": val_indices_hash,
            "split_seed": 2024, "split_strategy": "stratified", "history_shape": [16, 8],
            "movement_order": MOVEMENT_KEYS, "interval_seconds": 2, "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths), "over_512": sum(n > 512 for n in lengths),
            "sampled_token_lengths": [entry["original_tokens"] for entry in paired],
            "all_inputs_equivalent_except_history_modality": True,
        }
        report["scenarios"][scenario] = scenario_report
        examples[scenario] = paired
    (output / "information_match_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "paired_input_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(item["over_512"] for item in report["scenarios"].values()):
        raise SystemExit("Text input exceeds max_length=512; refusing to launch")


if __name__ == "__main__":
    main()
