from projection_train import crcd_relabel


def _record(base, waits):
    return {
        "snapshot_path": "snapshot.json",
        "intersection_id": "intersection_1_1",
        "base_action": base,
        "candidate_action_waits": waits,
        "candidate_action_metrics": {},
        "expert_action_margin": 0.1,
    }


def test_prcat_weights_source_relative_counterfactual_advantage():
    record = _record("ETWT", {"ETWT": 0.8, "NTST": 0.2, "ELWL": 0.7, "NLSL": 0.6})
    aligned = {("snapshot.json", "intersection_1_1"): {"30": record, "120": record, "300": record}}

    rows = crcd_relabel.build_teacher_records(
        aligned, "prcat", {"prcat_min_weight": 0.15}
    )

    assert len(rows) == 1
    assert rows[0]["expert_action"] == "NTST"
    assert rows[0]["sample_weight"] == 0.66
    assert rows[0]["prcat_source_deviation"] is True


def test_prcat_uses_h120_costs_not_h30_costs():
    rec30 = _record("ETWT", {"ETWT": 0.1, "NTST": 0.8, "ELWL": 0.7, "NLSL": 0.6})
    rec120 = _record("ETWT", {"ETWT": 0.8, "NTST": 0.2, "ELWL": 0.7, "NLSL": 0.6})
    aligned = {("snapshot.json", "intersection_1_1"): {"30": rec30, "120": rec120, "300": rec30}}

    row = crcd_relabel.build_teacher_records(aligned, "prcat", {"prcat_min_weight": 0.15})[0]

    assert row["expert_action"] == "NTST"
    assert row["candidate_action_waits"] == rec120["candidate_action_waits"]
    assert row["sample_weight"] == 0.66
