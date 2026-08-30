import json
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from recompute_matrix_reports_offline import (
    build_edit_scope_proof_record,
    merge_authoritative_task_payload,
    project_relative_or_absolute,
    write_jsonl,
)
from evaluation.edit_scope import canonical_json_sha256


def test_frozen_task_truth_replaces_stale_ir_without_changing_model_output() -> None:
    task = {
        "sample_id": "task_123",
        "edit_query": "updated query",
        "canonical_constraint_ir": {"nodes": [{"id": "edit_logic_0"}]},
        "edit_target_constraints": [{"id": "edit_logic_0"}],
        "origin_plan": {"itinerary": [{"day": 1}]},
    }
    evaluation = {
        "sample_id": "sample_000001",
        "edit_query": "stale query",
        "canonical_constraint_ir": {"nodes": []},
        "edit_target_constraints": [],
        "edited_plan": {"itinerary": [{"day": 1}, {"day": 2}]},
        "success": True,
        "metrics": {"total_tokens": 42},
        "trace": ["immutable model trace"],
    }

    merged = merge_authoritative_task_payload(task, evaluation)

    assert merged["sample_id"] == "task_123"
    assert merged["edit_query"] == "updated query"
    assert merged["canonical_constraint_ir"]["nodes"] == [{"id": "edit_logic_0"}]
    assert merged["edit_target_constraints"] == [{"id": "edit_logic_0"}]
    assert merged["edited_plan"] == evaluation["edited_plan"]
    assert merged["success"] is True
    assert merged["metrics"] == {"total_tokens": 42}
    assert merged["trace"] == ["immutable model trace"]


def test_full_edit_scope_proof_record_is_persisted_deterministically(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    evaluation_path = project_root / "evaluation.json"
    task_path = project_root / "task.json"
    cascade = {
        "algorithm": {"version": "1.1.0"},
        "verified_removable_units": [
            {
                "witness": {
                    "candidate_plan": {"itinerary": [{"day": 1}]},
                    "gate": {"passed": True},
                }
            }
        ],
    }
    record = build_edit_scope_proof_record(
        task_id="task_123",
        method="react",
        model="model_a",
        evaluation_path=evaluation_path,
        task_path=task_path,
        cascade=cascade,
    )

    output = tmp_path / "proofs.jsonl"
    write_jsonl(output, [record])
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["task_id"] == "task_123"
    assert saved["cascade"] == cascade
    assert saved["cascade_result_sha256"] == canonical_json_sha256(cascade)
    assert output.read_text(encoding="utf-8") == (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    )


def test_temporary_proof_output_path_can_be_recorded(tmp_path: Path) -> None:
    assert project_relative_or_absolute(tmp_path / "proofs.jsonl").startswith("/")
