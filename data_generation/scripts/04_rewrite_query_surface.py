#!/usr/bin/env python3
"""Step 4: Rewrite surface query while preserving frozen hard/soft truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = Path(DATA_GEN_DIR).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(DATA_GEN_DIR) / "utils"))

from llm_client import load_client_from_env
from constraint_ir.adapters import build_edit_targets_from_ir
from constraint_ir.compiler import build_canonical_constraint_ir_from_structured_inputs


def _load_step2_module():
    module_path = Path(__file__).resolve().parent / "02_generate_query.py"
    spec = importlib.util.spec_from_file_location("generate_query_step2_surface_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_query_step2_surface_module"] = module
    spec.loader.exec_module(module)
    return module


step2 = _load_step2_module()


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    batch_path = Path(batch_dir)
    samples: List[Dict[str, Any]] = []
    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            with open(sample_file, "r", encoding="utf-8") as handle:
                samples.append(json.load(handle))
        except Exception as exc:
            print(f"Warning: Failed to load {sample_file}: {exc}")
    print(f"Loaded {len(samples)} samples from {batch_dir}")
    return samples


def save_sample(sample: Dict[str, Any], output_dir: str) -> None:
    sample_file = Path(output_dir) / f"{sample['sample_id']}.json"
    with open(sample_file, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)


def filter_pending_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        if status != "03_completed":
            continue
        if not isinstance(sample.get("canonical_edit_query"), str) or not sample["canonical_edit_query"].strip():
            continue
        pending.append(sample)
    print(f"Found {len(pending)} samples needing surface rewrite (from {len(samples)} total)")
    return pending


def _freeze(items: List[Dict[str, Any]]) -> List[str]:
    return sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items)


def _validate_frozen_truth_presence(sample: Dict[str, Any], final_query: str, surface_query_spec: Dict[str, Any], surface_semantic_slots: Dict[str, Any]) -> Tuple[bool, str]:
    canonical_constraint_ir = sample.get("canonical_constraint_ir")
    if not isinstance(canonical_constraint_ir, dict) or not canonical_constraint_ir:
        return False, "missing_canonical_constraint_ir"
    expected_hard = sample.get("edit_target_constraints") if isinstance(sample.get("edit_target_constraints"), list) else []
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
    surface_ir = build_canonical_constraint_ir_from_structured_inputs(
        template_text=str(trace.get("template", "") or ""),
        constraint_type=str(trace.get("constraint_type", "") or ""),
        instantiated_slots={
            "query_spec": surface_query_spec,
            "semantic_slots": surface_semantic_slots,
            "constraints": sample.get("constraints"),
            "origin_plan": sample.get("origin_plan"),
            "origin_query_structured": sample.get("origin_query_structured"),
            "surface_query": final_query,
        },
        template_semantic_spec=(
            trace.get("template_semantic_spec")
            if isinstance(trace.get("template_semantic_spec"), dict)
            else {}
        ),
        meta={
            "template_id": str(trace.get("template_id", "") or ""),
            "query_generation_trace": trace,
        },
    )
    rebuilt_bundle = build_edit_targets_from_ir(surface_ir)
    rebuilt_hard = rebuilt_bundle["edit_target_constraints"]
    rebuilt_tags = rebuilt_bundle["edit_target_preference_tags"]
    rebuilt_preferences = rebuilt_bundle["edit_target_preferences"]
    if _freeze(expected_hard) != _freeze(rebuilt_hard):
        return False, "surface_hard_truth_mismatch"
    expected_tags = sample.get("edit_target_preference_tags") if isinstance(sample.get("edit_target_preference_tags"), list) else []
    if _freeze(expected_tags) != _freeze(rebuilt_tags):
        return False, "surface_soft_tag_mismatch"

    expected_preferences = sample.get("edit_target_preferences") if isinstance(sample.get("edit_target_preferences"), list) else []
    if _freeze(expected_preferences) != _freeze(rebuilt_preferences):
        return False, "surface_soft_preference_mismatch"

    return True, ""


def rewrite_surface_for_sample(sample: Dict[str, Any], llm_client) -> Dict[str, Any]:
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
    template = {
        "template": str(trace.get("template", "") or ""),
        "soft_intensity": str(trace.get("soft_target_intensity", "") or ""),
    }
    params = trace.get("rendered_params") if isinstance(trace.get("rendered_params"), dict) else {}
    constraint_type = str(trace.get("constraint_type", "") or "")
    canonical_query = sample.get("canonical_edit_query", "")
    evidence = step2.build_origin_plan_evidence(sample)
    fallback_params = dict(params)
    fallback_params["_origin_plan"] = sample.get("origin_plan", {})
    canonical_proof = step2.validate_conflict_trigger(
        constraints_data=sample.get("constraints") or {},
        evidence=evidence,
        constraint_type=constraint_type,
        fallback_params=fallback_params,
        edit_query=canonical_query,
    )
    if canonical_proof is None or not canonical_proof.ok:
        return {
            "success": False,
            "reason_code": canonical_proof.reason_code if canonical_proof is not None else "canonical_proof_missing",
            "reason": canonical_proof.reason if canonical_proof is not None else "canonical proof missing",
        }

    surface_selection = step2.select_query_surface(
        llm_client=llm_client,
        sample_id=sample["sample_id"],
        canonical_query=canonical_query,
        constraint_type=constraint_type,
        template=template,
        target_city=str(sample.get("meta", {}).get("target_city", "") or ""),
        pois=[],
        params=params,
        evidence_trace=trace.get("evidence", {}) if isinstance(trace.get("evidence"), dict) else {},
        soft_target_str=(
            f"{trace.get('soft_target_family', '')}:{trace.get('soft_target_facet', '')}:{trace.get('soft_target_mode', '')}"
            if trace.get("soft_target_family") and trace.get("soft_target_facet") and trace.get("soft_target_mode")
            else "(none)"
        ),
        origin_query_text=sample.get("origin_query_text", ""),
        constraints_data=sample.get("constraints") or {},
        fallback_params=fallback_params,
        evidence=evidence,
        canonical_query_spec=trace.get("canonical_query_spec") if isinstance(trace.get("canonical_query_spec"), dict) else {},
        canonical_semantic_slots=trace.get("canonical_semantic_slots") if isinstance(trace.get("canonical_semantic_slots"), dict) else {},
        canonical_proof=canonical_proof,
        target_family=str(trace.get("soft_target_family", "") or ""),
        target_facet=str(trace.get("soft_target_facet", "") or ""),
        target_mode=str(trace.get("soft_target_mode", "") or ""),
        origin_profile=(sample.get("meta", {}) or {}).get("origin_soft_profile", []),
    )

    final_query = surface_selection["edit_query"]
    final_proof = surface_selection["surface_proof"]
    surface_query_spec = surface_selection.get("surface_query_spec") if isinstance(surface_selection.get("surface_query_spec"), dict) else {}
    surface_semantic_slots = surface_selection.get("surface_semantic_slots") if isinstance(surface_selection.get("surface_semantic_slots"), dict) else {}

    present_ok, present_reason = _validate_frozen_truth_presence(sample, final_query, surface_query_spec, surface_semantic_slots)
    if not present_ok:
        return {
            "success": False,
            "reason_code": present_reason,
            "reason": present_reason,
        }

    step2.register_query_surface(final_query)
    return {
        "success": True,
        "edit_query": final_query,
        "surface_query_spec": surface_query_spec,
        "surface_semantic_slots": surface_semantic_slots,
        "surface_semantic_parse_source": surface_selection.get("surface_semantic_parse_source", ""),
        "surface_quality_pass": bool(surface_selection.get("surface_quality_pass", False)),
        "surface_reject_reason": str(surface_selection.get("surface_reject_reason", "") or ""),
        "query_surface_source": str(surface_selection.get("query_surface_source", "") or ""),
        "surface_proof": final_proof,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite canonical query surfaces while preserving frozen truth")
    parser.add_argument("--input", required=True, help="Batch directory from step 3")
    parser.add_argument("--model", default="auto", choices=["auto", "siliconcloud", "dmxapi"], help="LLM model to use")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Rewrite Query Surface")
    print("=" * 60)

    llm_client = load_client_from_env(provider=args.model)
    samples = load_samples(args.input)
    pending = filter_pending_samples(samples)
    if not pending:
        print("No samples need surface rewrite. Exiting.")
        return

    success_count = 0
    fail_count = 0
    for index, sample in enumerate(pending, start=1):
        print(f"\n  [{index}/{len(pending)}] Processing {sample['sample_id']}...")
        try:
            result = rewrite_surface_for_sample(sample, llm_client)
            if not result.get("success"):
                raise ValueError(result.get("reason", "surface rewrite failed"))
            sample["edit_query"] = result["edit_query"]
            trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
            trace["surface_query_spec"] = result["surface_query_spec"]
            trace["surface_semantic_slots"] = result["surface_semantic_slots"]
            trace["surface_semantic_parse_source"] = result["surface_semantic_parse_source"]
            trace["query_surface_source"] = result["query_surface_source"]
            trace["surface_quality_pass"] = result["surface_quality_pass"]
            trace["surface_reject_reason"] = result["surface_reject_reason"]
            proof = result["surface_proof"]
            trace["surface_gate_pass"] = bool(proof.ok)
            trace["surface_gate_reason"] = str(proof.reason)
            trace["surface_proof_ok"] = bool(proof.ok)
            trace["surface_proof_reason_code"] = str(proof.reason_code)
            trace["surface_proof_evidence"] = proof.evidence_used
            sample["query_generation_trace"] = trace
            sample["meta"]["status"] = "04_completed"
            sample["meta"]["last_step"] = "04_rewrite_query_surface"
            sample["meta"]["failure_reason"] = ""
            sample["meta"]["failure_reason_code"] = ""
            save_sample(sample, args.input)
            success_count += 1
            print("    ✓ surface rewrite completed")
        except Exception as exc:
            sample.pop("edit_query", None)
            sample["meta"]["status"] = "04_failed"
            sample["meta"]["last_step"] = "04_rewrite_query_surface"
            sample["meta"]["failure_reason"] = str(exc)
            sample["meta"]["failure_reason_code"] = "surface_rewrite_failed"
            save_sample(sample, args.input)
            fail_count += 1
            print(f"    ✗ {exc}")

    print("\n" + "=" * 60)
    print("✓ Step 4 Complete!")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


if __name__ == "__main__":
    main()
