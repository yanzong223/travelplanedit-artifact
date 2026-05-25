#!/usr/bin/env python3
"""
Step 6: Final Validation
运行所有验证检查，生成 checks 字段

这是数据生成 pipeline 的第九步，验证数据质量。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(DATA_GEN_DIR, "utils"))

from poi_database import POIDatabase
from data_generation_validators import (
    CORE_PRIMARY_CHECKS,
    FULL_PRIMARY_CHECKS,
    OPTIONAL_EXECUTION_CHECKS,
    PRIMARY_CHECKS,
    REPORTED_NON_BLOCKING_CHECKS,
    validate_all,
)
from validation_runtime import (
    apply_validation_exception,
    apply_validation_outcome,
    build_validation_inputs,
    collect_check_messages,
    compute_batch_check_stats,
    update_batch_metadata,
)


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    """加载batch中的所有samples"""
    batch_path = Path(batch_dir)
    samples = []

    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
                samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load {sample_file}: {e}")

    print(f"Loaded {len(samples)} samples from {batch_dir}")
    return samples


def filter_pending_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤出需要处理的 samples（core 模式从 step5 开始，full 模式从 step8 开始）。"""
    pending = []
    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        if status in {"05_completed", "06_completed", "08_completed"}:
            pending.append(sample)

    print(f"Found {len(pending)} samples needing final validation (from {len(samples)} total)")
    return pending


def sample_requires_strategy_checks(sample: Dict[str, Any]) -> bool:
    status = sample.get("meta", {}).get("status", "")
    if status in {"06_completed", "07_completed", "08_completed"}:
        return True
    return bool(sample.get("strategy_plan"))


def validate_sample(
    sample: Dict[str, Any],
    poi_db: POIDatabase,
    *,
    require_solver_feasibility: bool = False,
    require_category_guard: bool = False,
    solver_valid_profile: str = "off",
) -> Dict[str, Any]:
    """
    运行所有验证检查

    Returns:
        checks 字典
    """
    return validate_all(
        **build_validation_inputs(sample),
        poi_db=poi_db,
        require_strategy_checks=sample_requires_strategy_checks(sample),
        require_solver_feasibility=require_solver_feasibility,
        require_category_guard=require_category_guard,
        solver_valid_profile=solver_valid_profile,
    )


def save_sample(sample: Dict[str, Any], output_dir: str):
    """保存sample到文件"""
    sample_file = Path(output_dir) / f"{sample['sample_id']}.json"
    tmp_file = sample_file.with_suffix(sample_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_file, sample_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run final validation checks")
    parser.add_argument("--input", required=True, help="Batch directory from step 5/8")
    parser.add_argument("--strict", action="store_true",
                       help="Strict mode: fail if any validation check fails")
    parser.add_argument(
        "--require-solver-feasibility",
        action="store_true",
        help="Include checks.solver_feasibility in all_pass for solver-valid benchmark builds",
    )
    parser.add_argument(
        "--require-category-guard",
        action="store_true",
        help="Include checks.category_guard_validity in all_pass",
    )
    parser.add_argument(
        "--solver-valid-profile",
        default="off",
        choices=["off", "normal", "intentional_infeasible"],
        help="Solver-valid validation profile. normal requires solver feasibility and category guards.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Step 6: Final Validation")
    print("=" * 60)

    # 1. 初始化 POI 数据库
    print("\n1. Initializing POI database...")
    try:
        poi_db = POIDatabase()
        print("  ✓ POI database initialized")
    except Exception as e:
        print(f"  ✗ Error initializing: {e}")
        sys.exit(1)

    # 2. 加载 samples
    print(f"\n2. Loading samples from: {args.input}")
    samples = load_samples(args.input)
    pending = filter_pending_samples(samples)

    if not pending:
        print("No samples need validation. Exiting.")
        return

    # 3. 运行验证
    print(f"\n3. Running validation checks for {len(pending)} samples...")
    success_count = 0
    fail_count = 0
    all_pass_count = 0

    for i, sample in enumerate(pending):
        try:
            print(f"\n  [{i+1}/{len(pending)}] Processing {sample['sample_id']}...")

            # 运行验证
            checks = validate_sample(
                sample,
                poi_db,
                require_solver_feasibility=args.require_solver_feasibility,
                require_category_guard=args.require_category_guard,
                solver_valid_profile=args.solver_valid_profile,
            )
            outcome = apply_validation_outcome(sample, checks, strict=args.strict)
            save_sample(sample, args.input)

            if outcome["all_pass"]:
                print(f"    ✓ All checks passed")
                all_pass_count += 1
                success_count += 1
                continue

            failed_checks = outcome["failed_checks"]
            print(f"    ⚠ Some checks failed: {', '.join(failed_checks)}")

            for check_name in failed_checks:
                messages = collect_check_messages(checks.get(check_name, {}))
                for message in messages[:2]:
                    print(f"        - {check_name}: {message}")

            success_count += 1

        except Exception as e:
            print(f"    ✗ Error during validation: {e}")
            import traceback
            traceback.print_exc()

            apply_validation_exception(sample, e)
            save_sample(sample, args.input)
            fail_count += 1

    # 4. 生成 batch 级别的统计
    print("\n4. Generating batch statistics...")
    batch_primary_checks = (
        FULL_PRIMARY_CHECKS
        if any(sample_requires_strategy_checks(sample) for sample in pending)
        else CORE_PRIMARY_CHECKS
    )
    if args.require_solver_feasibility and "solver_feasibility" not in batch_primary_checks:
        batch_primary_checks = list(batch_primary_checks) + ["solver_feasibility"]
    if args.solver_valid_profile == "normal" and "solver_feasibility" not in batch_primary_checks:
        batch_primary_checks = list(batch_primary_checks) + ["solver_feasibility"]
    if args.solver_valid_profile == "normal" and "temporal_window_feasibility" not in batch_primary_checks:
        batch_primary_checks = list(batch_primary_checks) + ["temporal_window_feasibility"]
    if (
        (args.require_category_guard or args.solver_valid_profile in {"normal", "intentional_infeasible"})
        and "category_guard_validity" not in batch_primary_checks
    ):
        batch_primary_checks = list(batch_primary_checks) + ["category_guard_validity"]

    batch_stats = compute_batch_check_stats(
        pending,
        batch_primary_checks,
        OPTIONAL_EXECUTION_CHECKS,
        REPORTED_NON_BLOCKING_CHECKS,
    )
    batch_checks = batch_stats["primary"]
    optional_checks = batch_stats["optional"]
    reported_checks = batch_stats["reported_non_blocking"]
    total = batch_stats["total"]

    print(f"\nBatch Validation Statistics (n={total}):")
    for check_name, count in batch_checks.items():
        rate = 100 * count / total if total > 0 else 0
        print(f"  {check_name}: {count}/{total} ({rate:.1f}%)")
    if any(item["applicable"] > 0 for item in optional_checks.values()):
        print("\nOptional Execution Checks:")
        for check_name, item in optional_checks.items():
            applicable = item["applicable"]
            if applicable == 0:
                continue
            rate = 100 * item["passed"] / applicable
            print(f"  {check_name}: {item['passed']}/{applicable} ({rate:.1f}%)")
    if any(item["applicable"] > 0 for item in reported_checks.values()):
        print("\nReported Non-blocking Checks:")
        for check_name, item in reported_checks.items():
            applicable = item["applicable"]
            if applicable == 0:
                continue
            rate = 100 * item["passed"] / applicable
            print(f"  {check_name}: {item['passed']}/{applicable} ({rate:.1f}%)")

    # 5. 总结
    print("\n" + "=" * 60)
    print("✓ Step 6 Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Processed: {success_count + fail_count}")
    print(f"  All passed: {all_pass_count}/{success_count} ({100*all_pass_count/success_count if success_count > 0 else 0:.1f}%)")
    print(f"  Failed: {fail_count}")
    print(f"  Success rate: {success_count}/{len(pending)} ({100*success_count/len(pending):.1f}%)")
    print(f"\nOutput location: {args.input}/")

    # 生成 batch metadata 更新
    metadata_file = update_batch_metadata(
        args.input,
        pending,
        batch_primary_checks,
        OPTIONAL_EXECUTION_CHECKS,
        REPORTED_NON_BLOCKING_CHECKS,
        timestamp=get_timestamp(),
    )
    print(f"\nUpdated batch metadata at {metadata_file}")


def get_timestamp() -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().isoformat()


if __name__ == "__main__":
    main()
