#!/usr/bin/env python3
"""
Step 5: Analyze Conflict
使用 LLM 分析 origin_plan 和 edit_query 之间的 conflicts

这是数据生成pipeline的第五步，提取 conflict_facts 和 conflict_labels。
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(DATA_GEN_DIR, "utils"))

from llm_client import load_client_from_env
from llm_debug import create_llm_debug_logger
from conflict_resolver import resolve_primary_conflict


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    """加载batch中的所有samples"""
    batch_path = Path(batch_dir)
    samples = []

    for sample_file in batch_path.glob("sample_*.json"):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
                samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load {sample_file}: {e}")

    print(f"Loaded {len(samples)} samples from {batch_dir}")
    return samples


def filter_pending_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤出需要处理的samples（要求 Step 4 已完成 surface rewrite）。"""
    pending = []
    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        if status == "04_completed":
            pending.append(sample)

    print(f"Found {len(pending)} samples needing conflict analysis (from {len(samples)} total)")
    return pending


def format_origin_plan(origin_plan: Dict[str, Any]) -> str:
    """格式化origin plan为可读字符串"""
    lines = []
    lines.append(f"出发城市: {origin_plan.get('start_city', 'N/A')}")
    lines.append(f"目标城市: {origin_plan.get('target_city', 'N/A')}")
    lines.append(f"人数: {origin_plan.get('people_number', 'N/A')}")

    itinerary = origin_plan.get("itinerary", [])
    lines.append(f"\n行程 ({len(itinerary)}天):")

    for day_plan in itinerary:
        day = day_plan.get("day", "?")
        lines.append(f"\n  Day {day}:")
        activities = day_plan.get("activities", [])

        for i, activity in enumerate(activities):
            start_time = activity.get("start_time", "")
            end_time = activity.get("end_time", "")
            act_type = activity.get("type", "")
            position = activity.get("position", activity.get("end", ""))
            start = activity.get("start", "")

            if act_type == "train":
                lines.append(f"    [{i}] {start_time}-{end_time} 火车: {start} -> {position}")
            elif act_type == "airplane":
                lines.append(f"    [{i}] {start_time}-{end_time} 飞机: {start} -> {position}")
            else:
                lines.append(f"    [{i}] {start_time}-{end_time} {act_type}: {position}")

    return "\n".join(lines)


def format_constraints(constraints: Dict[str, Any]) -> str:
    """格式化constraints为可读字符串"""
    lines = []

    # must_include
    must_include = constraints.get("must_include", [])
    if must_include:
        lines.append("\n必选POI:")
        for poi in must_include:
            if isinstance(poi, dict):
                name = poi.get("name", "")
                city = poi.get("city", "")
                poi_type = poi.get("type", "")
                lines.append(f"  - {name} ({poi_type}, {city})")

    # spatial
    spatial = constraints.get("spatial", {})
    if spatial:
        lines.append(f"\n空间约束:")
        for key, value in spatial.items():
            lines.append(f"  - {key}: {value}")

    # temporal
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        temporal = [temporal]
    if temporal:
        lines.append(f"\n时间约束:")
        for temp in temporal:
            if isinstance(temp, dict):
                day = temp.get("day")
                must_visit = temp.get("must_visit")
                if day is not None and must_visit:
                    if isinstance(must_visit, list):
                        visit_text = ", ".join(str(v) for v in must_visit if isinstance(v, str))
                    else:
                        visit_text = str(must_visit)
                    lines.append(f"  - day {day} must_visit: {visit_text}")
                for key, value in temp.items():
                    if key == "must_visit":
                        continue
                    lines.append(f"  - {key}: {value}")
            else:
                lines.append(f"  - {temp}")

    # resource
    resource = constraints.get("resource", {})
    if resource:
        lines.append(f"\n资源约束:")
        for key, value in resource.items():
            lines.append(f"  - {key}: {value}")

    forbidden = constraints.get("forbidden", [])
    if isinstance(forbidden, list) and forbidden:
        lines.append("\n禁止POI:")
        for poi_name in forbidden:
            if isinstance(poi_name, str) and poi_name:
                lines.append(f"  - {poi_name}")

    return "\n".join(lines) if lines else "(无约束)"


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    解析LLM返回的JSON

    Args:
        response_text: LLM返回的文本

    Returns:
        解析后的字典
    """
    # 尝试提取JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 如果没有code block，尝试直接解析
        json_str = response_text.strip()

    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"    Error parsing LLM response: {e}")
        print(f"    Response text: {response_text[:500]}...")
        # 返回默认结构
        return {
            "conflict_facts": [],
            "conflict_labels": []
        }


def _resolve_activity_from_citation(origin_plan: Dict[str, Any], citation: Any) -> Optional[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(citation, str):
        return None
    match = re.fullmatch(r"\s*itinerary\[(\d+)\]\.activities\[(\d+)\]\s*", citation)
    if not match:
        return None

    day_idx = int(match.group(1))
    act_idx = int(match.group(2))
    itinerary = origin_plan.get("itinerary", [])
    if not isinstance(itinerary, list) or day_idx < 0 or day_idx >= len(itinerary):
        return None

    day_plan = itinerary[day_idx]
    activities = day_plan.get("activities", [])
    if not isinstance(activities, list) or act_idx < 0 or act_idx >= len(activities):
        return None

    return day_idx, act_idx, day_plan, activities[act_idx]


def _is_fact_grounded_by_activity(fact_text: Any, day_plan: Dict[str, Any], activity: Dict[str, Any]) -> bool:
    if not isinstance(fact_text, str) or not fact_text.strip():
        return False
    fact = fact_text.strip()

    day_value = day_plan.get("day")
    if isinstance(day_value, int):
        if f"第{day_value}天" in fact or f"Day {day_value}" in fact:
            return True

    candidate_tokens = []
    for field in ["position", "start", "end", "type", "start_time", "end_time", "TrainID", "FlightID"]:
        value = activity.get(field)
        if isinstance(value, str) and value.strip():
            candidate_tokens.append(value.strip())

    transports = activity.get("transports", [])
    if isinstance(transports, list):
        for seg in transports:
            if not isinstance(seg, dict):
                continue
            for field in ["start", "end", "mode", "start_time", "end_time"]:
                value = seg.get(field)
                if isinstance(value, str) and value.strip():
                    candidate_tokens.append(value.strip())

    for token in candidate_tokens:
        if token and token in fact:
            return True

    return False


def _extract_activity_ticket_price(activity: Dict[str, Any]) -> Optional[float]:
    price = activity.get("price")
    if isinstance(price, (int, float)):
        return float(price)

    cost = activity.get("cost")
    tickets = activity.get("tickets")
    if isinstance(cost, (int, float)) and isinstance(tickets, int) and tickets > 0:
        return float(cost) / tickets
    return None


def _extract_fact_money_comparison(fact_text: str) -> Optional[Tuple[str, float]]:
    patterns = [
        (r"(?:不超过|不高于|至多|最多|低于|小于|<=|≤)\s*(\d+(?:\.\d+)?)\s*元", "le"),
        (r"(?:不少于|不低于|至少|>=|≥)\s*(\d+(?:\.\d+)?)\s*元", "ge"),
        (r"(?:超过|高于|大于|超出|>|＞)\s*(\d+(?:\.\d+)?)\s*元", "gt"),
        (r"(?:等于|为|是)\s*(\d+(?:\.\d+)?)\s*元", "eq"),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, fact_text)
        if match:
            return operator, float(match.group(1))
    return None


def _fact_matches_activity_numeric_state(fact_text: Any, activity: Dict[str, Any]) -> bool:
    if not isinstance(fact_text, str) or not fact_text.strip():
        return True

    comparison = _extract_fact_money_comparison(fact_text)
    if comparison is None:
        return True

    fact = fact_text.strip()
    is_ticket_price_fact = any(token in fact for token in ["门票", "票价", "门票价格", "价格"])
    is_cost_fact = any(token in fact for token in ["总花费", "总费用", "费用", "成本"])

    value: Optional[float] = None
    if is_ticket_price_fact:
        value = _extract_activity_ticket_price(activity)
    elif is_cost_fact:
        cost = activity.get("cost")
        if isinstance(cost, (int, float)):
            value = float(cost)

    if value is None:
        return True

    operator, threshold = comparison
    if operator == "gt":
        return value > threshold
    if operator == "ge":
        return value >= threshold
    if operator == "le":
        return value <= threshold
    if operator == "eq":
        return abs(value - threshold) < 1e-6
    return True


def filter_grounded_conflict_facts(origin_plan: Dict[str, Any], conflict_facts: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not isinstance(conflict_facts, list):
        return [], ["conflict_facts is not a list"]

    kept: List[Dict[str, Any]] = []
    dropped_reasons: List[str] = []

    for idx, fact_item in enumerate(conflict_facts):
        if not isinstance(fact_item, dict):
            dropped_reasons.append(f"fact[{idx}] is not an object")
            continue

        citation = fact_item.get("citation")
        resolved = _resolve_activity_from_citation(origin_plan, citation)
        if not resolved:
            dropped_reasons.append(f"fact[{idx}] invalid citation: {citation}")
            continue

        _, _, day_plan, activity = resolved
        fact_text = fact_item.get("fact")
        if not _is_fact_grounded_by_activity(fact_text, day_plan, activity):
            dropped_reasons.append(f"fact[{idx}] not grounded by cited activity")
            continue
        if not _fact_matches_activity_numeric_state(fact_text, activity):
            dropped_reasons.append(f"fact[{idx}] numeric mismatch with cited activity")
            continue

        kept.append(fact_item)

    return kept, dropped_reasons


def analyze_conflicts_for_sample(sample: Dict[str, Any], llm_client,
                                 purity_threshold: float = 0.7) -> Tuple[bool, Dict[str, Any]]:
    """
    分析sample中的conflicts

    Returns:
        (success, result) 元组
    """
    origin_plan = sample["origin_plan"]
    edit_query = sample["edit_query"]
    constraints = sample["constraints"]
    target_bucket = sample["meta"]["target_bucket"]
    constraint_type = str(
        sample.get("query_generation_trace", {}).get("constraint_type", "")
        or sample.get("constraint_type", "")
    ).strip()

    # 确保target_bucket是tuple格式
    if isinstance(target_bucket, list):
        target_bucket = tuple(target_bucket)

    # 构建 LLM prompt
    prompt = f"""你是一个旅行计划冲突分析专家。分析原始计划和编辑需求之间的冲突。

原始计划：
{format_origin_plan(origin_plan)}

编辑需求（自然语言）：
{edit_query}

约束条件：
{format_constraints(constraints)}

目标冲突类型（用于验证）：{target_bucket}

冲突分类体系（请严格按定义判断）：
- Scope:
  - parameter: 仅需调整已有活动参数（时长、起止时间、交通方式等），不增删活动，不跨天改造
  - structural: 需要日内结构变更（插入/删除/替换/重排）
  - compositional: 需要跨天组合级改造（增减天数、跨天拆分重构）
- Dimension:
  - temporal: 时间窗/时长/结束时间/日内时间资源
  - spatial: 距离/区域/可达性/空间邻近
  - resource: 预算/费用/票量等可量化资源
  - semantic: 类型/主题/风格偏好
  - structural: 天数、每天景点数量等宏观结构指标
  - sequence: 同日组合关系与顺序约束（例如先A后B）
- Violation:
  - overflow: 量化阈值超限（时间、距离、预算等）
  - overlap: 编辑需求新增的活动或时间窗与原计划时间占用冲突（可用时间不足）
  - discontinuity: 软一致性破坏（节奏/主题/风格）
  - infeasible / incompatibility: 可记录，但通常不作为编辑任务池主标签

任务：
1. 提取具体的冲突事实（conflict_facts）
2. 为每个事实定位到 plan 中的具体位置（使用 itinerary[day].activities[index] 格式）
3. 将每个冲突映射到 (scope, dimension, violation) 三元组

返回JSON格式：
{{
  "conflict_facts": [
    {{
      "fact": "<与原始行程可核对的冲突事实>",
      "citation": "itinerary[day_index].activities[activity_index]",
      "constraint_violated": "<被违反的约束>"
    }}
  ],
  "conflict_labels": [
    ["scope", "dimension", "violation"]
  ]
}}

注意：
- conflict_facts 中的 citation 必须引用具体的 itinerary 索引
- conflict_labels 是列表的列表，每个子列表包含 [scope, dimension, violation]
"""

    # 调用 LLM
    messages = [
        {"role": "system", "content": "You are a travel planning conflict analysis expert. Always return valid JSON."},
        {"role": "user", "content": prompt}
    ]

    debug_logger = create_llm_debug_logger("04", sample["sample_id"], "analyze_conflicts")
    response = llm_client.call_with_retry(
        messages,
        max_retries=2,
        debug_logger=debug_logger,
        debug_context={
            "target_bucket": target_bucket
        }
    )

    if response is None:
        return False, {"error": "LLM call failed"}

    # 解析响应
    result = parse_llm_response(response.content)

    # 提取冲突标签和事实
    conflict_labels = result.get("conflict_labels", [])
    conflict_facts_raw = result.get("conflict_facts", [])
    conflict_facts, dropped_fact_reasons = filter_grounded_conflict_facts(origin_plan, conflict_facts_raw)
    if dropped_fact_reasons:
        print(f"    → Dropped {len(dropped_fact_reasons)} ungrounded conflict facts")

    # Rule-based 主冲突唯一化（替代 LLM 二次判定）
    print("    → Running rule-based conflict resolver...")
    resolution = resolve_primary_conflict(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        conflict_labels_llm=conflict_labels,
        target_bucket=target_bucket,
        purity_threshold=purity_threshold,
        constraint_type=constraint_type,
    )

    target_confirmed = bool(resolution.get("target_confirmed", False))
    match_type_rule = str(resolution.get("match_type_rule", "mismatch"))
    purity_score_rule = float(resolution.get("purity_score_rule", 0.0))
    primary_conflict = resolution.get("primary_conflict")
    target_confidence = purity_score_rule
    target_explanation = f"rule_based(match_type={match_type_rule}, purity={purity_score_rule:.2f})"

    print(
        f"    → Rule resolver: confirmed={target_confirmed}, "
        f"match_type={match_type_rule}, purity={purity_score_rule:.2f}"
    )

    analysis_result = {
        "conflict_facts": conflict_facts,
        "conflict_facts_grounding": {
            "kept": len(conflict_facts),
            "dropped": len(dropped_fact_reasons),
            "drop_reasons": dropped_fact_reasons[:10],
        },
        "conflict_labels": conflict_labels,  # 保持原始格式（列表）
        "conflict_set": resolution.get("conflict_set", conflict_labels),
        "primary_conflict": primary_conflict,
        "secondary_conflicts": resolution.get("secondary_conflicts", []),
        "purity_score_rule": purity_score_rule,
        "match_type_rule": match_type_rule,
        "resolver_trace": resolution.get("trace", {}),
        "conflict_trace": resolution.get("trace", {}),
        "target_confirmed": target_confirmed,
        "target_confidence": target_confidence,
        "target_explanation": target_explanation,
        "match_type": match_type_rule,  # 兼容旧字段
        "purity_score": purity_score_rule,  # 兼容旧字段
        "primary_label": primary_conflict,  # 兼容旧字段
    }

    return True, analysis_result


def save_sample(sample: Dict[str, Any], output_dir: str):
    """保存sample到文件"""
    sample_file = Path(output_dir) / f"{sample['sample_id']}.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Analyze conflicts between origin plan and edit query")
    parser.add_argument("--input", required=True, help="Batch directory from step 3")
    parser.add_argument("--model", default="auto", choices=["auto", "siliconcloud", "dmxapi"],
                       help="LLM model to use (default: auto)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Max retries per sample (default: 3)")
    parser.add_argument("--purity-threshold", type=float, default=0.7,
                       help="Strong-match purity threshold in [0,1] (default: 0.7)")

    args = parser.parse_args()
    args.purity_threshold = max(0.0, min(1.0, args.purity_threshold))

    print("=" * 60)
    print("Step 4: Analyze Conflict")
    print("=" * 60)

    # 1. 初始化 LLM 客户端
    print("\n1. Initializing LLM client...")
    try:
        llm_client = load_client_from_env(provider=args.model)
        print("  ✓ LLM client initialized")
    except Exception as e:
        print(f"  ✗ Error initializing: {e}")
        sys.exit(1)

    # 2. 加载 samples
    print(f"\n2. Loading samples from: {args.input}")
    samples = load_samples(args.input)
    pending = filter_pending_samples(samples)

    if not pending:
        print("No samples need conflict analysis. Exiting.")
        return

    # 3. 分析 conflicts
    print(f"\n3. Analyzing conflicts for {len(pending)} samples...")
    success_count = 0
    fail_count = 0
    target_confirmed_count = 0

    for i, sample in enumerate(pending):
        try:
            print(f"\n  [{i+1}/{len(pending)}] Processing {sample['sample_id']}...")

            # 分析 conflicts
            success, result = analyze_conflicts_for_sample(
                sample,
                llm_client,
                purity_threshold=args.purity_threshold
            )

            if success:
                # 保存结果
                sample["conflict_facts"] = result.get("conflict_facts", [])
                sample["conflict_labels"] = result.get("conflict_labels", [])
                sample["conflict_set"] = result.get("conflict_set", result.get("conflict_labels", []))
                sample["primary_conflict"] = result.get("primary_conflict")
                sample["secondary_conflicts"] = result.get("secondary_conflicts", [])
                sample["purity_score_rule"] = result.get("purity_score_rule", 0.0)
                sample["match_type_rule"] = result.get("match_type_rule", "mismatch")
                sample["resolver_trace"] = result.get("resolver_trace", {})
                sample["meta"]["status"] = "05_completed"
                sample["meta"]["last_step"] = "05_analyze_conflict"
                sample["meta"]["retry_count"] = 0
                sample["meta"]["target_confirmed"] = result.get("target_confirmed", False)
                sample["meta"]["target_confidence"] = result.get("target_confidence", 0.0)
                sample["meta"]["target_explanation"] = result.get("target_explanation", "")
                sample["meta"]["match_type"] = result.get("match_type_rule", "mismatch")
                sample["meta"]["purity_score"] = result.get("purity_score_rule", 0.0)
                sample["meta"]["primary_label"] = result.get("primary_conflict")

                # 立即保存
                save_sample(sample, args.input)

                if result.get("target_confirmed"):
                    print(f"    ✓ Conflict analysis complete (target confirmed: {result['target_confirmed']}, confidence: {result.get('target_confidence', 0.0):.2f})")
                    target_confirmed_count += 1
                else:
                    print(f"    ⚠ Conflict analysis complete but target bucket NOT confirmed")
                    print(f"      Target: {sample['meta']['target_bucket']}")
                    print(f"      Conflict set: {result.get('conflict_set', [])}")
                    print(f"      Confidence: {result.get('target_confidence', 0.0):.2f}")
                    print(f"      Match type: {result.get('match_type_rule', 'mismatch')}")
                    print(f"      Purity score: {result.get('purity_score_rule', 0.0):.2f}")
                    print(f"      Primary label: {result.get('primary_conflict')}")
                    explanation = result.get("target_explanation", "")
                    if explanation:
                        print(f"      Explanation: {explanation[:100]}...")

                success_count += 1
            else:
                # 失败
                error_msg = result.get("error", "Unknown error")
                sample["meta"]["status"] = "04_failed"
                sample["meta"]["failure_reason"] = error_msg
                sample["meta"]["retry_count"] = sample["meta"].get("retry_count", 0) + 1
                save_sample(sample, args.input)

                print(f"    ✗ Failed: {error_msg}")
                fail_count += 1

        except Exception as e:
            print(f"    ✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

            sample["meta"]["status"] = "04_failed"
            sample["meta"]["failure_reason"] = f"Exception: {str(e)}"
            sample["meta"]["retry_count"] = sample["meta"].get("retry_count", 0) + 1
            save_sample(sample, args.input)
            fail_count += 1

    # 4. 总结
    print("\n" + "=" * 60)
    print("✓ Step 4 Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Target confirmed: {target_confirmed_count}/{success_count}")
    print(f"  Success rate: {success_count}/{len(pending)} ({100*success_count/len(pending):.1f}%)")
    print(f"\nOutput location: {args.input}/")
    print(f"\nNext step:")
    print(f"  python scripts/06_generate_strategy.py --input {args.input}")


if __name__ == "__main__":
    main()
