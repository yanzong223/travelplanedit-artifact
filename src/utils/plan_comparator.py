"""
Plan Comparison Utilities

计算和分析旅行计划修改前后的详细差异。
提供变量级别、约束级别和结构级别的对比分析。
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.models.enums import ConstraintType, VariableType
from core.models.factor_graph import FactorGraph
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VariableDiff:
    """变量差异记录"""

    variable_id: str
    diff_type: str  # "added", "removed", "modified", "unchanged"
    before_data: Optional[Dict[str, Any]] = None
    after_data: Optional[Dict[str, Any]] = None
    field_changes: Optional[Dict[str, Tuple[Any, Any]]] = None  # field -> (old, new)


@dataclass
class ConstraintDiff:
    """约束差异记录"""

    constraint_id: str
    diff_type: str  # "added", "removed", "modified", "unchanged"
    before_data: Optional[Dict[str, Any]] = None
    after_data: Optional[Dict[str, Any]] = None
    field_changes: Optional[Dict[str, Tuple[Any, Any]]] = None


@dataclass
class PlanDifference:
    """完整的计划差异分析结果"""

    plan_id: str
    comparison_timestamp: str
    variables_diff: List[VariableDiff]
    constraints_diff: List[ConstraintDiff]

    # 统计信息
    variables_added: int
    variables_removed: int
    variables_modified: int
    constraints_added: int
    constraints_removed: int
    constraints_modified: int

    # 影响分析
    affected_variable_ids: Set[str]
    affected_constraint_ids: Set[str]

    # 操作映射（如果可用）
    applied_operations: Optional[List[Dict[str, Any]]] = None


class PlanComparator:
    """旅行计划对比器"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def compare_plans(
        self,
        original_graph: FactorGraph,
        modified_graph: FactorGraph,
        plan_id: str,
        applied_operations: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanDifference:
        """
        对比两个计划图，生成详细差异报告

        Args:
            original_graph: 原始计划图
            modified_graph: 修改后计划图
            plan_id: 计划ID
            applied_operations: 应用的操作列表（可选）

        Returns:
            详细的差异分析结果
        """
        self.logger.info(f"开始对比计划 {plan_id}")

        # 计算变量差异
        variables_diff = self._compare_variables(
            original_graph.variables, modified_graph.variables
        )

        # 计算约束差异
        constraints_diff = self._compare_constraints(
            original_graph.constraints, modified_graph.constraints
        )

        # 统计变更
        var_stats = self._calculate_variable_stats(variables_diff)
        const_stats = self._calculate_constraint_stats(constraints_diff)

        # 收集受影响的ID
        affected_vars = set()
        affected_constraints = set()

        for var_diff in variables_diff:
            if var_diff.diff_type in ["added", "removed", "modified"]:
                affected_vars.add(var_diff.variable_id)

        for const_diff in constraints_diff:
            if const_diff.diff_type in ["added", "removed", "modified"]:
                affected_constraints.add(const_diff.constraint_id)

        # 创建差异对象
        plan_diff = PlanDifference(
            plan_id=plan_id,
            comparison_timestamp=datetime.now().isoformat(),
            variables_diff=variables_diff,
            constraints_diff=constraints_diff,
            variables_added=var_stats["added"],
            variables_removed=var_stats["removed"],
            variables_modified=var_stats["modified"],
            constraints_added=const_stats["added"],
            constraints_removed=const_stats["removed"],
            constraints_modified=const_stats["modified"],
            affected_variable_ids=affected_vars,
            affected_constraint_ids=affected_constraints,
            applied_operations=applied_operations,
        )

        self.logger.info(
            f"计划对比完成: {var_stats['modified']} 变量修改, " f"{const_stats['modified']} 约束修改"
        )

        return plan_diff

    def _compare_variables(
        self, original_vars: Dict[str, Any], modified_vars: Dict[str, Any]
    ) -> List[VariableDiff]:
        """对比变量字典"""
        diffs = []

        # 处理字典格式的变量
        if isinstance(original_vars, dict):
            orig_dict = original_vars
        else:
            # 兼容列表格式
            orig_dict = {var.id: var for var in original_vars}

        if isinstance(modified_vars, dict):
            mod_dict = modified_vars
        else:
            # 兼容列表格式
            mod_dict = {var.id: var for var in modified_vars}

        all_var_ids = set(orig_dict.keys()) | set(mod_dict.keys())

        for var_id in all_var_ids:
            orig_var = orig_dict.get(var_id)
            mod_var = mod_dict.get(var_id)

            if orig_var and mod_var:
                # 变量存在于两个版本中，检查是否修改
                field_changes = self._compare_variable_fields(orig_var, mod_var)
                if field_changes:
                    diffs.append(
                        VariableDiff(
                            variable_id=var_id,
                            diff_type="modified",
                            before_data=orig_var.model_dump(),
                            after_data=mod_var.model_dump(),
                            field_changes=field_changes,
                        )
                    )
                else:
                    diffs.append(
                        VariableDiff(
                            variable_id=var_id,
                            diff_type="unchanged",
                            before_data=orig_var.model_dump(),
                            after_data=mod_var.model_dump(),
                        )
                    )
            elif orig_var and not mod_var:
                # 变量被删除
                diffs.append(
                    VariableDiff(
                        variable_id=var_id,
                        diff_type="removed",
                        before_data=orig_var.model_dump(),
                    )
                )
            elif not orig_var and mod_var:
                # 变量被添加
                diffs.append(
                    VariableDiff(
                        variable_id=var_id,
                        diff_type="added",
                        after_data=mod_var.model_dump(),
                    )
                )

        return diffs

    def _compare_constraints(
        self, original_constraints: Dict[str, Any], modified_constraints: Dict[str, Any]
    ) -> List[ConstraintDiff]:
        """对比约束字典"""
        diffs = []

        # 处理字典格式的约束
        if isinstance(original_constraints, dict):
            orig_dict = original_constraints
        else:
            # 兼容列表格式
            orig_dict = {const.id: const for const in original_constraints}

        if isinstance(modified_constraints, dict):
            mod_dict = modified_constraints
        else:
            # 兼容列表格式
            mod_dict = {const.id: const for const in modified_constraints}

        all_const_ids = set(orig_dict.keys()) | set(mod_dict.keys())

        for const_id in all_const_ids:
            orig_const = orig_dict.get(const_id)
            mod_const = mod_dict.get(const_id)

            if orig_const and mod_const:
                # 约束存在于两个版本中，检查是否修改
                field_changes = self._compare_constraint_fields(orig_const, mod_const)
                if field_changes:
                    diffs.append(
                        ConstraintDiff(
                            constraint_id=const_id,
                            diff_type="modified",
                            before_data=orig_const.model_dump(),
                            after_data=mod_const.model_dump(),
                            field_changes=field_changes,
                        )
                    )
                else:
                    diffs.append(
                        ConstraintDiff(
                            constraint_id=const_id,
                            diff_type="unchanged",
                            before_data=orig_const.model_dump(),
                            after_data=mod_const.model_dump(),
                        )
                    )
            elif orig_const and not mod_const:
                # 约束被删除
                diffs.append(
                    ConstraintDiff(
                        constraint_id=const_id,
                        diff_type="removed",
                        before_data=orig_const.model_dump(),
                    )
                )
            elif not orig_const and mod_const:
                # 约束被添加
                diffs.append(
                    ConstraintDiff(
                        constraint_id=const_id,
                        diff_type="added",
                        after_data=mod_const.model_dump(),
                    )
                )

        return diffs

    def _compare_variable_fields(
        self, orig_var: Any, mod_var: Any
    ) -> Optional[Dict[str, Tuple[Any, Any]]]:
        """比较变量的字段变更"""
        changes = {}

        # 比较所有模型字段
        orig_data = orig_var.model_dump()
        mod_data = mod_var.model_dump()

        for field_name in set(orig_data.keys()) | set(mod_data.keys()):
            orig_value = orig_data.get(field_name)
            mod_value = mod_data.get(field_name)

            if orig_value != mod_value:
                changes[field_name] = (orig_value, mod_value)

        return changes if changes else None

    def _compare_constraint_fields(
        self, orig_const: Any, mod_const: Any
    ) -> Optional[Dict[str, Tuple[Any, Any]]]:
        """比较约束的字段变更"""
        changes = {}

        # 比较所有模型字段
        orig_data = orig_const.model_dump()
        mod_data = mod_const.model_dump()

        for field_name in set(orig_data.keys()) | set(mod_data.keys()):
            orig_value = orig_data.get(field_name)
            mod_value = mod_data.get(field_name)

            if orig_value != mod_value:
                changes[field_name] = (orig_value, mod_value)

        return changes if changes else None

    def _calculate_variable_stats(
        self, variables_diff: List[VariableDiff]
    ) -> Dict[str, int]:
        """计算变量变更统计"""
        stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}

        for var_diff in variables_diff:
            stats[var_diff.diff_type] += 1

        return stats

    def _calculate_constraint_stats(
        self, constraints_diff: List[ConstraintDiff]
    ) -> Dict[str, int]:
        """计算约束变更统计"""
        stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}

        for const_diff in constraints_diff:
            stats[const_diff.diff_type] += 1

        return stats

    def save_difference_report(
        self,
        plan_diff: PlanDifference,
        output_path: Path,
        include_unchanged: bool = False,
    ) -> None:
        """
        保存差异报告到文件

        Args:
            plan_diff: 计划差异对象
            output_path: 输出文件路径
            include_unchanged: 是否包含未变更的项目
        """
        # 准备保存数据
        save_data = {
            "plan_id": plan_diff.plan_id,
            "comparison_timestamp": plan_diff.comparison_timestamp,
            "summary": {
                "variables": {
                    "added": plan_diff.variables_added,
                    "removed": plan_diff.variables_removed,
                    "modified": plan_diff.variables_modified,
                },
                "constraints": {
                    "added": plan_diff.constraints_added,
                    "removed": plan_diff.constraints_removed,
                    "modified": plan_diff.constraints_modified,
                },
                "affected_variables": list(plan_diff.affected_variable_ids),
                "affected_constraints": list(plan_diff.affected_constraint_ids),
            },
            "applied_operations": plan_diff.applied_operations,
        }

        # 过滤变量差异
        variables_to_save = []
        for var_diff in plan_diff.variables_diff:
            if include_unchanged or var_diff.diff_type != "unchanged":
                variables_to_save.append(asdict(var_diff))
        save_data["variables_diff"] = variables_to_save

        # 过滤约束差异
        constraints_to_save = []
        for const_diff in plan_diff.constraints_diff:
            if include_unchanged or const_diff.diff_type != "unchanged":
                constraints_to_save.append(asdict(const_diff))
        save_data["constraints_diff"] = constraints_to_save

        # 保存到文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"差异报告已保存到: {output_path}")


def create_plan_comparator() -> PlanComparator:
    """创建计划对比器实例"""
    return PlanComparator()
