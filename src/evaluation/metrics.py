"""
Evaluation Metrics for TPE System

Based on experiment_design.md, implements:
1. Constraint Satisfaction Rate (CSR)
2. Efficiency Metrics
3. Success Rate Metrics
4. Plan Consistency Metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json


@dataclass
class ConstraintDefinition:
    """
    统一的约束定义结构，支持 base/edit 约束区分和冲突标记
    """
    id: str
    source: str  # "base" 或 "edit"
    type: str  # 如 "budget_limit", "must_visit_reorder", "days_limit", ...
    params: Dict[str, Any] = field(default_factory=dict)
    dsl: Optional[str] = None
    relation: Optional[str] = None  # 如 "override" / None
    overridden_by: Optional[str] = None  # 若被某条 edit 约束覆盖，填对方 id
    description: Optional[str] = None  # 可读描述（用于日志/调试）


@dataclass
class SingleConstraintEval:
    """
    单条约束的评估结果
    """
    id: str
    source: str
    type: str
    satisfied: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintSatisfactionMetrics:
    """
    约束满足率（Constraint Satisfaction Rate, CSR）
    
    - DR (Delivery Rate): Plan 输出是否可执行（结构合法）
    - EPR (Environmental Pass Rate): 外部事实匹配（开放时间、交通可达）
    - LPR (Logical Pass Rate): 时间不重叠、顺序正确、路线连贯
    - FPR (Final Pass Rate): 综合以上（全部满足的比例）
    
    FPR = DR * EPR * LPR
    """
    
    # 全局约束满足率
    total_constraints: int = 0
    satisfied_constraints: int = 0
    
    # 分项指标
    delivery_rate: float = 0.0  # DR: 结构合法性
    environmental_pass_rate: float = 0.0  # EPR: 外部事实匹配
    logical_pass_rate: float = 0.0  # LPR: 逻辑一致性
    final_pass_rate: float = 0.0  # FPR: 综合通过率
    
    # 详细约束违反信息
    violated_constraints: List[Dict[str, Any]] = field(default_factory=list)
    constraint_details: Dict[str, Any] = field(default_factory=dict)

    # Phase 2 新增字段：per-constraint 细粒度结果
    base_csr: Optional[float] = None  # base constraints 满足率
    edit_csr: Optional[float] = None  # edit constraints 满足率
    conflict_break_rate: Optional[float] = None  # 被 override 的 base 约束中，有多少被破坏
    constraint_results: List[SingleConstraintEval] = field(default_factory=list)  # 每条约束的详细结果
    
    def calculate_csr(self) -> float:
        """计算全局约束满足率"""
        if self.total_constraints == 0:
            return 0.0
        return self.satisfied_constraints / self.total_constraints
    
    def calculate_fpr(self) -> float:
        """计算综合通过率 FPR = DR * EPR * LPR"""
        self.final_pass_rate = (
            self.delivery_rate * 
            self.environmental_pass_rate * 
            self.logical_pass_rate
        )
        return self.final_pass_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "csr": self.calculate_csr(),
            "dr": self.delivery_rate,
            "epr": self.environmental_pass_rate,
            "lpr": self.logical_pass_rate,
            "fpr": self.calculate_fpr(),
            "total_constraints": self.total_constraints,
            "satisfied_constraints": self.satisfied_constraints,
            "violated_constraints": self.violated_constraints,
            "constraint_details": self.constraint_details,
            # Phase 2 新增字段
            "base_csr": self.base_csr,
            "edit_csr": self.edit_csr,
            "conflict_break_rate": self.conflict_break_rate,
            "constraints": [
                {
                    "id": r.id,
                    "source": r.source,
                    "type": r.type,
                    "satisfied": r.satisfied,
                    "details": r.details,
                }
                for r in self.constraint_results
            ]
        }


@dataclass
class EfficiencyMetrics:
    """
    效率指标（Efficiency）
    
    比较三类方法:
    - full replan（不做 scope，只做全局重规划）
    - oracle scope → 局部重规划（消融）
    - TPE heuristic scope（我们的）
    """
    
    # 生成时延
    total_duration_seconds: float = 0.0
    avg_latency_seconds: float = 0.0
    
    # Token 使用
    total_prompt_tokens: int = 0
    total_generation_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_episode: float = 0.0
    
    # Workflow 步骤
    total_workflow_steps: int = 0
    avg_workflow_steps: float = 0.0
    
    # 方法类型标记（用于对比实验）
    method_type: Optional[str] = None  # 'full_replan', 'oracle_scope', 'tpe_heuristic'
    
    def calculate_averages(self, num_episodes: int):
        """计算平均指标"""
        if num_episodes > 0:
            self.avg_latency_seconds = self.total_duration_seconds / num_episodes
            self.avg_tokens_per_episode = self.total_tokens / num_episodes
            self.avg_workflow_steps = self.total_workflow_steps / num_episodes
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "avg_latency_seconds": self.avg_latency_seconds,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "avg_tokens_per_episode": self.avg_tokens_per_episode,
            "total_workflow_steps": self.total_workflow_steps,
            "avg_workflow_steps": self.avg_workflow_steps,
            "method_type": self.method_type
        }


@dataclass
class SuccessRateMetrics:
    """
    成功率指标

    - ESR (Edit Success Rate): 编辑成功率 - 是否满足所有hard constraint

    注意：URSR (User Request Satisfaction Rate) 已停用，因为缺少明确的判断逻辑
    """

    total_episodes: int = 0
    successful_edits: int = 0

    # 失败原因统计
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def calculate_esr(self) -> float:
        """计算编辑成功率 ESR"""
        if self.total_episodes == 0:
            return 0.0
        return self.successful_edits / self.total_episodes

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "esr": self.calculate_esr(),
            "total_episodes": self.total_episodes,
            "successful_edits": self.successful_edits,
            "failure_reasons": self.failure_reasons
        }


@dataclass
class ConsistencyMetrics:
    """
    计划一致性指标（Plan Consistency）
    
    - CR (Content Retention): 内容保留率
    - MED (Minimal Edit Distance): 最小修改距离
    - STC (Spatio-temporal Coherence): 计划时空一致性（可选）
    """
    
    # 内容保留率
    original_variables_count: int = 0
    retained_variables_count: int = 0
    content_retention: float = 0.0
    
    # 最小修改距离
    edit_distance: int = 0
    poi_sequence_similarity: float = 0.0
    
    # 时空一致性（可选）
    spatio_temporal_coherence: Optional[float] = None
    adjacent_poi_valid: int = 0
    adjacent_poi_total: int = 0
    
    # 详细修改信息
    modified_variables: List[str] = field(default_factory=list)
    added_variables: List[str] = field(default_factory=list)
    removed_variables: List[str] = field(default_factory=list)
    
    def calculate_cr(self) -> float:
        """
        计算内容保留率 CR
        CR = |PLAN_origin ∩ PLAN_edited| / |PLAN_origin|
        """
        if self.original_variables_count == 0:
            return 0.0
        self.content_retention = self.retained_variables_count / self.original_variables_count
        return self.content_retention
    
    def calculate_stc(self) -> Optional[float]:
        """
        计算时空一致性 STC
        STC = (1/|E|) * Σ I{Time(i) + Travel(i,j) <= Time(j)}
        """
        if self.adjacent_poi_total == 0:
            return None
        self.spatio_temporal_coherence = self.adjacent_poi_valid / self.adjacent_poi_total
        return self.spatio_temporal_coherence
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content_retention": self.calculate_cr(),
            "edit_distance": self.edit_distance,
            "poi_sequence_similarity": self.poi_sequence_similarity,
            "spatio_temporal_coherence": self.calculate_stc(),
            "original_variables_count": self.original_variables_count,
            "retained_variables_count": self.retained_variables_count,
            "modified_variables": self.modified_variables,
            "added_variables": self.added_variables,
            "removed_variables": self.removed_variables
        }


@dataclass
class ScopeQualityMetrics:
    """
    修改范围相关指标（Scope Quality）- 已停用

    注意：此类指标已停用，原因：
    1. 缺少可靠的黄金标准标注
    2. Scope定义在不同场景下差异较大
    3. 当前评估主要关注结果质量而非编辑范围

    保留数据结构以向后兼容，但不再计算
    """

    # Scope 对比
    gold_scope_size: int = 0
    predicted_scope_size: int = 0
    overlap_size: int = 0

    # 计划总大小
    total_plan_size: int = 0

    # 标记为已停用
    enabled: bool = False

    def calculate_precision(self) -> float:
        """Scope-P = (pred ∩ gold) / pred - 已停用"""
        if not self.enabled:
            return 0.0
        if self.predicted_scope_size == 0:
            return 0.0
        return self.overlap_size / self.predicted_scope_size

    def calculate_recall(self) -> float:
        """Scope-R = (pred ∩ gold) / gold - 已停用"""
        if not self.enabled:
            return 0.0
        if self.gold_scope_size == 0:
            return 0.0
        return self.overlap_size / self.gold_scope_size

    def calculate_f1(self) -> float:
        """F1 Score - 已停用"""
        if not self.enabled:
            return 0.0
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_minimality(self) -> float:
        """Minimality = 1 - (pred / plan) - 已停用"""
        if not self.enabled:
            return 0.0
        if self.total_plan_size == 0:
            return 0.0
        return 1.0 - (self.predicted_scope_size / self.total_plan_size)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "precision": self.calculate_precision(),
            "recall": self.calculate_recall(),
            "f1": self.calculate_f1(),
            "minimality": self.calculate_minimality(),
            "note": "Scope quality metrics are disabled"
        }


@dataclass
class EvaluationReport:
    """
    完整的评估报告，包含所有指标
    """
    
    # 基本信息
    experiment_id: str
    model_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 各类指标
    constraint_satisfaction: ConstraintSatisfactionMetrics = field(
        default_factory=ConstraintSatisfactionMetrics
    )
    efficiency: EfficiencyMetrics = field(
        default_factory=EfficiencyMetrics
    )
    success_rate: SuccessRateMetrics = field(
        default_factory=SuccessRateMetrics
    )
    consistency: ConsistencyMetrics = field(
        default_factory=ConsistencyMetrics
    )

    # scope_quality 已停用，不再包含在报告中

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典"""
        report = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "constraint_satisfaction": self.constraint_satisfaction.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "success_rate": self.success_rate.to_dict(),
            "consistency": self.consistency.to_dict(),
            "metadata": self.metadata
        }

        # scope_quality 已停用，不再输出到报告

        return report
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """导出为JSON"""
        json_str = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str
    
    def print_summary(self):
        """打印简要统计"""
        print("=" * 80)
        print(f"TPE Evaluation Report - {self.model_name}")
        print("=" * 80)
        
        print("\n【约束满足率 Constraint Satisfaction】")
        print(f"  CSR (总体):      {self.constraint_satisfaction.calculate_csr():.2%}")
        print(f"  DR (结构合法):   {self.constraint_satisfaction.delivery_rate:.2%}")
        print(f"  EPR (外部事实):  {self.constraint_satisfaction.environmental_pass_rate:.2%}")
        print(f"  LPR (逻辑一致):  {self.constraint_satisfaction.logical_pass_rate:.2%}")
        print(f"  FPR (综合通过):  {self.constraint_satisfaction.calculate_fpr():.2%}")
        
        print("\n【效率 Efficiency】")
        print(f"  平均时延:        {self.efficiency.avg_latency_seconds:.2f}s")
        print(f"  平均Token:       {self.efficiency.avg_tokens_per_episode:.0f}")
        print(f"  平均步骤数:      {self.efficiency.avg_workflow_steps:.1f}")
        
        print("\n【成功率 Success Rate】")
        print(f"  ESR (编辑成功):  {self.success_rate.calculate_esr():.2%}")
        print(f"  总Episodes:      {self.success_rate.total_episodes}")

        print("\n【一致性 Consistency】")
        print(f"  CR (内容保留):   {self.consistency.calculate_cr():.2%}")
        print(f"  MED (编辑距离):  {self.consistency.edit_distance}")
        if self.consistency.spatio_temporal_coherence is not None:
            print(f"  STC (时空一致):  {self.consistency.calculate_stc():.2%}")

        print("=" * 80)
