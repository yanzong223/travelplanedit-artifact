"""
Bucket Sampler
采样conflict bucket，控制分布
"""

import random
from typing import List, Tuple, Optional
import yaml


class BucketSampler:
    """Conflict bucket采样器"""

    # Scope定义
    SCOPES = ["parameter", "structural", "compositional"]

    # Dimension定义
    DIMENSIONS = ["temporal", "spatial", "resource", "structural", "semantic"]

    # Violation定义
    VIOLATIONS = ["overflow", "infeasible", "overlap", "discontinuity", "incompatibility"]

    def __init__(self, config_path: str = None):
        """
        初始化采样器

        Args:
            config_path: bucket distribution配置文件路径
        """
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.config = config
        else:
            # 默认配置
            self.config = {
                "scope_distribution": {
                    "parameter": 0.4,
                    "structural": 0.4,
                    "compositional": 0.2
                },
                "dimension_distribution": {
                    "temporal": 0.3,
                    "spatial": 0.2,
                    "resource": 0.2,
                    "structural": 0.2,
                    "semantic": 0.1
                },
                "violation_distribution": {
                    "overflow": 0.3,
                    "infeasible": 0.2,
                    "overlap": 0.2,
                    "discontinuity": 0.2,
                    "incompatibility": 0.1
                },
                "multi_label_config": {
                    "single_conflict": 0.6,
                    "double_conflict": 0.3,
                    "triple_conflict": 0.1
                },
                "day_distribution": {
                    "single_day": 0.7,
                    "multi_day": 0.3
                }
            }

    def sample_scope(self) -> str:
        """采样scope"""
        return self._weighted_sample(self.config["scope_distribution"])

    def sample_dimension(self) -> str:
        """采样dimension"""
        return self._weighted_sample(self.config["dimension_distribution"])

    def sample_violation(self) -> str:
        """采样violation"""
        return self._weighted_sample(self.config["violation_distribution"])

    def sample_bucket(self) -> Tuple[str, str, str]:
        """
        采样一个conflict bucket

        Returns:
            (scope, dimension, violation) 元组
        """
        scope = self.sample_scope()
        dimension = self.sample_dimension()
        violation = self.sample_violation()
        return (scope, dimension, violation)

    def sample_multi_label_buckets(self) -> List[Tuple[str, str, str]]:
        """
        采样多个conflict buckets（用于multi-label冲突）

        Returns:
            bucket列表
        """
        # 决定label数量
        label_count = self._sample_label_count()

        buckets = []
        for _ in range(label_count):
            bucket = self.sample_bucket()
            # 确保不重复
            while bucket in buckets:
                bucket = self.sample_bucket()
            buckets.append(bucket)

        return buckets

    def _sample_label_count(self) -> int:
        """采样冲突label数量"""
        config = self.config["multi_label_config"]
        rand = random.random()

        if rand < config["single_conflict"]:
            return 1
        elif rand < config["single_conflict"] + config["double_conflict"]:
            return 2
        else:
            return 3

    def _weighted_sample(self, distribution: dict) -> str:
        """根据权重采样"""
        items = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(items, weights=weights, k=1)[0]

    def is_multi_day_plan(self) -> bool:
        """
        决定是否生成多日plan

        Returns:
            是否为多日
        """
        config = self.config["day_distribution"]
        rand = random.random()
        return rand >= config["single_day"]

    def validate_bucket(self, bucket: Tuple[str, str, str]) -> bool:
        """
        验证bucket是否有效

        Args:
            bucket: (scope, dimension, violation) 元组

        Returns:
            是否有效
        """
        scope, dimension, violation = bucket
        return (scope in self.SCOPES and
                dimension in self.DIMENSIONS and
                violation in self.VIOLATIONS)

    def get_bucket_name(self, bucket: Tuple[str, str, str]) -> str:
        """
        获取bucket的名称

        Args:
            bucket: (scope, dimension, violation) 元组

        Returns:
            bucket名称字符串
        """
        scope, dimension, violation = bucket
        return f"{scope}_{dimension}_{violation}"

    def get_all_possible_buckets(self) -> List[Tuple[str, str, str]]:
        """
        获取所有可能的bucket组合

        Returns:
            所有bucket列表
        """
        buckets = []
        for scope in self.SCOPES:
            for dimension in self.DIMENSIONS:
                for violation in self.VIOLATIONS:
                    buckets.append((scope, dimension, violation))
        return buckets


if __name__ == "__main__":
    # 测试代码
    sampler = BucketSampler()

    print("=== 单bucket采样测试 ===")
    for i in range(10):
        bucket = sampler.sample_bucket()
        print(f"{i+1}. {sampler.get_bucket_name(bucket)}")

    print("\n=== Multi-label bucket采样测试 ===")
    for i in range(10):
        buckets = sampler.sample_multi_label_buckets()
        bucket_names = [sampler.get_bucket_name(b) for b in buckets]
        print(f"{i+1}. {bucket_names}")

    print("\n=== 多日plan采样测试 ===")
    multi_day_count = 0
    for i in range(100):
        if sampler.is_multi_day_plan():
            multi_day_count += 1
    print(f"多日plan数量: {multi_day_count}/100")

    print("\n=== 所有可能的buckets ===")
    all_buckets = sampler.get_all_possible_buckets()
    print(f"总数: {len(all_buckets)}")
