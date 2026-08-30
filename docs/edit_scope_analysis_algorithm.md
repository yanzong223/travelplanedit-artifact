# Edit Scope Analysis：依赖闭包反事实归因算法

> **论文术语映射（2026-08-15）：** 论文面向读者的上位名称为 **Editing Behavior**。
> 本文档保留 `Edit Scope Analysis`，因为它描述的是底层算法、代码入口和历史 artifact 字段。
> `Actual Edit Scope` 是 Editing Behavior 的一个具体观察对象；Level 2 的论文名称为
> **Edit Correctness**，实现中仍可能显示 `Request Satisfaction` 或 `Edit Success`。

本文档定义当前 Edit Scope Analysis 使用的统一算法、代码入口、输出契约和报告方式。算法名为 **Dependency-Closed Counterfactual Attribution（DCCA，依赖闭包反事实归因）**，当前程序证明契约版本为 `1.1.1`，rollback family 版本为 `dcca-rollback-family-v1`。`1.1.1` 将 `FlightID` 纳入航班身份、把 train--airplane 的同一行程腿变化合并为跨方式 replacement，并修正显式跨城交通方式请求的 direct-target attribution；rollback family 本身未改变。

## 1. 要解决的问题

Plan Validity（计划有效）和 Edit Correctness（实现字段：Request Satisfaction）只回答“结果是否成功”，不回答“为获得这个结果，模型改了多少，以及额外变化是否必要”。DCCA 在同时通过这两个 gate 的输出上，对模型实际产生的变化进行归因。

此前有两个主要误差来源：

1. 只比较活动增删、替换和时间，遗漏 `transports`、`cost`、`price`、`tickets`、`room_type`、`rooms` 等字段变化；
2. 单独撤回一个活动或交通字段后，可能人为制造断裂。例如原计划是 `A → B → C`，模型改成 `A → X → C`，只撤回活动而不联合恢复相邻交通，会得到“活动已回到 B，但交通仍从 X 出发”的无效反事实。

DCCA 因此先做字段级原子差分，再对每个目标外变化构造最小依赖闭包，并在所有受影响路线边界上联合修复后重新运行完整 gate。

## 2. 评价边界

DCCA 的设计目标是：覆盖实际字段变化、提供逐变化诊断、支持离线重算，并对“可撤回”给出可审计证据。

它评估：

- 模型输出相对原计划发生了哪些变化；
- 哪些变化直接实现编辑目标；
- 哪些变化在当前完整测试的 rollback family 内没有找到有效撤回方案；
- 哪些变化可在 evaluator 覆盖范围内安全撤回；
- 哪些变化因反事实证据不完整而暂时无法归因。

它不评估：

- 人类审美、体验价值或 evaluator 未编码的偏好；
- 唯一正确的编辑路径；
- 未经标注的全局 allowed edit set；
- “所有目标外变化在现实世界中都无关”这一强命题。

因此，Verified Removable-Change Rate 是成功输出上的条件性质量指标，不是第三个成功门槛，也不是全局无关性的证明。

## 3. 输入、单位与五类输出

输入为：原计划 $P_0$、编辑后计划 $P_1$、编辑约束 $C$、完整成功 gate $G$，以及可选的本地路线证据缓存 $R$。

基本计数单位是“推断出的 changed activity”，包括匹配活动上的一个或多个原子操作，以及插入活动。对于 `day_count` 等结构请求，算法允许加入一个 virtual scope unit。它不是逐 JSON 字段计数：同一活动同时改变时间、费用和交通，changed-activity count 仍为 1，但证明记录会保留全部 `atomic_op_types`。

每个 changed unit 最终只能属于以下一类：

| 类别 | 含义 | 证据标准 |
|---|---|---|
| `direct_target` | 直接实现明确编辑目标 | 实体、约束类型或结构目标可直接定位 |
| `rollback_required_support` | 在算法定义且完整测试的 rollback family 内未找到有效反事实 | `rollback_family_complete=true`，且所有已测试候选均失败 |
| `scope_authorized_completion` | 请求结构明确授权的补全 | 当前规则主要覆盖新增日的基本角色 |
| `verified_removable` | 至少存在一个完整有效的撤回候选 | rollback 后 Plan Validity 与 Edit Correctness 均保持通过 |
| `unresolved` | 当前证据不足以证明必要或可撤回 | rollback、gate 或路线证据不完整 |

守恒条件为：

\[
N_{\mathrm{all}}
=N_{\mathrm{direct}}+N_{\mathrm{rollback}}+N_{\mathrm{authorized}}
+N_{\mathrm{removable}}+N_{\mathrm{unresolved}}.
\]

公共 API 会自动检查这一条件；违反时直接报错，不生成看似合理的比例。

## 4. 原子差分

实现位于 `src/evaluation/benchmark/diffing.py`。活动先按实体、时间重叠和位置关系匹配，再产生以下原子操作：

- `insert`、`delete`、`replace`、`reorder`；
- `change_time`；
- `change_transport`；
- `change_attribute`。

比较前会规范化数字格式、时钟格式、文本空白与大小写、安全的交通模式别名，以及字典键顺序和不影响语义的元数据。

交通语义变化进一步记录为 mode、duration/timing、cost/capacity、endpoint、distance、topology 或 other semantic change。仅格式或未知元数据变化单独标为 equivalent rewrite，不与路线语义变化混淆。

## 5. DCCA 算法

```text
Algorithm DCCA(P0, P1, C, G, R)
  1. 报告层若判定 P1 未同时通过 Plan Validity 与 Edit Correctness：不计算质量指标
  2. Δ ← AtomicDiff(P0, P1)
  3. D ← GroundDirectTargetUnits(Δ, C)
  4. 对每个 u ∈ Δ \ D：
       a. Q0 ← Rollback(P1, u)
       b. K(u) ← MinimalDependencyClosure(u, P0, P1, Δ)
       c. 按版本化规则生成候选集合 Q_A(u)：
            - 单点 rollback Q0
            - 对 K(u) 的联合 rollback
            - 可复用的原计划 inbound route
            - 对所有断点联合枚举缓存中的 walk / metro / taxi 路线
       d. 对每个 q ∈ Q_A(u) 重新运行约束、可行性、交通连续性和完整 gate G
       e. 保存每个候选的哈希和验证结果；若存在完整有效 q：
            若 u 是 scope-authorized completion，则标 authorized
            否则保存 q 作为 witness，并标 verified_removable
          否则若 Complete_A(u) 且所有 q 均失败：标 rollback_required_support
          否则：标 unresolved
  5. 校验五类计数守恒并计算 lower / upper / proof coverage
  6. 返回逐单元证明记录和汇总指标
```

若 raw single rollback 自身制造了 transport discontinuity，它只作为生成 repair candidates 的中间状态，不直接作为 $Q_A(u)$ 中的有效候选；$Q_A(u)$ 收录其 dependency-closed 或 route-repaired descendants。可靠离线城市映射确认的新增跨城断裂可以作为结构性否定证据，不需要再枚举市内路线；既存跨城问题本身不能阻止一个不新增问题且完整 gate 通过的正向 witness。

定义

\[
\mathrm{Valid}(q) := \mathrm{LocalChecks}(q)=1 \land G(q)=1,
\]

其中 `LocalChecks` 包括编辑约束、repository feasibility、交通连续性和新增跨城断裂检查。$A$ 表示固定的算法与 family 版本。`Complete_A(u)` 不是“搜索了所有想象得到的行程”，而是以下条件均成立：

- $Q_A(u)$ 的所有算法分支均已生成或得到确定的否定结果；
- 若 rollback 产生需要市内修复的路线断点，每个断点的 `walk`、`metro`、`taxi` 状态均为 `ok` 或 `ok_no_route`，且断点数没有超过实现上限；
- baseline 和所有实际生成的候选都完成了同一版本的完整 gate；
- 候选生成、验证或数据读取过程没有异常。

三个量词是算法语义的核心：

\[
u\in\mathrm{Removable}
\iff \exists q\in\mathcal{Q}_A(u),\;\mathrm{Valid}(q)=1,
\]

\[
u\in\mathrm{RollbackRequiredSupport}
\iff \mathrm{Complete}_A(u)
\land \forall q\in\mathcal{Q}_A(u),\;\mathrm{Valid}(q)=0,
\]

\[
u\in\mathrm{Unresolved}
\iff \neg\mathrm{Complete}_A(u)\land
\neg\exists q\in\mathcal{Q}_A(u),\;\mathrm{Valid}(q)=1.
\]

这使 `unresolved` 表示“当前没有正向 witness，负向证据又不完整”，而不是“模型一定改错了”。正向和负向证据故意不对称：找到一个有效 witness 就足以证明 evaluator-relative removability；没有找到 witness 只有在 family 完整时才能支持负向结论。

### 5.1 分类优先级

1. 明确定位的目标单元先归为 `direct_target`，不进入 rollback 测试；
2. 若找到有效 witness，且该单元属于结构请求授权的基本补全，则归为 `scope_authorized_completion`；
3. 其余带有效 witness 的单元归为 `verified_removable`；
4. 没有 witness、但 `Complete_A(u)=true` 且所有候选失败，归为 `rollback_required_support`；
5. 其余情况归为 `unresolved`。

旧字段 `hard_required_support_change_count`、`required_support_change_count` 和 `hard_support_share` 只作为 v1.0 artifact 兼容别名保留。新文档、表格和正文不得用它们表达全局必要性。

### 5.2 三个可证明性质

这里的“程序保证”指 by-construction 分支、运行时断言、结果契约校验和可执行回归测试；当前没有使用 theorem prover 对 Python 实现做机械化形式验证。

| 性质 | v1.1 程序保证 | 不能推出的更强结论 |
|---|---|---|
| Determinism / reproducibility | 固定版本化的 $(P_0,P_1,C,G,R)$，且 $G$ 为纯确定函数时，diff、候选顺序、witness、归因和规范化结果哈希相同 | 若 $G$ 读取实时 API、当前时间或随机状态，则不保证跨运行相同 |
| Soundness of verified removable | 每个 `verified_removable` 单元都保存 witness $q$；其哈希必须出现在 `tested_candidate_outcomes` 中，且保存时 `Valid(q)=1` | 不证明该变化在现实世界中没有体验价值，也不证明所有 evaluator 都会接受 $q$ |
| Conservative incomplete evidence | 缺少 baseline/candidate gate、路线状态不完整、超过边界上限或发生异常时，若又没有正向 witness，则只能进入 `unresolved` | `unresolved` 不等于错误、必要或可撤回 |

此外，`rollback_required_support` 发布前必须满足 `rollback_family_complete=true`，且保存的所有 candidate outcomes 都是 `fully_valid=false`。这是对版本化 $Q_A(u)$ 的有限全称检查，不是对所有可能行程的全称证明。

## 6. 最小依赖闭包与多断点路线修复

行程内交通存储在目的活动的 `transports` 字段上。因此，当 `change_transport` 的目的活动被撤回时，算法检查它的前驱活动：

- 若前驱是模型插入的活动，将该插入一并撤回；
- 若前驱被删除、替换或修改，将恢复同一原边界所需的 changed predecessor 加入闭包；
- 只加入恢复该 inbound boundary 所需的变化，不扩张到整天或全计划。

例如，原计划为 `A → B → C`，编辑后为 `A → X → C`，而到达 `C` 的 inbound transport 也从 `B → C` 改为 `X → C`。若只撤回 `C` 上保存的 transport，会得到“前序仍是 X、路线却从 B 出发”的假反事实。最小 dependency closure 只把恢复这条边界所必需的 changed predecessor `X/B` 一并撤回；它不会自动撤回当天其他活动。

联合 rollback 可能产生多个断点。算法先按目的活动去重断点，然后为每个断点收集 `walk`、`metro`、`taxi` 候选，计算这些边界候选的笛卡尔积。每个组合一次性写回所有断点，再运行完整 gate。当前实现最多处理 6 个断点，以防异常计划导致指数爆炸；实际 187-task cohort 中观察到的断点数不超过 3。

路线证据状态：

- `ok`：有可测试路线；
- `ok_no_route`：查询成功且确认无路线，是完整的否定证据；
- `query_error` 或缺失：证据不完整，对应单元保持 unresolved。

算法也检查可靠映射下的新跨城相邻断裂，避免用市内路线修复跨城缺口。

## 7. 行程级例子

下面使用与程序测试相同的行程字段结构。时间仅为说明决策逻辑，不代表某个历史 artifact 已获得 v1.1 证明。

### 7.1 Verified removable：存在正向 witness

原计划 $P_0$：

```text
08:00--09:00  A 景点
11:00--12:00  old 景点
```

编辑请求要求把 `old` 换成 `target`。模型输出 $P_1$ 同时把 A 改为 `07:00--08:00`。`target` 的替换是 `direct_target`；A 的时间变化是候选单元 $u$。算法把 A 恢复为原时间，得到 $q$，而 `target` 仍存在、计划无新增冲突且完整 gate 通过。因此保存整个 $q$、其 SHA-256 和 gate components，并归为 `verified_removable`。

### 7.2 Rollback-required support：仅对完整测试族成立

若 `target` 有固定入场时间，A 提前是为留出已验证的通勤时间；撤回 A 的时间后，单点候选违反时间窗。若 rollback 还改变了交通边界，算法会继续测试 dependency closure、原计划可复用路线以及缓存中所有 `walk/metro/taxi` 联合组合。只有这些分支全部得到结论、完整 gate 均已运行且没有候选通过，才归为 `rollback_required_support`。这里可以说“在 $Q_A(u)$ 内不可撤回”，不能说“任何可能行程中都必然需要这项修改”。

### 7.3 Unresolved：API 查不到不能推出必要

仍以 `A → X → C` 为例。撤回 X 后需要重新验证 `A → C`。如果 metro 查询为 `query_error`，即使当前没有候选通过，也不能把 X 判为 rollback-required；该单元进入 `unresolved`。若以后补齐版本化 route cache，可以在不重跑模型的情况下重新归因。

### 7.4 “至少一个历史古迹”如何验证

对于 `semantic_type_requirement(value="历史古迹", min_count=1)`，验证器按 `target_city` 和 POI 精确名称查询 ChinaTravel Attractions 表的 `type` 字段：

\[
\mathrm{matched}=\sum_{a\in\mathrm{Attractions}(P)}
\mathbf{1}[\mathrm{type}(a)=\text{历史古迹}],
\qquad
\mathrm{pass}\iff \mathrm{matched}\ge 1.
\]

若约束限定某一天，只统计该日；若设置 `strict_majority=true`，还要求匹配数超过该 scope 内景点总数的一半。精确名称查不到时类型为空，不计为匹配。该步骤使用固定数据库查询，不使用 LLM；它的正确性上限由 POI 名称对齐和数据库类型标注决定。

### 7.5 现存 artifact 中的实际行程案例

历史诊断 `structural_temporal_overflow/sample_000003` 要求相邻景点通勤不超过 25 分钟。红岩是明确 direct target；删除重庆润泽射击俱乐部后，需要重新验证红岩革命纪念馆到民俗文化村的边界。冻结路线证据为：walk 258 分钟、metro 59 分钟、taxi 32 分钟，三个 mode 都超过上限。

这个案例展示了两点：第一，不能把删除活动后遗留的错误 transport endpoint 当成有效 rollback；第二，即使三个 mode 都失败，也只有在 v1.1 确认候选族完整且完整 gate 全部运行后，才能使用 `rollback_required_support`。该案例的现存分类来自 v1.0 artifact，因此当前只能作为 change/route measurement 案例，不能直接冒充 v1.1 completeness proof。

### 7.6 为什么 DCCA 不需要 LLM

DCCA 接收已经编译的结构化约束 $C$，不负责把自然语言请求解释成约束。归因阶段的操作均有确定性判定：

| 操作 | 证据来源 |
|---|---|
| plan diff 与活动匹配 | 结构化字段、规范化规则和固定 tie-breaking |
| rollback 与 dependency closure | 原计划、编辑计划和显式 transport topology |
| “历史古迹”等语义类型 | 固定 POI 数据库的精确名称查找 |
| 路线候选 | 版本化 `walk/metro/taxi` cache |
| 成功与可行性 | machine-checkable constraints 与完整 gate |

所以这里需要的是可重放的状态变换和验证，而不是生成式语义猜测。LLM 可以被用来提出 family 外的新 replanning 候选，但那会改变 $Q_A(u)$ 的定义、成本与可复现性，并且仍需确定性 gate 验证；它不是当前三个程序性质成立的必要条件。

## 8. 复杂度与数据分布

记 $n=|\Delta\setminus D|$ 为需要归因的非直接 changed units，$b_u$ 为撤回单元 $u$ 后受影响的路线边界数，$m=3$ 为固定路线模式数，$T_V$ 为一次本地约束与可行性检查成本，$T_G$ 为一次完整 gate 成本。当前 route cache 对每个边界、每个 mode 最多形成一个候选分支，因此：

\[
|Q_A(u)|\le 3 + m^{b_u},\qquad b_u\le 6,
\]

其中常数项覆盖单点 rollback、dependency closure 和可复用的 origin inbound repair；某些分支不适用时实际候选更少。归因阶段的上界可写为：

\[
O\!\left(T_{\mathrm{diff}}+
\sum_{u=1}^{n}|Q_A(u)|(T_V+T_G+|P_1|)\right).
\]

最坏路线组合为 $3^6=729$，所以理论上对边界数指数增长；上限 6 是显式的安全阀，不是“可扩展到任意复杂计划”的保证。绝大多数不破坏交通连续性的 rollback 只有一个候选，运行时间近似随 changed-unit 数线性增长。

复杂度与数据分布直接相关：

| 数据特征 | 对计算量的影响 | 对归因分布的影响 |
|---|---|---|
| 模型重写的活动越多 | $n$ 增大，近似线性增加 rollback 次数 | direct 之外的候选单元增多 |
| replacement / reorder / transport rewrite 越多 | 更容易产生较大的 $b_u$，路线组合数上升 | route evidence 不完整时 unresolved 增多 |
| 路线 cache 中 `ok` 比例高 | 需要实际运行更多候选 gate | 更可能找到 witness，或在完整搜索后得到 rollback-required |
| `ok_no_route` 比例高 | 无需实例化对应路线候选 | 可提供完整否定证据，不等于 query failure |
| `query_error` / missing 比例高 | 计算可能更快，但证据不完整 | 不得产生负向结论，unresolved 上升 |
| 行程更长但实际改动稀疏 | diff 成本增加，归因候选未必明显增加 | 主要由 $n$ 而不是总活动数决定 |

因此实验报告除运行时间外，还应报告 changed-unit 数、route-boundary 数分布、候选数分布和 unresolved 原因分布；只报告平均运行时间会掩盖模型 rewrite style 带来的差异。

## 9. 指标

只在 Gate 1 和 Gate 2 同时通过的输出上计算：

\[
R_{\mathrm{lower}}
=\frac{N_{\mathrm{removable}}}{N_{\mathrm{all}}},
\qquad
R_{\mathrm{upper}}
=\frac{N_{\mathrm{removable}}+N_{\mathrm{unresolved}}}{N_{\mathrm{all}}}.
\]

当 unresolved 为 0 时，lower 与 upper 相等。Unresolved rate 为：

\[
R_{\mathrm{unresolved}}
=\frac{N_{\mathrm{unresolved}}}{N_{\mathrm{all}}}.
\]

Proof coverage 为：

\[
R_{\mathrm{proof}}
=\frac{N_{\mathrm{direct}}+N_{\mathrm{rollback}}+N_{\mathrm{removable}}}
{N_{\mathrm{all}}}.
\]

`scope_authorized_completion` 依据策略授权，而非 rollback 证明，因此故意不进入 proof coverage 分子。报告时必须同时给出 eligible $n$、unresolved rate 和 proof coverage；方法间 scope 比较使用同任务 matched eligible cohort，all-eligible 只作 coverage-conditioned 描述。

## 10. 当前实现与代码入口

| 层 | 文件 | 职责 |
|---|---|---|
| 公共 API 与契约 | `src/evaluation/edit_scope.py` | 版本、五类枚举、计数/比例、守恒校验、统一入口 |
| 原子差分 | `src/evaluation/benchmark/diffing.py` | 活动匹配、字段规范化、原子操作 |
| DCCA 引擎 | `src/evaluation/cascade_analysis.py` | 目标定位、闭包 rollback、联合路线修复、逐单元归因 |
| 路线证据 | `src/evaluation/route_evidence.py` | 版本化 cache、key、状态校验 |
| Level 3 汇总 | `src/evaluation/benchmark/level3.py` | 将 DCCA 结果附加到 benchmark result |
| 离线重算 | `scripts/recompute_matrix_reports_offline.py` | 不重跑模型，重算 benchmark report |
| 路线预计算 | `scripts/precompute_matrix_cascade_route_evidence.py` | 为 unresolved 断点补齐本地路线证据 |
| cohort 分析 | `scripts/analyze_natural_feasible_187_edit_scope.py` | all-eligible、matched cohort 和字段级诊断 |

旧的 `evaluation.cascade_analysis.analyze_cascade` 仍保留用于兼容。新代码应使用：

```python
from evaluation.edit_scope import analyze_edit_scope

result = analyze_edit_scope(
    origin_plan,
    edited_plan,
    edit_constraints,
    route_evidence_cache=route_cache,
    full_gate_validator=validate_full_gate,
)
```

结果包含 `algorithm` 元数据。`validate_edit_scope_result()` 会检查计数守恒、公开比例、witness 哈希与候选族成员关系，以及 rollback-required completeness；`summarize_edit_scope_result()` 返回稳定的 `counts` 与 `rates` 视图。

## 11. 复现、测试与已有数据

核心回归测试：

```bash
PYTHONPATH=src uv run pytest \
  tests/test_edit_scope.py \
  tests/test_cascade_analysis.py \
  tests/test_benchmark_evaluation.py \
  tests/test_scope_reference.py -q
```

离线重算不会调用模型。使用固定 matrix manifest、原始 evaluation 文件和版本化 route cache 运行 `scripts/recompute_matrix_reports_offline.py --compute-cascade`，随后运行 `scripts/analyze_natural_feasible_187_edit_scope.py`。

v1.1 文档所列核心命令当前为 151 个测试通过，覆盖固定输入确定性、witness replay、候选族成员关系、完整负向证据、不完整 gate/route 进入 unresolved、scope reference、route-request 透传、proof artifact 持久化，以及 v1.0 字段兼容读取。

2026-08-18 完成的 187-task、12-group v1.1 离线重算提供以下 change measurement：

| v1.1 测量项 | 数值 |
|---|---:|
| Tasks / model-method groups | 187 / 12 |
| Eligible outputs | 503 |
| Changed / virtual impact units | 1,606 |
| Direct target | 329 |
| `rollback_required_support` | 970 |
| Scope-authorized completion | 6 |
| Verified removable | 301 |
| Unresolved | 0 |
| Pooled removable lower bound | 18.74% |
| Pooled removable upper bound | 18.74% |
| Pooled proof coverage | 99.63% |

正式报告位于 `reports_natural_feasible_187_dcca_v1_1_20260818_v2/`，每个 model-method 组同时保存一份完整 `*_edit_scope_proofs.jsonl`。路线预计算共解析 109 个唯一 rollback 断点；323 个 mode 查询返回路线、4 个明确无路线、0 个 query error。12 份 proof artifact 共包含 503 条完整 DCCA 结果；逐条 witness、candidate-family completeness、计数守恒、内部结果哈希和外部文件 SHA 均已通过程序校验。使用相同 manifest、v2 route cache 和代码在独立目录重跑后，12/12 proof 文件字节级一致，12/12 report 在仅规范化输出路径后结构一致。

版本迁移需要分两层解释。v4 历史 cohort 为 419 个 eligible outputs、1,351 个 units（326/796/6/223/0）；IR/ground-truth 修复后的 v5 cohort 为 503 个 outputs、1,606 个 units（329/1000/6/271/0）。在相同 v5 cohort 上运行 v1.1 后，总数、direct 和 authorized 不变；30 个 legacy hard units 找到完整 passing route-repair witness，转入 verified removable，其余 970 个满足 v1.1 rollback-required 契约。因此 419→503 是 evaluator 输入修复，1000/271→970/301 才是归因契约与完整路线证据共同带来的变化。

## 12. 已知限制与 claim 边界

- 目标定位仍包含按约束类型定义的 proxy；低置信度 fallback 必须单独报告。
- 当前 dependency closure 重点解决 inbound transport 与 changed predecessor；更一般的住宿连续性、跨日资源共享等依赖仍需显式建模。
- `Complete_A(u)` 只表示版本化 family 的生成和验证分支完整，不表示 $\mathcal{Q}_A(u)$ 覆盖现实世界中的所有协调重排或全局 replanning。
- 所有 soundness 结论都相对于当前 $G$。若约束编译、POI 类型、路线数据或 feasibility rule 错误，DCCA 会一致地复现该错误，而不会自动纠正 evaluator。
- Determinism 要求固定的 route cache、ontology/data snapshot 和纯确定 gate；实时 API 或隐式随机性必须先冻结为版本化输入。
- 语义约束不需要 LLM，是因为当前命题可还原为结构化字段、精确数据库类型与确定性计数；这不等于系统能自动理解 evaluator 未编码的审美或体验价值。
- Global transport 请求不应只有一个唯一 reference scope。更合理的标注对象是 allowed route-leg closures 的集合，并允许多种等价解。
- 若论文要声称 over-edit，应另行标注小规模 allowed edit set / dependency closure，报告 scope precision、recall 与多解一致性。DCCA 的 removable rate 可作为 evaluator-relative 质量证据，但不能替代人工 reference-scope 验证。
