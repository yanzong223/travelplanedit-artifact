# <run_id> 主实验分析

## 0. Sanity Check / 基础纠错与归因排查

- 运行是否完整：N=__，run success=__，evaluation failures=__，空输出/格式错误=__。
- API / 模型服务：quota、401、timeout、rate-limit、截断、空返回是否异常？
- 框架实现：framework、model、tool_profile、exposure_mode 是否与 registry 一致？是否跑成 fallback/default？
- 工具链：tool schema validation、missing field、tool empty result、retrieval/editor/verifier cascade failure 是否集中？
- IO 对齐：dataset split、sample set、run id、report paths 是否正确？是否混入旧结果或漏 category？
- Level3 cohort：L3 eligible n=__。若 cohort 偏小，禁止把 Level3 efficiency 与其他方法直接横向比较。
- 历史趋势：若某方法本次结果与历史差异过大，先标记 review，不直接写成方法结论。
- 结论：`pass / review / block`；分别列出工程异常、配置错误、运行错误、模型能力不足、方法设计缺陷。

不要写：“模型很差所以分数低。”

应该写：“先说明是否存在 API/tool/config/IO 异常；排除后，再归因为方法能力或 benchmark 暴露的短板。”

## 1. 本次实验结果摘要

- L1/L2/L3 各说明什么：可行计划、受约束编辑成功、通过前置 gate 后的 minimal-edit 代价。
- 本方法最可信的总体结论是什么？
- 哪些结果因为 sanity warning 需要谨慎？
- 当前结果是否适合进入论文主表，还是应先标记为待复核？

不要只复述分数。

应该把分数翻译成 benchmark 能力判断，并明确哪些判断受工程异常影响。

## 2. 横向对比

- 与同模型同 split 当前 best baseline 的 L1 overall、L2 combined、L3 eligible、L3 cost/retention 差距是多少？
- 方法之间的主要差异体现在哪里：可行性、origin preservation、edit target satisfaction，还是 minimal-edit 控制？
- 是否出现高 feasibility 但低 edit success？
- 是否出现低 L1 但高 L2 evaluable 的幸存者偏差？
- 新 baseline 插入后，是否改变当前 best baseline 的排序？

不要把不同 evaluable denominator 的数字直接横比。

应该同时报告 pass rate 与 `(n=...)`，尤其是 Level2 与 Level3。

## 3. 分 Level 解读

### Level1: Plan Validity

- 方法是否能输出 benchmark 可执行的旅行计划？
- 是否保留了未被编辑目标覆盖的 origin hard constraints？
- 若 L1 失败，主要来自 feasibility 还是 origin preservation？

### Level2: Edit Success

- 方法是否真正满足 edit logical targets？
- preference target 是否可评且达标？
- logical 与 preference 是否存在明显差距？
- hard-only 和 preference-heavy 子集是否需要分开讨论？

### Level3: Edit Efficiency / Minimal Edit

- 先报告 L3 eligible n 与 eligible rate。
- 再解释具体 edit operations、affected-day count、retention、sequence distance、activity change ratio 与 rollback attribution；不得再按 parameter/structural/compositional 三分类组织结论。
- 若 Level1/2 通过率低，Level3 只能作为条件成立后的 minimal-edit 描述，不能孤立解释为方法效率强弱。

不要写：“Level3 高/低说明方法编辑效率高/低。”

应该写：“在通过 Level1+Level2 的样本上，该方法表现出更小/更大的编辑 scope 或 change ratio。”

## 4. 失败模式 / Trade-off

- 工程异常：API、工具契约、配置、IO 是否足以解释异常结果？
- 运行错误：early stop、空输出、格式错误、解析失败、invalid final plan schema 是否集中？
- 方法短板：可行性、origin preservation、logical target、preference target、minimal-edit 控制分别有哪些问题？
- Trade-off：可行性 vs 编辑成功、编辑成功 vs 改动幅度、工具强度 vs schema 稳定性。
- 是否存在 retrieval / editor / planner / verifier 级联失效？

不要把工程异常写成方法能力结论。

应该先把异常归为工程异常 / 配置错误 / 运行错误，再写模型能力不足 / 方法设计缺陷 / benchmark 暴露出的真实短板。

## 5. 和 Benchmark 目标的对应关系

- 哪些结果支持“能生成可行计划”不等价于“会做受约束编辑”？
- 哪些结果体现 editing 与 from-scratch planning 的差异？
- 哪些方法只是在 feasibility 上较好，但没有稳定完成 edit target？
- 哪些结果说明 Level3 必须 gate-aware 解读？

## 6. 可写入论文的 Finding 草稿

- Finding 1: __
- Finding 2: __
- Finding 3: __

写法提示：

- 从 sanity check 通过的稳定现象中提炼 finding。
- 把异常结果写成 caveat 或 threat to validity，不直接写成结论。
- 对 Level3 使用“conditional on Level1+Level2 success”这样的限定语。
