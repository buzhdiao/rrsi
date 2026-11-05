# 08 · 结果解读：evolve / ID held-out / OOD 的区别

## 三种"分数"不是一回事

- **evolve 分数**：搜索所依据的 split（`Domain.evolve_ids`，`rrsi/domain.py:82`）上的
  分数。选择侧的每一轮 argmax 都在优化它，因此它**必然**上升——它的作用是证明
  "搜索机制在动"，不证明泛化。README 结果表中 `Role = evolve` 的行（如
  Terminal-Bench 2.1 的 74.2 -> 80.2）就是它。
- **ID held-out**：同一基准、同一分布内，但搜索从未见过的任务
  （`Domain.heldout_ids`，`rrsi/domain.py:83`）。workspace 实例由
  `domains/workspace/split_workspace.py` 从 Harvey LAB 检出中确定性切出 40 题，
  核心代码对它们零接触。README 中 Harvey LAB 的 "ID held-out" 86.9 -> 89.2 属于
  这一类：它是"在分布内泛化"的最强证据。
- **OOD（out-of-distribution）**：换基准的最终评估，如 coding 用 SWE-bench
  Verified、workspace 用 JobBench / GDPval / APEX-Agents、eng 用 Frontier-Eng。
  这些任务从未进入选择，衡量"学会的是通用能力还是背题"。

## 怎么跑

`rrsi.py:87-90` 的 `heldout` 子命令：`--set heldout` 评 ID held-out，`--set evolve`
评 evolve 集，`--ref` 可指定任意提交/分支（默认现任）。各 domain 的 OOD 脚本见
`domains/<name>/README.md`（如 `domains/coding/scripts/swe_eval.sh`、
`ood/run_*.sh`）。评估走的是与搜索完全相同的 `evaluate()` 管道（`rrsi/evaluate.py:142`），
产物落在 `jobs/heldout_<label>/eval.json`。

## 论文数字的正确读法

README 表中所有 Δ 都是"与同一时间窗内未演化 harness H_0 的差"。典型模式：
evolve 增益最大（+6.0），ID held-out 缩水但保持正值（+2.3），OOD 再缩水但依然为正
（+1.8 至 +4.7）。这个梯度本身就是论文的卖点：**正则化不是不让 evolve 分数涨，
而是让涨的部分有一部分是通用的**——对照 unregularized 基线的"evolve 大涨、OOD 归零"
才能看出差别。

## 换策略模型也是泛化测试

README 指出同一 coding 实例把冻结策略从 Claude 换成 Gemini 3.5 Flash 后，
Terminal-Bench 64.6 -> 78.7、SWE-bench 76.8 -> 79.0：为 A 模型搜出的 harness 改动
在 B 模型上仍大部分有效，说明学到的机制对"策略模型"这一维度也有迁移性。判读任何
RRSI 结果时的三个问题：这行分数是哪个 split？它进没进过选择？对照的 H_0 是不是
同一时间窗、同一评估管道？
