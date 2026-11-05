# 05 · Domain 适配器模式：coding / workspace / eng 三个实例

## 接口即契约

`rrsi/domain.py:73` 的 `Domain` 类定义了核心需要的**全部**基准能力，核心自己从不运行
agent、从不判分、从不读轨迹格式（`rrsi/domain.py:28-43` 模块说明）。接口分五组：

- 任务划分：`evolve_ids` / `heldout_ids` / `smoke_ids`（`rrsi/domain.py:82-85`）；
- Evaluate：`run`（在 worktree 下跑 k trials，必须可续跑）与 `score`
  （`rrsi/domain.py:89-101`），被 `rrsi/evaluate.py:142` 调用；
- 证据：`load_trial` / `render_trace` / `task_row`，供分析师、摘要器与提案者读取；
- 闸门：`smoke`（活性检查，明确"不是选择规则"）、`critic_patterns`（泄漏黑名单，
  `rrsi/domain.py:131`）、`guards`（非补偿性判据，`rrsi/domain.py:103`）；
- 文本：`briefs`（四个搜索角色各自的 domain 段落）、`constitution`（SKILL.md +
  PATTERNS.md，`rrsi/domain.py:140`）。

适配器是 `domains/<name>/adapter.py` 导出的单例 `DOMAIN`，由 `load_domain`
（`rrsi/domain.py:145`）用 `importlib.util.spec_from_file_location` 按文件路径动态加载
并注入 `root`。

## 三个实例，一套方法

- **coding**（`domains/coding/adapter.py:106`，`CodingDomain`）：Terminal-Bench 2.1，
  经 harbor 运行器（Docker）；S = 通过隐藏测试的 trial 占比；`heldout_ids` 为空——
  整个套件都是 evolve 集，泛化靠 OOD 的 SWE-bench Verified。
- **workspace**（`domains/workspace/adapter.py:92`，`WorkspaceDomain`）：Harvey LAB，
  经 MCP 网关 + ReAct toolbelt harness；奖励是"通过准则数/总准则数"并按准则数加权
  （权重 = 准则总数，使 Ŝ 恰为准则通过占比，见 `rrsi/evaluate.py:38-41` 的说明）；
  有真正的 ID held-out 40 题（由 `split_workspace.py` 确定性切出）。
- **eng**（`domains/eng/adapter.py:76`，`EngDomain`）：EngDesign，冻结的逐任务代码
  verifier；extra 里保留 valid 率与 no-payload 率并喂给 `guards`
  （`domains/eng/adapter.py:221`）——工程设计的底线（必须真的交出可验证的产出）
  不参与加权。

## 模式的收益

同一个 `Run.round`（`rrsi/loop.py:269`）驱动三个实例：分数聚合、噪声带校准、成本规则、
修剪全部复用；domain 之间只差"怎么跑、怎么判、怎么读证据、怎么说话"。新增一个基准 =
一个 adapter 文件 + 起始 harness + `rrsi.json` 超参，核心零改动（README "Adding a
domain" 一节的承诺）。代价是适配器必须自己实现续跑（`run` 的 resume-safe 契约写在
`rrsi/domain.py:90-93`），核心通过 `invalid_missing_frac`（`rrsi/config.py:83` 附近）
兜底校验缺失 trial 的比例。
