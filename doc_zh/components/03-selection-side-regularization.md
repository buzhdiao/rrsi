# 03 · 选择侧正则化：泄漏筛查、噪声调整地板、成本规则、within-band、修剪

选择侧是论文的 Algorithm 2，全部实现为纯函数（`rrsi/selection.py:28-39` 模块说明）。
一轮里 `m` 个候选各自带着测量 (S', C') 进入 `select_round`（`rrsi/selection.py:143`）。

## 评估前的 critic 泄漏筛查

评估一个候选是有代价的，所以筛查发生在**花掉评估之前**（`rrsi/critic.py:28-41` 模块
说明）。两层：确定性预检 `precheck`（`rrsi/critic.py:129`）用通用黑名单（凭据正则，
`rrsi/critic.py:63`）加 domain 专属模式（判分器路径、任务 id 等）直接判；LLM 审查
`review`（`rrsi/critic.py:137`）按六类必拒理由审意图与内容——任务特化、退化改动、
钻判分器空子、未声明捆绑、运行时记忆泄漏、无界工作。被拒的候选退回提案者做**有界
修复**（最多 `repair_rounds` 轮，`rrsi/loop.py:551` 附近的循环），修不好就丢弃并只
在历史里留档（`delta_S=None`，不算证据）。

## 噪声调整地板（noise-adjusted floor）

`admissible` 的第一关：`S' >= S* - delta`（`rrsi/selection.py:126` 附近的 `floor`）。
`S*` 是历史最好分，`delta` 是同一 harness 两次独立评估的分数差噪声带。低于地板的
候选视为"落在评估噪声里"，即使名义上比现任高一丁点也不接受——这直接堵死了
"靠噪声累积假进步"的通路。

## 成本规则（cost rule）

第二关分两支（`rrsi/selection.py:99` 的 `cost_rule`）：

- **真实增益**（`ΔS > delta`）：L1 预算 `ΔC <= beta0 + beta1·ΔS`（Eq. tokenbudget）。
  新增的推理 token 必须由实测增益买单：`beta0=0.10` 是白送的 10% 相对成本，
  `beta1=40` 表示每 +1pp 分数可再花 +0.4 相对成本（`rrsi/config.py:57` 的单位说明）。
- **within-band**（增益在噪声带内）：形状化规则
  `w_s·ΔS - w_c·ΔC + w_n·ν(l') > 0`。带内分数增益几乎不值钱，只认省 token 或新结构：
  `ν(l')`（novelty，`rrsi/components.py:129`）统计候选触及的、现任从未成功采用过的
  结构组件（`K_str = client_tool / skill / memory / subagent`，`rrsi/components.py:60`）。
  novelty 只做带内决胜，永远不会让一个低分候选翻盘。

第三关是 domain 的**非补偿性判据** `guards`（`rrsi/selection.py:135` 附近；eng 实例的
"有效产出率大跌/无产出率上升"即此类，见 `domains/eng/rrsi.json` 的
`max_valid_rate_drop` / `max_no_payload_rise`）——底线不参与加权，一票否决。

## argmax 与单调性

全部通过的候选中取 `S'` 最大的（`rrsi/selection.py:155` 的 `max`）；一个都没有则
`H_{t+1} = H_t`（`rrsi/loop.py:381`）——搜索可以原地踏步，绝不回退。`S*` 只升不降
（`rrsi/loop.py:381` 附近 `max(S*, S_{t+1})`），地板随之单调抬升。

## 修剪

选择侧的"减法"：连续 `n_prune` 轮没有严格正增益的组件进 `B_t`（`rrsi/history.py:178`，
Eq. prune），下一轮以"待拆除机制清单"的形式交回提案侧。增与减共同把搜索钉在
"每单位 token 成本换来的真实增益"上。
