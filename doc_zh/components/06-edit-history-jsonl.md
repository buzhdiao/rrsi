# 06 · 编辑历史 JSONL 与可审计性

## 每个编辑一行记录

`History`（`rrsi/history.py:74` 的类）把论文 Eq. (history) 的 `L_t` 落成
`runs/<domain>/history.jsonl`：**每个编辑一条 JSONL 记录**。打包 n 个编辑（n <= b_t）
的候选只测得一次 (ΔS, ΔC, a)，包内每条编辑都携带这份测量（`append_candidate`，
`rrsi/history.py:96`）；随着 b_t 退火到 1，记录自然退化为"单组件证据"。记录字段覆盖
轮次、变体、编辑 id、组件标签 l'、假设 h'、目标失败模式、预测受影响任务、diff 路径、
ΔS/ΔC、是否被接受（a_i）、结局、打包大小与详情——从"当时想干什么"到"事后测得什么"
一条链。

## 只算测过的证据

未测量不等于证据：被 critic 拒绝、smoke 失败、评估无效的候选记为 `delta_S=None`
（`rrsi/history.py:113` 附近），不进入已尝试集 `T_t`（`rrsi/history.py:139` 的
`tried` 只看 `measured()`，`rrsi/history.py:135`）也不进入近期收益 `g_t`。这保证
`U_t`（未尝试组件）与修剪集 `B_t` 的统计不被垃圾候选污染。反过来，"发出过但没测到"
的尝试也有账：`attempted`（`rrsi/history.py:146` 附近）按组件统计全部提交过的编辑，
用于诊断。

## 给搜索自己的反馈

历史不只是日志，它回到搜索里：

- `render`（`rrsi/history.py:194`）给提案者最近 40 条紧凑视图，且有测量记录优先——
  无测量的闸门失败只保留最近 4 条，"一墙的失败记录是反馈回路，不是证据"；
- `accepted_edits` / `incumbent_component_counts`（`rrsi/history.py:153/165`）给出
  现任机制清单，是 novelty 项与修剪集 `B_t` 的数据源；
- 归因记分板 `attribution.jsonl`（`rrsi/loop.py:194` 的 `attribute`）另记每个编辑的
  预测命中率与"未预测到的回退"，最近 20 条回显给提案者——过度声称会被记账。

## 可审计性的工程保障

追加写 + 每行时间戳（`rrsi/history.py:88-93` 附近的 `append`），崩溃不丢已写记录；
每轮指令（b_t、sigma_t、B_t、delta、S\*）单独落盘 `directives.json`
（`rrsi/loop.py:325` 附近），每轮裁决落盘 `decisions.json`（`rrsi/loop.py:351`）。
需要修正时不是原地改：`readjudicate` 用 `replace_round`（`rrsi/history.py:121`）删掉
整轮逐编辑记录再重写，理由前缀 `[re-adjudicated delta=...]` 标明来龙去脉。
`rrsi.py:107-110` 的 `readjudicate` / `reevaluate` 子命令让"delta 改了之后重新裁决
历史轮"成为一等操作，而重裁决不花任何新评估。
