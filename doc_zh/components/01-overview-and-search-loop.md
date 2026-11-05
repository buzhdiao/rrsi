# 01 · RRSI 总体架构与搜索循环

## 一句话定位

RRSI（Regularized Recursive Self-Improvement，正则化递归自改进）不改进模型权重，
只改进 **harness**（agent 脚手架：提示词、控制流、工具、记忆、上下文管理）。它在固定的
evolve 任务集上反复起草、筛查、评估、挑选新的 harness 候选，用一整套"作用于搜索轨迹"
的正则化约束防止对训练任务过拟合。编辑空间保持开放——约束的是"搜索怎么走"，不是
"harness 里能放什么"。

## 代码地图

整个核心是一个与基准无关的包 `rrsi/`（`rrsi/__init__.py:38`），基准相关的一切都推到
`Domain` 适配器之后（见第 05 篇）。

- 入口 CLI：`rrsi.py:65`（`main()`），子命令 baseline / calibrate / round / run /
  readjudicate / reevaluate / heldout / smoke / status（`rrsi.py:29-44`）。
- 一轮的编排：`rrsi/loop.py:269`（`Run.round(t)`），按注释分七步：
  1. `Analyze(H_t, D_evolve)` 产出三透镜报告 `F_t`（`rrsi/loop.py:288`，分析师在
     `rrsi/analyst.py:145`）；
  2-3. 取退火编辑预算 `b_t`（`rrsi/schedule.py:51`）、停滞标志 `sigma_t`
     （`rrsi/history.py:220`）、已尝试集 `T_t`（`rrsi/history.py:139`）、探索指令
     `E_t`（`rrsi/history.py:228`）与修剪集 `B_t`（`rrsi/history.py:178`）
     （`rrsi/loop.py:313`）；
  4. 起草并筛选 `m` 个候选变体（`rrsi/loop.py:331`）：每个变体在自己的 git worktree
     里由提案者就地编辑（`rrsi/propose.py:274`），再过 critic 泄漏筛查
     （`rrsi/critic.py:137`）与活性 smoke（`rrsi/loop.py:625` 附近）；
  5. 通过闸门的候选并行评估（`rrsi/loop.py:341`，`rrsi/evaluate.py:142`）；
  6. Algorithm 2 裁决：可接纳性、argmax、S\*、历史落账（`rrsi/loop.py:347`，
     `rrsi/selection.py:143`）；
  7. 获胜者快进 `evolve/<domain>` 分支，更新 frontier（`rrsi/loop.py:371-373`）。
- 长程驱动：`rrsi/driver.py:67`（`drive()`）逐轮调用子进程，断点续跑，连续 3 次
  基础设施失败熔断（`rrsi/driver.py:56`）。

## 状态放在哪里

`runs/<domain>/` 是唯一事实来源：`frontier.json`（现任、S\*、轨迹、配置）、
`history.jsonl`（编辑历史 L_t）、`jobs/<job>/eval.json`（每次评估）、`r<t>/`（每轮的
报告、提案、diff、裁决）。所有写路径都是"先落盘再继续"，任何一轮崩溃后重跑都会复用
已完成的产物（如 `rrsi/loop.py:642` 复用 eval.json），因此整个搜索可断点续跑、可审计。

## 为什么这样设计

分数轨迹 `trajectory`、历史最好分 `S*` 与噪声带 `delta` 三个状态量撑起了选择侧的全部
判据；编辑历史 `L_t` 撑起了提案侧的全部条件化。两边的正则化都不需要访问模型内部，
只需要"每次改动的可测量后果"，这正是"证据可审计"的工程含义。
