# 02 · 提案侧正则化：退火编辑预算、编辑历史条件化、卡死重定向

提案侧回答一个问题：**每轮该让提案者改什么、改多少**。三个机制都在
`P_reg(. | H_t, F_t, L_t, b_t, E_t)` 的条件项里（`rrsi/propose.py:28-44` 模块说明）。

## 退火编辑预算 b_t（annealed edit budget）

`rrsi/schedule.py:51`（`edit_budget`）按论文 Eq. (anneal) 的余弦曲线从 `b_max` 退到
`b_min`：早期轮次允许一个候选打包多个协同编辑（探索期），后期每候选只剩一个独立编辑
（归因期）。它约束的是 `||z_t||_0`——提案中独立编辑的个数，不限制 harness 最终能包含
哪些机制。数值预算在提案者的 `done()` 里强制执行（`rrsi/propose.py:376`）：超预算的
提交被打回重报，绝不静默接受。每轮的预算随 `directives.json` 落盘（`rrsi/loop.py:325`
附近），供事后审计"当时允许改几个"。

## 编辑历史条件化（edit-history conditioning）

每轮把 `L_t` 的紧凑视图注入提案者上下文（`rrsi/history.py:194` 的 `render`，经
`rrsi/loop.py:322` 传入 `rrsi/propose.py:283` 附近的历史段）。提示词明确：被拒绝的
机制是**负证据**，不要原样重画；被接受的机制带着它挣来的增益，后续改进应对准
"有已知功劳"的组件，而不是"恰好赶上升势"的组件。再配一张**归因记分板**
（`rrsi/loop.py:194` 的 `attribute`：每条编辑预测了哪些任务、命中了几个、有没有伤到
预测之外的回退任务），把"过度声称"变成会被回看的记录。

## 卡死重定向（stall redirect 与 reserved slot）

停滞标志 `sigma_t = 1[S_t - S_{t-w} <= delta]`（`rrsi/history.py:220`）：w 轮内的
提升没有超过噪声带就算停滞。停滞后，本轮最后 `m_draft` 个候选槽位被设为
**保留探索槽位**（reserved slot，`rrsi/loop.py:337` 附近的 `reserved` 判定），必须
至少有一个编辑落在从未测量过的组件集 `U_t = K \ T_t` 上。这个约束被强制执行两次：
提案者 `done()` 里按声明标签查一次（`rrsi/propose.py:386-390`），critic 通过后按
**diff 实际读出的标签**再查一次（`rrsi/loop.py:576` 附近）——防止提案者把控制流改动
贴上"未尝试组件"的标签蒙混过关。

## 修剪集 B_t 的移交

近期收益 `g_t(l) <= 0` 的组件进入修剪集 `B_t`（`rrsi/history.py:178`），连同它在现任
harness 里的"被接受机制"清单一起交给提案者（`rrsi/propose.py:296` 附近的
`COMPONENTS TO PRUNE` 段）。拆除停止挣钱的机器本身就是一条合法编辑
（component=被剪组件，hypothesis="prune: ..."）——搜索于是既能长也能缩。
