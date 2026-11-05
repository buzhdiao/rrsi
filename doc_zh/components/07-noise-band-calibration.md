# 07 · 噪声带校准：calibrate.py 的 bootstrap

## delta 是什么

噪声带 `delta` 界定**同一个 harness 两次独立评估的分数差** `|Ŝ(H) - Ŝ(H)|`
（`rrsi/calibrate.py:28-41` 模块说明）。所有选择判据都建立在它之上：noise-adjusted
floor 是 `S* - delta`，成本规则里"真实增益"的定义是 `ΔS > delta`，停滞标志
`sigma_t` 比较的也是 w 轮提升是否 `<= delta`。delta 定小了噪声会冒充进步，定大了
真进步会被淹没——它必须来自测量，不是拍脑袋。

## 两条估计路径

`calibrate()`（`rrsi/calibrate.py:103`）接受一次或多次"未改动基础 harness"的评估：

- **R >= 2 次独立评估**：零假设差被直接观测——各次分数的标准差乘 `sqrt(2)`
  （两个独立同方差随机变量之差的方差是单者的两倍，`rrsi/calibrate.py:111`）。
  两两绝对差也记录进结果（`max_abs_diff`）供审计。
- **只有 1 次评估**：`bootstrap_se`（`rrsi/calibrate.py:70`）在每个任务的 trials 内
  有放回重采样 2000 次，重采样 Ŝ 的标准差即标准误；两次独立评估之差的标准差 =
  `sqrt(2)·se`（`rrsi/calibrate.py:117` 附近的 `sd_boot`）。重采样尊重任务内权重
  （Harvey LAB 按准则数加权），且用固定种子（`seed=7`）保证同一输入永远得到同一
  delta——校准本身可复现。

细节：多次评估时先把 trials 拼接（`pooled`，`rrsi/calibrate.py:88`）再重采样，然后
把合并数据（k 更大）的 se 折算回单次评估的尺度——se 与 `sqrt(k)` 成反比，故乘
`sqrt(k_pooled / k_0)`。优先用直接观测，除非它退化（例如几次分数完全相同）。

## 最后一步

`delta = z · sd(null ΔS)`，`z` 默认 2.0（`delta_z`，`rrsi/config.py`），约 97.5% 分位：
未改动的 harness 约 97.5% 的概率能自己越过 `S* - delta` 地板，而候选的增益必须超出
噪声带才被成本规则当作真实。结果写进 `runs/<domain>/calibration.json`（含方法名、
各估计量、任务数、基线 S/C），`Run.delta()`（`rrsi/loop.py:133`）按"配置显式值 >
校准文件 > 报错"的优先级解析。

## 什么时候重新校准

把 `rrsi.json` 里的 `delta` 设为 `null` 即让运行时用校准值；`rrsi.py:76` 的
`calibrate --jobs base,base2` 支持用多个基线评估直接观测零假设差。改变 delta 之后
不必重跑搜索：`rrsi.py:107` 的 `readjudicate --t <t>` 会对已存储的测量重新应用
Algorithm 2（新地板/新成本规则），一轮也不重测。
