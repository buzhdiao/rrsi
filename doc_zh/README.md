# RRSI 中文学习文档（doc_zh）

本目录是 [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses]
(https://arxiv.org/abs/2609.24972)（Google Research）的中文学习文档，配合仓库源码阅读。
所有 `file:line` 引用指向**当前工作区已加中文注释**的代码文件，行号可直接跳转。

## 目录结构

```
doc_zh/
├── README.md                    # 本索引
├── components/                  # 组件原理 8 篇
│   ├── 01-overview-and-search-loop.md        # 总体架构与搜索循环
│   ├── 02-proposal-side-regularization.md    # 提案侧正则化（b_t / L_t 条件化 / 卡死重定向）
│   ├── 03-selection-side-regularization.md   # 选择侧正则化（critic / 地板 / 成本规则 / 修剪）
│   ├── 04-git-worktree-evaluation.md         # git worktree 候选评估机制
│   ├── 05-domain-adapter-pattern.md          # Domain 适配器模式（三实例）
│   ├── 06-edit-history-jsonl.md              # 编辑历史 JSONL 与可审计性
│   ├── 07-noise-band-calibration.md          # 噪声带校准（bootstrap）
│   └── 08-reading-the-results.md             # 结果解读（evolve / ID held-out / OOD）
├── translations/
│   └── README.zh.md             # 根 README.md 的完整中文翻译
└── techniques-100.md            # Python 高级技法与重要库 100 讲
```

## 阅读路线建议

- **想跑起来**：先读翻译版 `translations/README.zh.md`（安装、LLM 配置、三条命令），
  再读第 01 篇对照 `rrsi.py` 的子命令。
- **想懂方法**：第 01 → 02 → 03 篇是主线（论文 Algorithm 1 / Algorithm 2 的代码化），
  第 07 篇补噪声带的来龙去脉，第 08 篇教你怎么读结果表。
- **想改代码 / 加基准**：第 04 篇（git 机制）与第 05 篇（Domain 接口）必读，
  第 06 篇解释所有落盘文件的用途。
- **学 Python**：`techniques-100.md` 的 100 条技法全部取自本仓库真实代码，按类可查。

## 核心代码中文注释索引

`rrsi/` 包与入口脚本共 18 个 Python 文件均已逐行加中文注释（只插入注释行/行尾注释，
代码语义零改动，已用 token 流比对验证）：

| 文件 | 内容 | 论文概念锚点 |
|---|---|---|
| `rrsi.py` | CLI 入口与子命令分发 | — |
| `rrsi/loop.py` | 一轮主循环（七步） | Algorithm 1 + 2 |
| `rrsi/schedule.py` | 退火编辑预算 | annealed edit budget b_t（Eq. anneal） |
| `rrsi/history.py` | 编辑历史与汇总 | L_t / T_t / g_t / U_t / sigma_t / B_t |
| `rrsi/propose.py` | 正则化提案者 | P_reg、\|\|z_t\|\|_0 <= b_t、reserved slot |
| `rrsi/analyst.py` | 批量分析师 | Analyze(H_t, D) -> F_t |
| `rrsi/digester.py` | 只读轨迹摘要子代理 | 摘要透镜 failure / capability_gap / success |
| `rrsi/critic.py` | 泄漏筛查 | Critic(H_t, H')、bounded repair |
| `rrsi/evaluate.py` | 评估与聚合 | Ŝ/Ĉ（Eq. estimate）、缺失 trial 记 0 |
| `rrsi/selection.py` | 选择侧纯函数 | noise-adjusted floor、cost rule、within-band、argmax |
| `rrsi/components.py` | 组件词表与标签校验 | K / K_str / novelty ν(l') |
| `rrsi/domain.py` | Domain 接口 | guards（非补偿性判据） |
| `rrsi/calibrate.py` | 噪声带校准 | delta = z·sd(null ΔS)、bootstrap |
| `rrsi/config.py` | 超参数 | 论文附录 Hyperparameters 全部符号 |
| `rrsi/gitops.py` | git 底层操作 | worktree 隔离、fast-forward、tree hash |
| `rrsi/driver.py` | 顺序驱动器 | 断点续跑、基础设施熔断 |
| `rrsi/llm.py` | 搜索角色 LLM 客户端 | 项目轮转、prompt cache |
| `rrsi/__init__.py` | 包说明与模块地图 | — |

## 术语对照（速查）

| 英文（论文） | 中文 | 首次出现 |
|---|---|---|
| harness | （agent）脚手架 | 根 README Overview |
| annealed edit budget b_t | 退火编辑预算 | Eq. (anneal)，`rrsi/schedule.py:51` |
| edit history L_t | 编辑历史 | Eq. (history)，`rrsi/history.py` |
| tried set T_t / untried U_t | 已尝试/未尝试组件集 | `rrsi/history.py:139`、`exploration` |
| stall flag sigma_t | 停滞标志 | Eq. (explore)，`rrsi/history.py:220` |
| noise-adjusted floor | 噪声调整地板 | Eq. (floor)，`rrsi/selection.py:126` 附近 |
| cost rule | 成本规则 | Eq. (tokenbudget)，`rrsi/selection.py:99` |
| within-band rule | 噪声带内规则 | Alg. 2 第 5 行，`rrsi/selection.py:109` |
| prune set B_t | 修剪集 | Eq. (prune)，`rrsi/history.py:178` |
| novelty ν(l') | 新颖项 | `rrsi/components.py:129` |
| reserved exploration slot | 保留探索槽位 | `rrsi/loop.py:337` 附近 |
| bounded repair | 有界修复 | `rrsi/loop.py:551` 附近 |
| evolve set / ID held-out / OOD | 搜索集/同分布留出/分布外 | 第 08 篇 |
