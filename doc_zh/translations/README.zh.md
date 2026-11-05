# RRSI：正则化递归自改进的 Agent Harness 搜索（中文翻译）

> 本文件是仓库根 [README.md](../../README.md) 的完整中文翻译，供学习参考；
> 如有出入以英文原文为准。术语首次出现时附英文原文。

查看我们的[论文](https://arxiv.org/abs/2609.24972)与[项目主页](https://regularized-rsi.com/)了解更多细节。

## 🔥 更新

- [09/21/2026] 我们的[论文](https://arxiv.org/abs/2609.24972)已发布！欢迎阅读！[[项目主页]](https://regularized-rsi.com/)

## 🧬 概述（Overview）

<p align="center">
  <img src="./assets/rrsi_overview.png" width="92%" alt="RRSI overview: proposal-side and selection-side regularization of the harness search">
</p>

一个 LLM agent 的能力很大程度上由它的 **harness**（脚手架：围绕冻结模型的提示词、
控制流、工具、记忆与上下文管理）决定。针对固定的 evolve 集（evolve set）演化 harness
虽然有效，但会过拟合：harness 会记住训练任务，分布内（in-distribution）的大幅提升
到了分布外（out-of-distribution）会缩水甚至消失。RRSI 的做法是：保持 harness 的
**编辑空间开放**，转而对穿过这个空间的**搜索轨迹**做正则化。

在**提案侧**（proposal side）：退火预算（annealed budget）限制一个候选最多打包多少
个独立编辑；提案者（proposer）被完整的编辑历史条件化，被证伪的假设不会被原样重画；
停摆的运行会被重定向到从未动用过的组件。在**选择侧**（selection side）：critic 在
评估之前筛查每个候选是否包含套件专属逻辑；噪声调整地板（noise-adjusted floor）拦下
评估方差之内的"增益"；成本规则（cost rule）要求新增的推理 token 由实测增益买单；
不再起作用的组件会被修剪（prune）。

### ✨ 核心特性

* **开放的编辑空间，正则化的搜索。** 提示词、控制流、配置、上下文管理、工具、技能
  （skill）、记忆与子代理（sub-agent）都可以修改；约束作用于"搜索如何移动"，而不
  作用于"harness 里可以包含什么"。
* **一套方法，三个实例。** 同一个循环驱动终端 agent（Terminal-Bench 2.1）、文档工作
  agent（Harvey LAB）与工程设计 agent（EngDesign）；每个实例就是一个 `Domain`
  适配器加一份起始 harness。
* **候选住在 git worktree 里。** 每个候选 harness 都在自己的 worktree 中起草、筛查、
  评估，worktree 检出自 `evolve/<domain>` 分支；接受某个候选就把分支快进
  （fast-forward）过去，因此"现任"（incumbent）永远是一个提交。
* **可审计的证据。** 编辑历史按编辑记录组件、假设、实测分数与成本变化以及裁决；
  提案者、分析师与 critic 收到的提示词都是 `domains/<name>/` 下的普通文件。

## 🧩 论文到代码（Method to code）

| 论文 | 代码 |
|---|---|
| 经验分数与成本估计 | `rrsi/evaluate.py: aggregate`（按 trial 加权奖励；缺失 trial 记 0 且分母不减） |
| 退火编辑预算 b_t | `rrsi/schedule.py: edit_budget`，在提案者的 done() 中强制执行 |
| 编辑历史 L_t、已尝试集 T_t、近期收益 g_t | `rrsi/history.py: History`（每编辑一条 JSONL 记录；只有成为 H_{t+1} 的候选的编辑 a = 1） |
| 停滞标志、未尝试组件、探索指令 | `rrsi/history.py: stall_flag, exploration`；保留槽位在 `rrsi/propose.py` 中强制执行 |
| Analyze(H_t, D) | `rrsi/analyst.py` 派发 `rrsi/digester.py` |
| 带（组件, 假设, diff）标签的提案者 | `rrsi/propose.py`；标签由 `rrsi/components.py` 对照 diff 校验 |
| Critic（评估前的泄漏筛查） | `rrsi/critic.py`（domain 正则黑名单 + LLM 审查，有界修复） |
| 并行评估 | `rrsi/evaluate.py`，`Run.round` 线程池 |
| 噪声调整地板、成本规则、within-band 规则、argmax | `rrsi/selection.py` |
| 新颖性 nu_t（从未进入获胜编辑的结构组件类型） | `rrsi/components.py: novelty`，作用于 K_str = client_tool, skill, memory, subagent |
| 修剪集 B_t | `History.prune_set`，连同待拆除的"被接受机制"一并交给提案者 |
| 噪声带 delta | 每实例固定写在 `rrsi.json`（0.017 / 0.004 / 0.020）；`delta` 为 `null` 时由 `rrsi/calibrate.py` 重估（对基础评估的 trials 做 bootstrap，或重复基础评估） |
| 非补偿性 domain 判据 | `Domain.guards`（工程设计实例：有效率下降、无提交率上升） |

---

## ⚡️ 快速开始（Quickstart）

### 0. 安装

```bash
git clone https://github.com/google-research/rrsi.git && cd rrsi
pip install -e ".[dev]"            # 搜索核心（Python 3.10 或更新）
python3 -m pytest tests
```

各基准运行器住在自己的环境里：coding 实例用 harbor（`domains/coding/.venv`），
workspace 与 eng 实例用一个装了 `pip install -e ".[agentic]"` 的 Python 3.11 环境
（`RRSI_AGENT_PYTHON`）。

### 1. LLM 配置

提案者、分析师、critic 与冻结策略（frozen policy）都是 Vertex AI 上的 Claude Opus 4.8
（coding 实例的 `policy_model`，另两个实例的 `ORCHESTRATOR_MODEL`；任何 LiteLLM
模型字符串都可用）。Harvey LAB 的评审（judge）是 Gemini 3.5 Flash。

```bash
gcloud auth application-default login
export VERTEX_PROJECT="your-project-id" VERTEXAI_PROJECT="your-project-id"
export VERTEX_LOCATION=global VERTEXAI_LOCATION=global
export RRSI_VERTEX_PROJECTS="your-project-id"
```

### 2. 运行一个实例

每个实例的形状都一样：

```bash
python3 rrsi.py --domain <coding|workspace|eng> smoke     # 活性：编译、构造、跑几个任务
python3 rrsi.py --domain <name> baseline                   # Evaluate(H_0)，播种 runs/<name>/frontier.json
python3 rrsi.py --domain <name> run                        # 第 0..T-1 轮，可断点续跑；touch runs/<name>/STOP 可停止
python3 rrsi.py --domain <name> status
```

每一轮在各自的 git worktree 里起草两个候选、筛查它们、在完整 evolve 集上评估两个，
然后把 `evolve/<name>` 快进到获胜者。`runs/<name>/` 保存 frontier、编辑历史与原始
trial。超参数在 `domains/<name>/rrsi.json`，可在命令行覆盖（`--T`、`--k`、`--delta`、
`--beta1` 等）；`readjudicate --t <t>` 对已存储的一轮重新应用 Algorithm 2，
`reevaluate --t <t>` 在基础设施故障后重测某轮。

想运行哪个实例，请参阅其专门文档，了解环境、评估协议与分布外（OOD）运行：

- [`domains/coding`](../domains/coding/README.md)：Terminal-Bench 2.1，然后 SWE-bench Verified
- [`domains/workspace`](../domains/workspace/README.md)：Harvey LAB，然后 JobBench、GDPval 与 APEX-Agents
- [`domains/eng`](../domains/eng/README.md)：EngDesign，然后 EngDesign v1 与 Frontier-Eng

每家的极简版：

```bash
# coding：Docker + harbor
python3 -m venv domains/coding/.venv && domains/coding/.venv/bin/pip install "harbor>=0.18"
python3 rrsi.py --domain coding baseline && python3 rrsi.py --domain coding run
bash domains/coding/scripts/swe_eval.sh                       # H_0 与现任在 SWE-bench Verified 上的评估

# workspace：固定提交的 Harvey LAB 检出；首次使用时从它生成任务划分
git clone https://github.com/harveyai/harvey-labs.git && (cd harvey-labs && git checkout 1da4750 && uv sync)
export HARVEY_LAB_ROOT=$PWD/harvey-labs RRSI_AGENT_PYTHON=~/venvs/rrsi-agentic/bin/python
python3 rrsi.py --domain workspace baseline && python3 rrsi.py --domain workspace run
python3 rrsi.py --domain workspace heldout --label champ      # 40 道 held-out 任务；其余见 ood/run_{jobbench,gdpval,apex}.sh

# eng：verifier 布局的官方 EngDesign 任务、判分 venv、被囚禁的工具网关
git clone https://github.com/AGI4Engineering/EngDesign.git
python3 domains/eng/scripts/engdesign/build_engdesign_bench.py --engdesign-open EngDesign/EngDesign-Open --out domains/eng/engdesign_bench
python3 -m venv domains/eng/.venvs/engdesign && domains/eng/.venvs/engdesign/bin/pip install -r domains/eng/scripts/engdesign/requirements.txt
bash domains/eng/scripts/preflight.sh
python3 rrsi.py --domain eng baseline && python3 rrsi.py --domain eng run
bash domains/eng/scripts/final_eval.sh frontier               # Frontier-Eng，需要 Frontier-Engineering 检出
```

## 📊 结果（Results）

以下数字来自论文：每个实例的冻结策略都是 Claude Opus 4.8，所有数字都与同一时间窗内
未演化的 harness H_0 对比。"Evolve" 是 harness 被搜索所依据的划分；其余各行从未进入
选择。Terminal-Bench、SWE-bench、JobBench、GDPval、APEX-Agents 与 EngDesign 报告
通过率（pass rate），Harvey LAB 报告通过准则占比（rubric criteria passed），Frontier-Eng
报告奖牌分（Medal points）。

| 领域 | 基准 | 角色 | H_0 | RRSI | Δ |
|:---|:---|:---|:---:|:---:|:---:|
| 编程（Coding） | Terminal-Bench 2.1 | evolve | 74.2 | **80.2** | +6.0 |
| 编程（Coding） | SWE-bench Verified | OOD | 82.0 | **83.8** | +1.8 |
| 智能工作区（Agentic workspace） | Harvey LAB | evolve | 89.4 | **90.5** | +1.1 |
| 智能工作区（Agentic workspace） | Harvey LAB | ID held-out | 86.9 | **89.2** | +2.3 |
| 智能工作区（Agentic workspace） | JobBench | OOD | 36.0 | **40.7** | +4.7 |
| 智能工作区（Agentic workspace） | GDPval | OOD | 48.8 | **52.3** | +3.5 |
| 智能工作区（Agentic workspace） | APEX-Agents | OOD | 34.2 | **37.9** | +3.7 |
| 工程设计（Engineering design） | EngDesign | evolve | 50.0 | **54.9** | +4.9 |
| 工程设计（Engineering design） | Frontier-Eng | OOD | 17.7 | **22.0** | +4.3 |

搜索并不绑定单一策略家族：把冻结策略换成 Gemini 3.5 Flash 后，同一个 coding 实例在
Terminal-Bench 2.1 上从 64.6 提升到 78.7，在 SWE-bench Verified 上从 76.8 提升到 79.0。

## 🧱 添加一个 domain

一个 domain 就是一个模块 `domains/<name>/adapter.py`，导出 `DOMAIN`——
`rrsi.domain.Domain` 的一个实例，需要实现：

* `evolve_ids`、`heldout_ids`、`smoke_ids`：任务划分；
* `run(root, runs_dir, job, ids, k)` 与 `score(runs_dir, job, ids, k)`：运行检出于
  `root` 的 harness 并返回逐任务 trial 奖励（Evaluate）；
* `load_trial`、`render_trace`、`task_row`：分析师、摘要器与提案者读取的证据；
* `smoke`：候选在评估前的活性检查；
* `critic_patterns`、`component_signals`、`briefs`、`guards`：domain 的泄漏黑名单、
  diff 到组件的信号、各角色提示词与非补偿性接受判据；

外加 `harness_path`（可演化的目录）、`SKILL.md` 与 `PATTERNS.md`（提案者的宪法）
以及 `rrsi.json`（超参数）。核心自己从不读轨迹格式，也从不读基准目录。

## 🧪 测试

```bash
python3 -m pytest tests            # 或：python3 tests/test_core.py
```

## 🙏 致谢

起始 harness 是 [harbor](https://github.com/laude-institute/harbor) 的 Terminus-2
agent 与 [archipelago](https://github.com/Mercor-Intelligence/archipelago) 的
react_toolbelt agent 及其运行器。各实例在 [Terminal-Bench](https://github.com/harbor-framework/terminal-bench)、[SWE-bench Verified](https://github.com/SWE-bench/SWE-bench)、[Harvey LAB](https://github.com/harveyai/harvey-labs)、[JobBench](https://github.com/Job-Bench/job-bench-eval)、[GDPval](https://openai.com/index/gdpval/)、[APEX-Agents](https://www.mercor.com/apex/apex-agents-leaderboard/)、[EngDesign](https://github.com/AGI4Engineering/EngDesign) 与 [Frontier-Eng](https://github.com/Einsia/Frontier-Engineering) 上评估。

## 💬 引用（Citation）

```bibtex
@article{xia2026rrsi,
  title={RRSI: Regularized Recursive Self-Improvement of Agent Harnesses},
  author={Xia, Peng and Han, Rujun and Wang, Zifeng and Chen, Yanfei and Zhuang, Yufan and Lee, Yoonho and Huang, Chengsong and Yu, Han and CuiZhu, Zhongying and Ming, Yifei and Yao, Huaxiu and Gokturk, Burak and Pfister, Tomas and Lee, Chen-Yu},
  journal={arXiv preprint arXiv:2609.24972},
  year={2026}
}
```

## 参与贡献（Contributing）

参见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 许可证（License）

Apache 2.0；见 [LICENSE](../../LICENSE)。`third_party/` 下的第三方代码遵循其自身许可证。

## 免责声明（Disclaimer）

这不是 Google 官方支持的产品。本项目不参与 [Google 开源软件漏洞奖励计划](https://bughunters.google.com/open-source-security)。
