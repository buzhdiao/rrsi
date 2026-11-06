# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Google LLC 的 Apache-2.0 许可证头（仓库标准文件头，保留原文）。
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coding instance: Terminal-Bench 2.1 through harbor with a Terminus-2 fork.

Evaluate = `scripts/run_eval.sh <job> <dataset> <k> <concurrency> [-i task ...]`
run from the candidate's worktree (PYTHONPATH = that worktree's third_party/,
so `--agent harbor_terminus2:AgentHarness` resolves to the candidate's harness). One
harbor job = one evaluation; trial directories are `<task>__<hash>` under
runs/coding/jobs/<job>/. A trial's reward is 1.0 iff the task's hidden tests
pass; a missing or errored trial counts 0.0 with the full denominator.
"""

# 【模块说明（中文）】coding 域适配器：经 harbor + Terminus-2 fork 评估 Terminal-Bench 2.1。
# Evaluate = 在候选 worktree 里运行 `scripts/run_eval.sh <job> <dataset> <k> <并发> [-i 任务...]`
# （PYTHONPATH 指向该 worktree 的 third_party/，因此 --agent harbor_terminus2:AgentHarness
# 解析到候选自己的 harness）。一个 harbor job = 一次评估；trial 目录形如
# runs/coding/jobs/<job>/<task>__<hash>/。奖励定义：隐藏测试全部通过记 1.0；
# trial 缺失或报错按 0.0 计且保留完整分母（missing-as-zero，防止靠摧毁 trial 涨分）。

# 延迟类型注解求值
from __future__ import annotations

# json：解析 result.json 与 rrsi.json 配置
import json
# os：环境变量读写（RRSI_CODING_PYTHON、MODEL 等）与 os.environ 传递
import os
# re：基础设施故障特征、任务名泄漏黑名单的正则
import re
# subprocess：调用 run_eval.sh、compileall、rm 等外部进程
import subprocess
# sys：调整 sys.path 以导入 rrsi 包与同目录的 briefs/render
import sys
# time：给挪走的残缺作业目录加时间戳后缀
import time
# Path：路径对象
from pathlib import Path

# HERE：本文件所在目录 domains/coding/
HERE = Path(__file__).resolve().parent
# 把仓库根（domains/ 的上一级）加入 sys.path，以便导入 rrsi 包
sys.path.insert(0, str(HERE.parent.parent))
# 把本目录加入 sys.path，以便导入同级的 briefs/render 模块
sys.path.insert(0, str(HERE))
# Domain：域适配器接口（run/score/load_trial/render_trace/smoke/guards 等）
from rrsi.domain import Domain          # noqa: E402
# TaskResult：每任务的打分结果（rewards/tokens/missing/extra）
from rrsi.evaluate import TaskResult    # noqa: E402
# briefs：本域四个搜索角色（analyst/digester/proposer/critic）的提示词段落
import briefs                            # noqa: E402
# render：把 harbor trial 渲染成 analyst/digester/proposer 读的文本
import render                            # noqa: E402

# CFG：本域配置 rrsi.json（数据集名、smoke 任务、并发、策略模型、policy 标签等）
CFG = json.loads((HERE / "rrsi.json").read_text())
# PYBIN：评分 venv 的 python（smoke 里跑 compileall / 构造检查用），可用环境变量覆盖
PYBIN = os.environ.get("RRSI_CODING_PYTHON", str(HERE / ".venv" / "bin" / "python"))


# 定义 _load_result：读取一个 trial 目录的 result.json；不存在或 JSON 损坏时返回 None
def _load_result(trial_dir: Path) -> dict | None:
    # result.json：harbor 为每个 trial 写的结果文件（奖励、异常、token 用量等）
    rj = trial_dir / "result.json"
    # 文件不存在说明该 trial 没有产出结果（被杀/未完成），按缺失处理
    if not rj.exists():
        return None
    # 正常路径：解析并返回结果字典
    try:
        return json.loads(rj.read_text())
    # JSON 半截（进程被杀时写到一半）等价于没有结果，同样返回 None
    except json.JSONDecodeError:
        return None


# 定义 _short：从完整 task_name（如 "terminal-bench/xyz"）截取末段短任务名（作分组键）
def _short(result: dict) -> str:
    # 取 "/" 后的最后一段
    return str(result.get("task_name", "")).split("/")[-1]


# 定义 _reward：从 result.json 挖出 harbor verifier 给出的标量奖励；字段缺失返回 None
def _reward(result: dict) -> float | None:
    # verifier_result.rewards.reward 是 harbor 的奖励字段（1.0 通过 / 0.0 失败）
    val = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    # 有值就转 float；缺失（trial 异常/没跑完）返回 None，由调用方按 0 处理
    return float(val) if val is not None else None


# _INFRA_EXC：基础设施故障的正则特征（docker/compose/镜像拉取/磁盘满/连接拒绝/守护进程/网络不存在等）。
# 命中说明 trial 根本没拿到可用环境，属"缺失"而非"agent 失败"——不应算进 agent 的成绩
_INFRA_EXC = re.compile(r"docker|compose|image|pull|registry|no space left|connection refused|"
                        r"daemon|network .* (not found|already exists)", re.I)


# 定义 infra_failure：判断 trial 是否死于基础设施（harbor/docker 报错），
# 而非 agent 超时或测试未通过——前者在计分时按"缺失"处理而不是 agent 的失败
def infra_failure(result: dict) -> bool:
    """True when the trial never got a working environment (harbor/docker error),
    as opposed to the agent timing out or the tests failing."""
    # docstring：trial 没拿到可用环境（harbor/docker 错误）时为 True；agent 超时或测试失败不算
    e = result.get("exception_info")
    # 没有异常信息 => 不存在基础设施故障
    if not e:
        return False
    # 异常信息可能是结构化 dict（类型+消息），也可能是纯字符串
    if isinstance(e, dict):
        typ, msg = str(e.get("exception_type", "")), str(e.get("exception_message", ""))
    else:
        typ, msg = "", str(e)
    # agent 自身超时 / verifier 超时是真实失败，不算基础设施问题
    if typ in ("AgentTimeoutError", "VerifierTimeoutError"):
        return False
    # 消息命中基础设施特征，或异常类型是 RuntimeError/OSError（docker/compose 常见抛法）即判为基础设施故障
    return bool(_INFRA_EXC.search(msg)) or typ in ("RuntimeError", "OSError")


# 定义 _tokens：该 trial 的 token 用量（输入+输出）；异常 trial 返回 None（数据不可信）
def _tokens(result: dict) -> int | None:
    # 带异常信息的 trial，其 token 统计不完整，直接放弃
    if result.get("exception_info"):
        return None
    # agent_result 里分别记录输入/输出 token 数（缺省按 0）
    ar = result.get("agent_result") or {}
    tok = (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)
    # 0 也视为"无数据"（返回 None），避免把真 0 与缺失混为一谈
    return tok or None


# CodingDomain：coding 域适配器——把 Terminal-Bench 2.1（经 harbor + Terminus-2 fork）
# 接入 RRSI 的 Domain 接口（run/score/load_trial/render_trace/smoke 等）
class CodingDomain(Domain):
    # domain 名：分支 evolve/coding、runs/coding 等都由它派生
    name = "coding"
    # 可演化 harness 相对 domains/coding/ 的位置
    harness_path = "../../third_party/harbor_terminus2"
    # 提案者可新建/编辑的文件扩展名白名单
    source_exts = {".py", ".txt", ".md", ".json", ".sh"}
    # briefs：四个搜索角色的 domain 段落；{policy} 占位符替换成配置里的冻结策略标签
    briefs = {k: v.replace("{policy}", CFG.get("policy_label", "the frozen policy"))
              for k, v in {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
                           "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}.items()}
    # critic_patterns：确定性泄漏黑名单（critic.precheck 用，(正则, 理由) 对）——
    # diff 里出现 verifier 测试路径、奖励文件、CTRF 报告、官方 solution 等都意味着在读判分材料
    critic_patterns = [
        # diff 引用了 /tests 路径（verifier 隐藏测试所在）
        (r"/tests\b", "verifier tests path in diff"),
        # diff 引用了 test_outputs（verifier 测试文件）
        (r"test_outputs", "verifier test file reference in diff"),
        # diff 引用了奖励文件 reward.txt / reward.json
        (r"\breward\.(txt|json)\b", "reward file reference in diff"),
        # diff 引用了 CTRF 测试报告（verifier 的机器可读结果格式）
        (r"\bctrf\b", "verifier report reference in diff"),
        # diff 引用了官方 solution 路径（直读答案即作弊）
        (r"/solution\b", "oracle solution path in diff"),
        # diff 引用了 task.toml（任务配置，含测试入口信息）
        (r"task\.toml", "task config reference in diff"),
        # diff 引用了 verifier 日志路径
        (r"/logs/verifier", "verifier logs path in diff"),
    ]
    # component_signals：diff 正则 -> 组件标签的 domain 专属信号（供 rrsi.components 给编辑分类）。
    # 每项 (组件名, [正则...])：diff 命中哪组特征模式，就把这次编辑归到哪个 harness 组件
    component_signals = [
        # memory：跨任务持久状态（TBMH_STATE_DIR 环境变量）
        ("memory",          [r"TBMH_STATE_DIR"]),
        # subagent：子代理调用（generate_response）
        ("subagent",        [r"generate_response"]),
        # context_mgmt：上下文管理（摘要、unwind、free_tokens 等）
        ("context_mgmt",    [r"_summarize", r"summariz", r"proactive_summarization",
                             r"unwind", r"free_tokens"]),
        # output_plumbing：输出管道（终端输出截断/折叠、粘贴缓冲）
        ("output_plumbing", [r"_limit_output_length", r"terminal_output", r"truncat",
                             r"_collapse", r"paste.?buffer", r"load-?buffer"]),
        # control_flow：主循环控制（agent loop、episode、完成判定、按键注入）
        ("control_flow",    [r"_run_agent_loop", r"episode", r"is_task_complete",
                             r"_pending_completion", r"keystrokes"]),
        # config：数值配置（阈值、max_bytes、时长等魔法数）
        ("config",          [r"= *\d+ *#", r"threshold", r"max_bytes", r"duration"]),
        # prompt：提示词工程（模板文件、验证提示、hint 生成）
        ("prompt",          [r"templates/.*\.txt", r"_VERIFICATION", r"_get_.*hint",
                             r"prompt", r"system"]),
    ]

    # 定义 __init__：加载任务清单，并把"基准任务名"追加进 critic 泄漏黑名单
    def __init__(self):
        # tb21_tasks.txt：本实例的 Terminal-Bench 2.1 任务名清单（每行一个，去空行）
        self._tasks = [x.strip() for x in (HERE / "data" / "tb21_tasks.txt")
                       .read_text().splitlines() if x.strip()]
        # 把每个任务名编译成词边界正则（前后不能是字母数字/下划线/连字符）追加进黑名单——
        # diff 里出现基准任务名即视为记忆了具体任务（task knowledge），必拒
        self.critic_patterns = list(self.critic_patterns) + [
            (rf"(?<![\w-]){re.escape(n)}(?![\w-])", f"benchmark task name '{n}' in diff")
            for n in self._tasks]

    # ---- task sets ---------------------------------------------------------
    # —— 任务集合划分 ——
    # evolve 集：搜索全程使用的任务 id（这里=全部 TB2.1 任务）
    def evolve_ids(self) -> list[str]:
        return list(self._tasks)

    # 同分布 held-out 集：搜索从不接触；本域没有
    def heldout_ids(self) -> list[str]:
        return []          # SWE-bench Verified is run as a separate harbor dataset
        # held-out 为空：SWE-bench Verified 作为独立的 harbor 数据集另行评估（最终裁判面）

    # smoke 活性检查用的小任务集
    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        # 固定取 rrsi.json 指定的 smoke 任务（不按现任分数挑）
        return list(CFG["smoke_tasks"])

    # ---- Evaluate ----------------------------------------------------------
    # —— Evaluate(H', D, k)：调用 harbor 执行评估 ——
    # 定义 _harbor：在 worktree root 下启动一个 harbor 作业（job），跑 dataset 上
    # 的 ids（None=全部任务）各 k 个 trial，日志追加到 runs_dir/logs/；支持续跑语义
    def _harbor(self, root: Path, runs_dir: Path, job: str, ids: list[str] | None,
                k: int, dataset: str, log_prefix: str) -> None:
        # jobs 根目录：runs/coding/jobs/
        jobs = runs_dir / "jobs"
        # 不存在则创建（含父目录）
        jobs.mkdir(parents=True, exist_ok=True)
        # 本作业的目录
        jdir = jobs / job
        # 作业目录已存在：检查完整性，决定跳过还是挪走重跑
        if jdir.exists():
            # 统计已有的、非基础设施故障的 trial 数（infra 故障不算有效产出）
            n_have = sum(1 for rj in jdir.glob("*/result.json")
                         if not infra_failure(_load_result(rj.parent) or {}))
            # 期望 trial 总数 = 任务数 x k
            n_want = len(ids if ids is not None else self._tasks) * k
            # 已够数：视为完成，不重跑（resume-safe 的"跳过"分支）
            if n_have >= n_want:
                print(f"[coding] job {job} already complete ({n_have} trials); not re-running",
                      flush=True)
                return
            # harbor cannot resume a job; a partial dir (killed run, or an earlier
            # candidate under the same name) would be scored as mostly-missing.
            # harbor 不支持断点续跑：残缺目录（被杀的运行，或同名旧 candidate）会被打成大量缺失
            stale = jdir.with_name(f"{job}.stale.{int(time.time())}")
            # 原地改名挪走旧目录，腾出位置从头重跑
            jdir.rename(stale)
            print(f"[coding] job {job} was incomplete ({n_have}/{n_want} trials); moved to "
                  f"{stale.name} and re-running from scratch", flush=True)
        # run_eval.sh：harbor 评估入口脚本（在候选 worktree 内执行）
        script = root / "domains" / "coding" / "scripts" / "run_eval.sh"
        # 组装命令：<script> <job> <dataset> <k> <并发> --jobs-dir <jobs 目录>
        cmd = [str(script), job, dataset, str(k), str(CFG.get("concurrency", 10)),
               "--jobs-dir", str(jobs)]
        # 只跑部分任务时逐个用 -i 指定（terminal-bench/<task> 形式）；全集则不传（更快）
        if ids is not None and set(ids) != set(self._tasks):
            for t in ids:
                cmd += ["-i", f"terminal-bench/{t}"]
        # 环境变量：RRSI_CODING_ROOT（domain 根）、RRSI_CODING_VENV（评分 venv）、
        # MODEL（冻结策略模型，默认 vertex_ai/gemini-3.5-flash）
        env = {**os.environ, "RRSI_CODING_ROOT": str(root / "domains" / "coding"),
               "RRSI_CODING_VENV": str((HERE / ".venv").resolve()),
               "MODEL": CFG.get("policy_model", "vertex_ai/gemini-3.5-flash")}
        # 日志路径：runs/coding/logs/<前缀或 job 名>.log（追加写，保留历史）
        log = runs_dir / "logs" / f"{log_prefix or job}.log"
        # 确保日志目录存在
        log.parent.mkdir(parents=True, exist_ok=True)
        # 追加模式打开日志，stdout+stderr 一并重定向
        with open(log, "a") as lf:
            # 阻塞运行 harbor（cwd=domain 目录，保证相对路径解析一致）
            r = subprocess.run(cmd, cwd=str(root / "domains" / "coding"),
                               stdout=lf, stderr=subprocess.STDOUT, env=env)
        # 非 0 退出码只告警不抛——打分阶段会把缺失 trial 记 0（完整分母），不在这里中断搜索
        if r.returncode != 0:
            print(f"[coding] WARNING harbor rc={r.returncode} (see {log})", flush=True)

    # 定义 run：Domain.run 的实现——用配置里的数据集名调用 _harbor
    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        self._harbor(Path(root), Path(runs_dir), job, ids, k, CFG["dataset"], log_prefix)

    # 定义 _trials：扫描作业目录，把每个 trial 目录按（短）任务名分组
    def _trials(self, runs_dir: Path, job: str) -> dict[str, list[Path]]:
        # 结果容器：task 名 -> trial 目录列表
        by_task: dict[str, list[Path]] = {}
        # 作业目录路径
        jdir = Path(runs_dir) / "jobs" / job
        # 作业不存在（还没跑过）：返回空映射
        if not jdir.exists():
            return by_task
        # 遍历作业下排序后的每个条目
        for td in sorted(jdir.iterdir()):
            # 只关心目录（trial 目录形如 <task>__<hash>）
            if not td.is_dir():
                continue
            # 读不出 result.json 的 trial（未完成/损坏）跳过
            res = _load_result(td)
            if res is None:
                continue
            # 按短任务名归组追加
            by_task.setdefault(_short(res), []).append(td)
        # 返回 task -> [trial 目录] 映射
        return by_task

    # 定义 score：把 harbor 产物折算成每任务 TaskResult 与全局聚合。
    # 奖励=隐藏测试通过与否（1.0/0.0）；缺失或基础设施故障的 trial 记 0 且留在分母里（完整分母）
    def score(self, runs_dir, job, ids, k):
        # 取该作业全部 trial 目录（按任务分组）
        trials = self._trials(Path(runs_dir), job)
        # 每任务结果
        per = {}
        # 全局通过 trial 计数
        total_pass = 0
        # 逐任务统计
        for t in ids:
            # 该任务的前 k 个 trial 目录（多出的忽略，分母恰为 k）
            dirs = trials.get(t, [])[:k]
            # 奖励列表与 token 列表
            rewards, toks = [], []
            # 逐 trial 读结果
            for td in dirs:
                # 读 result.json（读不到按空 dict）
                res = _load_result(td) or {}
                # 基础设施故障的 trial 不计入 rewards（下面按缺失补 0）
                if infra_failure(res):
                    continue                  # counted as missing below
                # verifier 奖励（字段缺失按 0）
                rw = _reward(res)
                rewards.append(rw if rw is not None else 0.0)
                # token 用量（异常 trial 为 None）
                toks.append(_tokens(res))
            # 缺失 trial 数 = k - 有效奖励数
            missing = k - len(rewards)
            # 缺失位补 0 奖励（missing-as-zero：完整分母）
            rewards += [0.0] * missing
            # 缺失位 token 记 None
            toks += [None] * missing
            # 本任务通过数（奖励恰为 1.0 的 trial 数）
            passes = sum(1 for x in rewards if x == 1.0)
            # 累加进全局通过数
            total_pass += passes
            # 记录每任务结果：rewards/tokens/missing，extra 里附 passes 与 trial 目录路径
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing,
                                extra={"passes": passes,
                                       "trial_dirs": [str(d) for d in dirs]})
        # 返回每任务结果与聚合：总通过数、总 trial 数、通过率
        return per, {"total_passes": total_pass, "n_trials": len(ids) * k,
                     "pass_rate": total_pass / max(1, len(ids) * k)}

    # ---- evidence ----------------------------------------------------------
    # —— 证据读取（供 analyst/digester/proposer）——
    # 定义 load_trial：取指定 trial 的轻量摘要记录（轨迹渲染的入口）
    def load_trial(self, runs_dir, job, task_id, trial):
        # 该任务的 trial 目录列表
        dirs = self._trials(Path(runs_dir), job).get(task_id, [])
        # 序号越界：没有这条 trial
        if trial >= len(dirs):
            return None
        # 读 result.json（读不到按空 dict）
        res = _load_result(dirs[trial]) or {}
        # agent 元数据（episode 数、摘要次数等）
        meta = (res.get("agent_result") or {}).get("metadata") or {}
        # 返回摘要：任务 id、trial 目录、奖励、episode 数、异常信息前 200 字符
        return {"task_id": task_id, "trial_dir": str(dirs[trial]),
                "reward": _reward(res), "episodes": meta.get("n_episodes"),
                "exception": (str(res.get("exception_info"))[:200]
                              if res.get("exception_info") else None)}

    # 定义 render_trace：把一条 trial 记录渲染成可读文本（委托给 render 模块）
    def render_trace(self, rec, detail=False):
        # detail=True 时放开各段截断上限
        return render.render_full(rec, detail=detail)

    # 定义 task_row：任务表里该轨迹的一行摘要（奖励、episode、异常）
    def task_row(self, task_id, rec, tr):
        # 拼一行：任务 id | 各 trial 奖励 | 渲染 trial 的奖励 | episode 数 | 异常前 60 字符
        return (f"{task_id} | rewards={tr.rewards if tr else '?'} | "
                f"rendered_trial_reward={rec.get('reward')} | "
                f"episodes={rec.get('episodes')} | exception={str(rec.get('exception'))[:60]}")

    # ---- gates -------------------------------------------------------------
    # —— 闸门 ——
    # 定义 smoke：评估前的活性检查（不是选择规则）：编译 harness -> 构造 AgentHarness ->
    # 清空旧目录实跑 smoke 任务 -> 要求所有 trial 都有结果且无异常
    def smoke(self, root, runs_dir, job, ids):
        # 统一转 Path
        root, runs_dir = Path(root), Path(runs_dir)
        # domain 目录（worktree 内）
        cdir = root / "domains" / "coding"
        # 第 1 关：用评分 venv 的 python 对 harness 目录做 compileall 语法编译
        comp = subprocess.run([PYBIN, "-m", "compileall", "-q", str(self.harness_dir(root))],
                              capture_output=True, text=True)
        # 编译失败：直接判死，附编译输出尾段
        if comp.returncode != 0:
            return False, {"stage": "compile", "detail": (comp.stdout + comp.stderr)[-1500:]}
        # 第 2 关：在 PYTHONPATH 指向候选 third_party/ 的前提下构造 AgentHarness，
        # 并断言 harness 名是 'rrsi-terminus2'——确保装载的是候选自己的 harness 代码
        code = ("from pathlib import Path\nfrom harbor_terminus2 import AgentHarness\n"
                "a = AgentHarness(logs_dir=Path('/tmp/rrsi_ctor_check'), "
                f"model_name='{CFG.get('policy_model', 'vertex_ai/gemini-3.5-flash')}')\n"
                "assert AgentHarness.name() == 'rrsi-terminus2'\nprint('CTOR_OK')\n")
        # 执行构造检查脚本（cwd=domain 目录）
        ctor = subprocess.run([PYBIN, "-c", code], cwd=str(cdir), capture_output=True,
                              text=True, env={**os.environ, "PYTHONPATH": str(root / "third_party")})
        # 没看到 CTOR_OK：构造失败，附 stderr/stdout 尾段
        if "CTOR_OK" not in ctor.stdout:
            return False, {"stage": "ctor", "detail": (ctor.stderr or ctor.stdout)[-1200:]}
        # 第 3 关：删掉同名作业目录后实跑一遍 smoke 任务（每任务 1 个 trial）
        jdir = runs_dir / "jobs" / job
        # 递归删除旧目录（若在），保证 smoke 从干净状态开始
        subprocess.run(["rm", "-rf", str(jdir)])
        # 跑 harbor：ids 里每个任务各 1 个 trial
        self._harbor(root, runs_dir, job, ids, 1, CFG["dataset"], job)
        # found：有结果的 trial 数；details：异常/缺失详情
        found, details = 0, {}
        # 遍历该作业下所有 trial 目录
        for td in (jdir.glob("*") if jdir.exists() else []):
            # 读 result.json
            res = _load_result(td)
            # 没有 result 的 trial 不计入 found（后面按缺失报告）
            if res is None:
                continue
            # 有结果，计数
            found += 1
            # trial 带异常：记入 details（截 400 字符）
            if res.get("exception_info"):
                details[td.name] = str(res["exception_info"])[:400]
        # 找到的 trial 数不足：补一条缺失说明
        if found < len(ids):
            details["missing_trials"] = f"{found}/{len(ids)}"
        # 任何异常/缺失：smoke 不通过
        if details:
            return False, {"stage": "smoke_run", **details}
        # 全部通过：返回成功与 trial 数
        return True, {"stage": "smoke_run", "n": found}


# 模块级单例：load_domain 约定从 adapter 模块读取 DOMAIN
DOMAIN = CodingDomain()
