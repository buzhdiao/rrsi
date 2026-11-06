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
# 模块说明:agentic-workspace 实例的 Domain 适配器——通过内置(vendored)官方
# react_toolbelt 引擎运行 Harvey LAB。Evaluate = k 次生产 pass(用候选 worktree 的
# produce.sh + workspace_driver.py 产出),每次 pass 后跟基准自带的量规评审器
# (workspace_judge.py)。任务 J 的样本 s 位于 runs/workspace/jobs/J/s<s>/<任务>/ 下,
# 含 trajectory.json、output/(交付物)与 scores.json(逐判据判定)。一次 trial 的
# 奖励 = 通过判据数/判据总数,按判据总数加权,因此 S_hat 就是基准的判据通过率;
# 缺失的 trial 记为其任务所有判据失败。
"""Agentic-workspace instance: Harvey LAB through the vendored official
react_toolbelt engine (engine/runner/agents/react_toolbelt_agent -> harness/).

Evaluate = k production passes (produce.sh + workspace_driver.py from the candidate's
worktree) each followed by the benchmark's own rubric judge (workspace_judge.py).
Sample s of job J lives under runs/workspace/jobs/J/s<s>/<task>/ with
trajectory.json, output/ (the deliverables) and scores.json (per-criterion
verdicts). A trial's reward is criteria passed / criteria total, weighted by
criteria total, so S_hat is the benchmark's criterion pass rate; a missing
trial fails every criterion of its task.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 定位本目录与仓库根,并把两者加进 sys.path 以导入 rrsi 包与同目录模块
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
import briefs                            # noqa: E402
import render                            # noqa: E402

# 读取本域配置(rrsi.json)
CFG = json.loads((HERE / "rrsi.json").read_text())
# Harvey LAB 检出根:优先环境变量,其次配置,最后相对路径缺省
HARVEY_LAB_ROOT = Path(os.environ.get("HARVEY_LAB_ROOT", CFG.get("harvey_lab_root") or "harvey-labs"))
# 进化/保留集划分文件路径(首次使用时确定性生成并缓存)
SPLIT_PATH = HERE / "data" / "split_workspace.json"
# 划分缓存,避免重复读盘/生成
_SPLIT: dict | None = None


def split() -> dict:
    """The evolve / held-out split. Not distributed with the repository: it is
    generated deterministically from the Harvey LAB checkout on first use (see
    split_workspace.py) and cached at SPLIT_PATH."""
    global _SPLIT
    if _SPLIT is None:
        # 划分文件不存在则现场确定性生成(进化集/保留集划分,论文可复现)
        if not SPLIT_PATH.exists():
            import split_workspace                       # noqa: WPS433
            split_workspace.write_split(str(SPLIT_PATH), str(HARVEY_LAB_ROOT))
        _SPLIT = json.loads(SPLIT_PATH.read_text())
    return _SPLIT


# 运行 Harvey LAB 评审的 python(其 venv);代理用的 python(默认当前解释器)
HARVEY_PY = Path(os.environ.get("HARVEY_PY", str(HARVEY_LAB_ROOT / ".venv" / "bin" / "python")))
AGENT_PY = os.environ.get("RRSI_AGENT_PYTHON", sys.executable)


def _name(task_id: str) -> str:
    # 任务 id 中的 '/' 压平成 '__',用作目录名
    return task_id.replace("/", "__")


def _task_json(task_id: str) -> dict:
    # 读取任务的 task.json(量规载体);读取失败返回空 dict
    p = HARVEY_LAB_ROOT / "tasks" / task_id / "task.json"
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


class WorkspaceDomain(Domain):
    # workspace 域的 Domain 实现:任务集、Evaluate、证据与门禁
    name = "workspace"
    # 被进化/审查的 harness(脚手架)所在路径(相对仓库根)
    harness_path = "../../third_party/archipelago/harness_workspace"
    # 四个搜索角色(分析师/消化器/提议者/批评家)的领域简报
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    # 越界审查模式:diff 中出现这些内容即视为泄露/作弊(task id、world id、
    # 评审器引用、task.json)
    critic_patterns = [
        (r"task_[0-9a-f]{32}", "task id in diff"),
        (r"world_[0-9a-f]{32}", "world id in diff"),
        (r"llm_judge|score_rubric|criteria_results", "judge reference in diff"),
        (r"task\.json", "task.json (rubric carrier) referenced in diff"),
    ]
    # 组件信号:harness 中的代码特征 -> 组件归类(用于批评家定位结构杠杆)
    component_signals = [
        ("memory",          [r"DRMH_STATE_DIR", r"digest\("]),
        ("context_mgmt",    [r"resum\.py", r"ReSum", r"summariz", r"compact",
                             r"_should_compress", r"keep_last"]),
        ("output_plumbing", [r"tool_result\.py", r"truncat", r"HEAD_CHARS", r"TAIL_CHARS",
                             r"MAX_RESULT_TOKENS", r"_process_result"]),
        ("control_flow",    [r"_handle_tool_calls", r"def step", r"final_answer",
                             r"has_incomplete_todos", r"max_steps", r"_execute_mcp_tool"]),
        ("config",          [r"= *\d+ *#", r"threshold", r"_CAP\b", r"_LIMIT\b"]),
        ("prompt",          [r"SYSTEM_PROMPT", r"WRAPPER_TEMPLATE", r"prompt", r"description"]),
    ]

    def __init__(self):
        # 每任务判据数缓存,避免反复读 task.json
        self._criteria_count: dict[str, int] = {}

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        # 进化集任务 id 列表
        return list(split()["tasks"]["evolve"])

    def heldout_ids(self) -> list[str]:
        # 保留集(held-out)任务 id 列表
        return list(split()["tasks"]["heldout"])

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        # 冒烟(smoke)测试用的少量任务
        return list(split()["smoke"])

    def regression_threshold(self, k: int) -> float:
        # 回归门禁阈值:进化后均分下降超过 5% 判为回归
        return 0.05

    def _n_criteria(self, task_id: str) -> int:
        # 查询任务的判据数(带缓存)
        if task_id not in self._criteria_count:
            self._criteria_count[task_id] = len(_task_json(task_id).get("criteria") or [])
        return self._criteria_count[task_id]

    # ---- Evaluate ----------------------------------------------------------
    def _env(self, runs_dir: Path, job: str) -> dict:
        # 为子进程构造环境变量:各目录、网关端口(按 job 哈希错开避免冲突)、模型
        port = CFG["gateway_port"] + (abs(hash(job)) % 40)
        return {**os.environ,
                "RRSI_JOBS_DIR": str(runs_dir / "jobs"),
                "RRSI_HARNESS_MODULE": "harness_workspace.main",
                "RRSI_WORKSPACES": str(runs_dir / "workspaces"),
                "RRSI_LOGS": str(runs_dir / "logs"),
                "GATEWAY_PORT": str(port), "HARVEY_LAB_ROOT": str(HARVEY_LAB_ROOT),
                "VENV": AGENT_PY, "MAX_CONCURRENT": str(CFG["max_concurrent"]),
                "ORCHESTRATOR_MODEL": os.environ.get("ORCHESTRATOR_MODEL",
                                                     "vertex_ai/claude-opus-4-8")}

    def _produce(self, root: Path, runs_dir: Path, label: str, ids: list[str]) -> int:
        # 运行一次生产 pass:把任务 id 清单写入文件,调用候选 worktree 的 produce.sh
        wdir = root / "domains" / "workspace"
        jobs = runs_dir / "jobs"
        (jobs / label).mkdir(parents=True, exist_ok=True)
        tf = jobs / label / "_tasks.json"
        tf.write_text(json.dumps(ids))
        # 产出日志文件
        log = runs_dir / "logs" / f"{label.replace('/', '_')}.produce.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = self._env(runs_dir, label.split("/")[0])
        # 最多重试 3 次:统计已产出轨迹的任务数,全部完成才提前返回
        for attempt in range(3):
            with open(log, "a") as lf:
                subprocess.run(["bash", str(wdir / "produce.sh"), label, str(tf)],
                               cwd=str(wdir), env=env, stdout=lf, stderr=subprocess.STDOUT)
            done = sum(1 for t in ids if (jobs / label / _name(t) / "trajectory.json").exists())
            if done >= len(ids):
                return done
            # 未全部完成则打印进度并重试
            print(f"[workspace] produce {label}: {done}/{len(ids)} after attempt {attempt + 1}; "
                  f"retrying", flush=True)
        return done

    def _judge(self, root: Path, runs_dir: Path, label: str) -> None:
        # 评审一轮产出:调用 Harvey LAB venv 下的 workspace_judge.py 打量规分
        wdir = root / "domains" / "workspace"
        log = runs_dir / "logs" / f"{label.replace('/', '_')}.judge.log"
        env = {**os.environ, "GOOGLE_GENAI_USE_VERTEXAI": "true",
               "GOOGLE_CLOUD_PROJECT": os.environ.get("VERTEX_PROJECT", ""),
               "GOOGLE_CLOUD_LOCATION": "global", "HARVEY_LAB_ROOT": str(HARVEY_LAB_ROOT),
               "RRSI_JOBS_DIR": str(runs_dir / "jobs")}
        # 最多重试 3 次:成功标志是返回码 0 且 SUMMARY.json 已写出
        for attempt in range(3):
            with open(log, "a") as lf:
                r = subprocess.run([str(HARVEY_PY), str(wdir / "workspace_judge.py"), label,
                                    "--judge-model", CFG["judge_model"]],
                                   cwd=str(wdir), env=env, stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode == 0 and (runs_dir / "jobs" / label / "SUMMARY.json").exists():
                return
            # 失败则打印重试信息
            print(f"[workspace] judge {label}: retry {attempt + 1}", flush=True)

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        # Evaluate 主流程:对 k 个样本依次产出+评审,label 形如 job/s0
        root, runs_dir = Path(root), Path(runs_dir)
        for s in range(k):
            label = f"{job}/s{s}"
            self._produce(root, runs_dir, label, ids)
            self._judge(root, runs_dir, label)

    def score(self, runs_dir, job, ids, k):
        # 汇总打分:读每个 trial 的 scores.json,组装 TaskResult 与全局统计
        runs_dir = Path(runs_dir)
        per = {}
        crit_pass = crit_tot = all_pass = 0
        for t in ids:
            rewards, weights, toks, missing = [], [], [], 0
            for s in range(k):
                # trial 目录与其 scores.json
                d = runs_dir / "jobs" / job / f"s{s}" / _name(t)
                sp = d / "scores.json"
                if sp.exists():
                    # 有评分:奖励 = 通过判据数/判据总数,权重 = 判据总数
                    j = json.loads(sp.read_text())
                    total = max(1, int(j.get("total") or 0))
                    rewards.append(j.get("passed", 0) / total)
                    weights.append(float(total))
                    crit_pass += j.get("passed", 0)
                    crit_tot += total
                    all_pass += int(j.get("all_pass") or 0)
                else:
                    # 缺失 trial:按判据数加权记 0 分(惩罚缺失)
                    missing += 1
                    n = max(1, self._n_criteria(t))
                    rewards.append(0.0)
                    weights.append(float(n))
                    crit_tot += n
                # 尽力读取该 trial 的 token 消耗
                tok = None
                tp = d / "trajectory.json"
                if tp.exists():
                    try:
                        tok = (json.loads(tp.read_text()).get("usage") or {}).get("total_tokens")
                    except Exception:  # noqa: BLE001
                        tok = None
                toks.append(tok or None)
            # 组装每任务的 TaskResult
            per[t] = TaskResult(rewards=rewards, weights=weights, tokens=toks, missing=missing)
        # 全局统计:判据通过数/总数、全过任务数、判据通过率(criterion pass rate)
        return per, {"criteria_passed": crit_pass, "criteria_total": crit_tot,
                     "all_pass_tasks": all_pass,
                     "criterion_rate": crit_pass / max(1, crit_tot)}

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        # 加载单条 trial 证据:轨迹、评分、产出文件清单;无轨迹返回 None
        d = Path(runs_dir) / "jobs" / job / f"s{trial}" / _name(task_id)
        tp, sp = d / "trajectory.json", d / "scores.json"
        if not tp.exists():
            return None
        try:
            traj = json.loads(tp.read_text())
        except Exception:  # noqa: BLE001
            return None
        scores = json.loads(sp.read_text()) if sp.exists() else {}
        # 收集 output/ 下的交付物文件名
        outs = sorted(p.name for p in (d / "output").glob("*")) if (d / "output").is_dir() else []
        return {"task_id": task_id, "domain": task_id.split("/")[0], "trial": trial,
                "trajectory": traj, "scores": scores, "output_files": outs}

    def render_trace(self, rec, detail=False):
        # 把 trial 记录渲染成文本(交给 render.render_full)
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        # 摘要行:任务、判据通过情况、试验均值、状态、步数、交付物、失败判据索引
        sc = rec.get("scores") or {}
        fails = [i for i, c in enumerate(sc.get("criteria") or []) if c.get("verdict") != "pass"]
        traj = rec.get("trajectory") or {}
        steps = sum(1 for m in traj.get("messages", []) if m.get("role") == "assistant")
        return (f"{task_id} | {rec.get('domain')} | criteria {sc.get('passed')}/{sc.get('total')} "
                f"(mean over trials {tr.mean:.3f}) | status={traj.get('status')} | "
                f"steps={steps} | deliverables={rec.get('output_files')} | "
                f"failed_criteria={fails[:25]}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        # 冒烟门禁:先编译检查 harness,再真实跑一轮 s0,全部任务出轨迹才算过
        root, runs_dir = Path(root), Path(runs_dir)
        # 第一步:对 harness 目录做 compileall 语法检查
        comp = subprocess.run([AGENT_PY, "-m", "compileall", "-q", str(self.harness_dir(root))],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "detail": (comp.stdout + comp.stderr)[-1500:]}
        # 第二步:清掉旧 s0 目录后跑一轮生产 pass
        label = f"{job}/s0"
        shutil.rmtree(runs_dir / "jobs" / label, ignore_errors=True)
        done = self._produce(root, runs_dir, label, ids)
        if done < len(ids):
            return False, {"stage": "smoke_run", "detail": f"{done}/{len(ids)} trajectories"}
        return True, {"stage": "smoke_run", "n": done}


# 模块级单例:框架按 name 发现本域
DOMAIN = WorkspaceDomain()
