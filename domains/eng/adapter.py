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
"""Engineering-design instance: EngDesign-Open (61 tasks) through the vendored
official react_toolbelt engine and each task's frozen code verifier.

Evaluate = bench/run_tasks.py (k trials per task, resume-safe) from the
candidate's worktree, then bench/verify.py (frozen verifiers, deterministic).
Trial i of task X in job J is runs/eng/jobs/J/X/t<i>/{payload.py, traj.json,
meta.json, verdict.json}. The reward is the verifier's binary `passed` (the
paper's headline metric); combined_score, valid and no_payload are kept as
extras, and the last two feed the two domain guards (Sec. 3.3: a candidate
must satisfy several non-compensatory criteria).
"""
# 【模块说明（中文）】eng（工程设计）域适配器：EngDesign-Open 61 题，经内置的官方
# react_toolbelt 引擎 + 每题自己的冻结代码 verifier 评估。
# Evaluate 两步：bench/run_tasks.py（每题 k 个 trial，可续跑，跑在候选 worktree 里），
# 然后 bench/verify.py（冻结 verifier，确定性）。作业 J 中任务 X 的第 i 个 trial 落在
# runs/eng/jobs/J/X/t<i>/{payload.py, traj.json, meta.json, verdict.json}。
# 奖励 = verifier 的二元 `passed`（论文的主指标）；combined_score / valid / no_payload
# 存进 extra，其中后两者喂给两个 domain guard（论文 3.3 节：候选必须同时满足多个
# 非补偿性判据，违反即一票否决）。

# 延迟类型注解求值
from __future__ import annotations

# json：解析 rrsi.json 配置、split_engd.json 划分与 trial 的 meta/verdict
import json
# os：环境变量（各路径/端口覆盖）与 os.environ 传递
import os
# socket：探测 MCP gateway 端口是否已就绪
import socket
# subprocess：启动 gateway、运行 run_tasks.py / verify.py / compileall
import subprocess
# sys：sys.path 调整；verify.py 用 sys.executable 跑
import sys
# time：等待 gateway 启动的轮询间隔
import time
# Path：路径对象
from pathlib import Path

# HERE：本文件所在目录 domains/eng/
HERE = Path(__file__).resolve().parent
# 把仓库根加入 sys.path，以便导入 rrsi 包
sys.path.insert(0, str(HERE.parent.parent))
# 把本目录加入 sys.path，以便导入同级的 briefs/render
sys.path.insert(0, str(HERE))
# Domain：域适配器接口
from rrsi.domain import Domain          # noqa: E402
# TaskResult：每任务打分结果
from rrsi.evaluate import TaskResult    # noqa: E402
# briefs：本域四个搜索角色的提示词段落
import briefs                            # noqa: E402
# render：把 eng trial 渲染成可读文本（含失败分类）
import render                            # noqa: E402

# CFG：本域配置 rrsi.json（smoke_n、guard 上限、gateway 端口、agent_python 等）
CFG = json.loads((HERE / "rrsi.json").read_text())
# SPLIT：任务划分 split_engd.json（evolve 集等；本域无 held-out）
SPLIT = json.loads((HERE / "data" / "split_engd.json").read_text())
# AGENT_PY：跑 agent 的 python 解释器（环境变量 > 配置 > 当前解释器）
AGENT_PY = os.environ.get("RRSI_AGENT_PYTHON", CFG.get("agent_python") or sys.executable)
# BENCH_ROOT：基准树根（含每题冻结 verifier），默认本目录下 engdesign_bench/
BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", str(HERE / "engdesign_bench")))
# GRADING_PY：跑 verifier 的专用 venv python（engdesign 依赖隔离）
GRADING_PY = Path(os.environ.get("GRADING_PYTHON",
                                 str((HERE / ".venvs").resolve() / "engdesign" / "bin" / "python")))
# GATEWAY_PORT：MCP gateway（工具环境）端口，默认 8996
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", str(CFG.get("gateway_port", 8996))))


# 定义 _port_up：探测 127.0.0.1:port 是否有进程在监听（gateway 活着吗）
def _port_up(port: int) -> bool:
    # 新建一个 TCP socket
    s = socket.socket()
    # 连接超时 1 秒（快速失败）
    s.settimeout(1)
    # connect_ex 返回 0 表示连接成功（端口有人监听）；无论结果都关闭 socket
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


class EngDomain(Domain):
    # domain 名：分支 evolve/eng、runs/eng 等由此派生
    name = "eng"
    # 可演化 scaffold 相对 domains/eng/ 的位置
    harness_path = "../../third_party/archipelago/harness_eng"
    # briefs：四个搜索角色的 domain 段落（原样引用，无占位符替换）
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    # critic_patterns：确定性泄漏黑名单（(正则, 理由) 对，critic.precheck 用）——
    # 覆盖判分材料、基准树、路径逃逸、verifier 内部、基准名/任务 id、工具链、联网装包七类
    critic_patterns = [
        # 命中判分/参考答案工件名（evaluate.py、solution、rubrics、metrics.json、run_eval.sh 等）
        (r"evaluate\.py|solution\.txt|solution\.sv|rubrics?\.txt|rubics\.txt|"
         r"reference\.txt|metrics\.json|artifacts\.json|frontier_eval|run_eval\.sh",
         "names grading / reference-solution artefacts"),
        # 伸手去够未污染的基准树（engdesign_bench、EngDesignOpen、grading venv）
        (r"engdesign_bench|benchmarks/EngDesignOpen|\.venvs/engdesign",
         "reaches for the pristine benchmark tree"),
        # 绝对路径或向上逃逸出 agent 工作区（HOME 环境变量、/home、三级上跳、chdir 到根）
        (r"os\.environ\[[\"\']HOME|/home/|\.\./\.\./\.\.|os\.chdir\(\s*[\"\']/",
         "absolute or upward path escape out of the agent workspace"),
        # 把 verifier 的打分内部（阈值、score_max、combined_score）编码进 scaffold
        (r"pass_threshold|score_max|combined_score|\bcombined score\b",
         "encodes the verifier's scoring internals into the scaffold"),
        # 基准名硬编码进 scaffold
        (r"EngDesign|eng_design|engdesign", "benchmark name hardcoded in scaffold"),
        # 基准任务 id（AB_01 这样的编号）硬编码进 scaffold——记忆具体任务
        (r"\b(?:AB|AM|AV|CY|DL|HC|HJ|JY|KV|LX|NS_PA_SS|RK|WJ|XG|XW|XY|XZ|YJ|YX|"
         r"Yiqi|Yuqi|ZC|ZH|libin2)_\d\d\b", "benchmark task id hardcoded in scaffold"),
        # 把 scaffold 绑死在本套件部分任务的工具链上（iverilog/vvp/octave/ffmpeg/cvxopt/control）
        (r"iverilog|vvp\b|octave|ffmpeg|cvxopt|\bcontrol\.matlab\b",
         "keys the scaffold to the toolchain of some tasks in THIS suite"),
        # 从牢笼内部联网或装包（pip/apt/conda install、urllib/requests/curl/wget、裸 socket、URL）
        (r"pip[\"'\],\s]+install|apt-get|conda[\"'\],\s]+install|urllib|"
         r"requests\.(?:get|post)|curl\s+-|wget\s|socket\.socket|http://|https://",
         "network or package install from inside the jail"),
    ]
    # component_signals：diff 正则 -> 组件标签的 domain 专属信号（给 rrsi.components 分类用）
    component_signals = [
        # memory：跨任务持久状态（DRMH_STATE_DIR）与摘要沉淀（digest(）
        ("memory",          [r"DRMH_STATE_DIR", r"digest\("]),
        # context_mgmt：上下文压缩（ReSum/resum.py、compact、keep_last 等）
        ("context_mgmt",    [r"resum\.py", r"ReSum", r"summariz", r"compact",
                             r"_should_compress", r"keep_last"]),
        # output_plumbing：工具结果输出管道（tool_result.py、截断、HEAD/TAIL_CHARS、token 上限）
        ("output_plumbing", [r"tool_result\.py", r"truncat", r"HEAD_CHARS", r"TAIL_CHARS",
                             r"MAX_RESULT_TOKENS", r"_process_result"]),
        # control_flow：主循环控制（工具调用分发、step、final_answer、todo 门控、max_steps）
        ("control_flow",    [r"_handle_tool_calls", r"def step", r"final_answer",
                             r"has_incomplete_todos", r"max_steps", r"_execute_mcp_tool"]),
        # config：数值配置（阈值、_CAP/_LIMIT 等魔法数）
        ("config",          [r"= *\d+ *#", r"threshold", r"_CAP\b", r"_LIMIT\b"]),
        # prompt：提示词工程（prompts.py、SYSTEM_PROMPT、任务模板）
        ("prompt",          [r"prompts\.py", r"SYSTEM_PROMPT", r"TASK_TEMPLATE", r"prompt",
                             r"description"]),
    ]

    # 定义 __init__：把 evolve 集任务按 id 建索引（load_trial 渲染时要查任务元数据）
    def __init__(self):
        # 任务 id -> 任务记录 的字典
        self._tasks = {t["id"]: t for t in SPLIT["evolve"]}

    # ---- task sets ---------------------------------------------------------
    # —— 任务集合划分 ——
    # evolve 集：搜索用的 61 个 EngDesign 任务
    def evolve_ids(self) -> list[str]:
        return [t["id"] for t in SPLIT["evolve"]]

    # 同分布 held-out：本域没有
    def heldout_ids(self) -> list[str]:
        return []          # EngDesign v1 / Frontier-Eng are run by scripts/final_eval.sh
        # held-out 为空：EngDesign v1（加固版）/ Frontier-Eng 由 scripts/final_eval.sh 另行评估

    # smoke 活性检查用的小任务集
    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        # 想要的任务数（默认 4）
        n = CFG.get("smoke_n", 4)
        # 全部 evolve 任务 id
        ids = self.evolve_ids()
        # 有现任（incumbent）每任务分数可用时，优先挑"现任稳定全过"的任务（最小噪声）
        if incumbent_per_task:
            # solid：现任每个 trial 奖励都 >= 1.0 的任务
            solid = [i for i in ids if i in incumbent_per_task
                     and min(incumbent_per_task[i].rewards or [0]) >= 1.0]
            # 够数就用稳定任务（smoke 失败更可能是 scaffold 坏了而不是任务太难）
            if len(solid) >= n:
                return solid[:n]
        # 兜底：直接取前 n 个任务
        return ids[:n]

    # ---- Evaluate ----------------------------------------------------------
    # —— Evaluate(H', D, k)：跑 agent + 冻结 verifier ——
    # 定义 _env：为子进程构造环境变量（runs 目录、划分、基准树、grading venv、
    # harness 模块名、工作区根、gateway 端口、agent 解释器）
    def _env(self, runs_dir: Path) -> dict:
        return {**os.environ, "RRSI_RUNS_DIR": str(runs_dir),
                "RRSI_SPLIT_PATH": str(HERE / "data" / "split_engd.json"),
                "BENCH_ROOT": str(BENCH_ROOT), "GRADING_PYTHON": str(GRADING_PY),
                "RRSI_HARNESS_MODULE": "harness_eng.main",
                "WORKSPACE_BASE": str(runs_dir / "workspaces"),
                "GATEWAY_PORT": str(GATEWAY_PORT), "AGENT_PYTHON": AGENT_PY}
        # 关键点：RRSI_HARNESS_MODULE 指向 harness_eng.main——worktree 里的可演化 scaffold；
        # WORKSPACE_BASE 按 job 隔离，互不污染

    # 定义 _ensure_gateway：确保 MCP gateway（工具环境）已在本进程外运行，没有就拉起并等就绪
    def _ensure_gateway(self, root: Path, runs_dir: Path) -> None:
        # 端口已有人监听：gateway 已在，直接复用（无状态，可服务所有并发 trial）
        if _port_up(GATEWAY_PORT):
            return
        # 先建好工作区根目录（gateway 与 trial 都要用）
        (runs_dir / "workspaces").mkdir(parents=True, exist_ok=True)
        # 用 gateway.sh start 拉起 gateway（日志在 logs/gateway.log）
        subprocess.run(["bash", str(HERE / "scripts" / "gateway.sh"), "start"],
                       env=self._env(runs_dir), cwd=str(HERE))
        # 轮询最多 40 秒等端口就绪
        for _ in range(40):
            if _port_up(GATEWAY_PORT):
                return
            time.sleep(1)
        # 超时未就绪：直接失败（没有工具环境后续全部没意义）
        raise RuntimeError("engdesign gateway failed to start (see logs/gateway.log)")

    # 定义 _run_and_verify：Evaluate 的两步——run_tasks.py 跑 agent，verify.py 冻结评分
    def _run_and_verify(self, root: Path, runs_dir: Path, job: str, ids: list[str],
                        k: int, log_name: str) -> None:
        # 先保证 gateway 在线
        self._ensure_gateway(root, runs_dir)
        # worktree 内的 eng 域目录
        eng = root / "domains" / "eng"
        # 日志路径（追加写）
        log = runs_dir / "logs" / f"{log_name}.log"
        # 确保日志目录存在
        log.parent.mkdir(parents=True, exist_ok=True)
        # 子进程环境变量
        env = self._env(runs_dir)
        # 第 1 步命令：AGENT_PY 跑 bench/run_tasks.py --job <job> --n k --ids <逗号分隔>
        cmd = [AGENT_PY, str(eng / "bench" / "run_tasks.py"), "--job", job, "--n", str(k),
               "--ids", ",".join(ids)]
        # 运行 agent（stdout+stderr 全进日志）
        with open(log, "a") as lf:
            r = subprocess.run(cmd, cwd=str(eng), env=env, stdout=lf, stderr=subprocess.STDOUT)
        # 运行失败只告警不抛——缺失 trial 会按 0 计（完整分母），verify 仍照常执行
        if r.returncode != 0:
            print(f"[eng] WARNING run rc={r.returncode} (see {log})", flush=True)
        # 第 2 步命令：冻结 verifier 批量评分（用当前解释器跑，与 agent 解释器解耦）
        with open(log, "a") as lf:
            v = subprocess.run([sys.executable, str(eng / "bench" / "verify.py"), "--job", job],
                               cwd=str(eng), env=env, stdout=lf, stderr=subprocess.STDOUT)
        # verify 失败必须抛错：没有 verdict 就打分会把一切误判成 0，绝不能带病继续
        if v.returncode != 0:
            raise RuntimeError(f"verify failed for job={job} (rc={v.returncode}); see {log}")

    # 定义 run：Domain.run 的实现——日志名缺省用 job 名
    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        self._run_and_verify(Path(root), Path(runs_dir), job, ids, k, log_prefix or job)

    # 定义 _records：读一个任务在作业下的全部 trial 记录（要 meta+verdict 齐全才算数）
    def _records(self, runs_dir: Path, job: str, tid: str) -> list[dict]:
        # 结果列表
        out = []
        # 该任务的 trial 目录
        tdir = runs_dir / "jobs" / job / tid
        # 任务目录不存在：没跑过
        if not tdir.is_dir():
            return out
        # 遍历排序后的 t* trial 目录
        for trial in sorted(tdir.glob("t*")):
            # meta.json（跑完的标记）与 verdict.json（评完的标记）缺一不可
            if not (trial / "meta.json").is_file() or not (trial / "verdict.json").is_file():
                continue
            # 读文件失败（半截 JSON）按坏 trial 处理
            try:
                v = json.loads((trial / "verdict.json").read_text())
                meta = json.loads((trial / "meta.json").read_text())
            except Exception:  # noqa: BLE001
                continue
            # 收集三个 verifier 轴 + 提交状态 + token
            out.append({"score": float(v.get("combined_score") or 0.0),
                        "passed": bool(v.get("passed")), "valid": float(v.get("valid") or 0.0),
                        "no_payload": bool(meta.get("no_payload")),
                        "tokens": meta.get("total_tokens")})
        # 返回有效 trial 记录
        return out

    # 定义 score：把 trial 产物折算成每任务 TaskResult 与全局聚合（四条轴：passed/
    # combined_score/valid/no_payload）；缺失 trial 全轴记 0（no_payload 记 1），分母不缩
    def score(self, runs_dir, job, ids, k):
        # 统一转 Path
        runs_dir = Path(runs_dir)
        # 每任务结果
        per = {}
        # 全局计数器：通过数 / 有效数 / 无提交数
        passes = valid = nopay = 0
        # 全局 combined_score 列表
        scores = []
        # 逐任务统计
        for t in ids:
            # 该任务前 k 个有效 trial 记录
            recs = self._records(runs_dir, job, t)[:k]
            # 主轴奖励：verifier 的二元 passed
            rewards = [1.0 if r["passed"] else 0.0 for r in recs]
            # token 用量（非整数记 None）
            toks = [r["tokens"] if isinstance(r["tokens"], int) else None for r in recs]
            # 缺失 trial 数 = k - 有效记录数
            missing = k - len(recs)
            # 缺失位奖励补 0（missing-as-zero）
            rewards += [0.0] * missing
            # 缺失位 token 记 None
            toks += [None] * missing
            # combined_score 轴（缺失补 0）
            cs = [r["score"] for r in recs] + [0.0] * missing
            # valid 轴（缺失补 0）
            vs = [r["valid"] for r in recs] + [0.0] * missing
            # no_payload 轴（缺失记 1：连跑都没跑成等价于什么都没交）
            npay = [1.0 if r["no_payload"] else 0.0 for r in recs] + [1.0] * missing
            # 累加全局通过数
            passes += int(sum(rewards))
            # 累加全局有效数
            valid += sum(vs)
            # 累加全局无提交数
            nopay += sum(npay)
            # 收集 combined_score
            scores += cs
            # 记录每任务结果，extra 带三条副轴（valid/no_payload 喂 guards）
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing,
                                extra={"combined_scores": cs, "valid": vs, "no_payload": npay})
        # 全局分母（防 0）
        n = max(1, len(ids) * k)
        # 返回每任务结果与聚合：通过率、平均 combined_score、valid 率、no_payload 率
        return per, {"total_passes": passes, "n_trials": len(ids) * k, "pass_rate": passes / n,
                     "mean_combined_score": sum(scores) / n, "valid_rate": valid / n,
                     "no_payload_rate": nopay / n}

    # 定义 guards：两个非补偿性 domain 判据（论文 3.3 节）——valid 率不许大降、
    # no_payload 率不许大升；违反即返回理由列表（selection 一票否决）
    def guards(self, incumbent, candidate) -> list[str]:
        # 违规理由列表
        out = []
        # valid 率跌幅 = 现任 - 候选
        d_valid = (incumbent.extra.get("valid_rate") or 0.0) - (candidate.extra.get("valid_rate") or 0.0)
        # 超过配置允许的最大跌幅：记一条违规（"damage guard"，防靠激进设计涮分）
        if d_valid > CFG["max_valid_rate_drop"]:
            out.append(f"valid rate fell {d_valid:+.3f} (limit {CFG['max_valid_rate_drop']})")
        # no_payload 率涨幅 = 候选 - 现任
        d_np = (candidate.extra.get("no_payload_rate") or 0.0) - (incumbent.extra.get("no_payload_rate") or 0.0)
        # 超过允许的最大涨幅：记一条违规（防"交白卷"型退化）
        if d_np > CFG["max_no_payload_rise"]:
            out.append(f"no-submission rate rose {d_np:+.3f} (limit {CFG['max_no_payload_rise']})")
        # 返回全部违规（空列表 = 通过）
        return out

    # ---- evidence ----------------------------------------------------------
    # —— 证据读取（供 analyst/digester/proposer）——
    # 定义 load_trial：取指定 trial 的原始记录（任务元数据 + 渲染输入），委托给 render 模块
    def load_trial(self, runs_dir, job, task_id, trial):
        # 查任务元数据（domain 字段等）；查不到用占位
        task = self._tasks.get(task_id) or {"id": task_id, "domain": "?"}
        # render.load_trial 读 meta/verdict/traj 并组装记录
        return render.load_trial(Path(runs_dir) / "jobs" / job, task, trial)

    # 定义 render_trace：把一条 trial 记录渲染成可读文本
    def render_trace(self, rec, detail=False):
        # detail=True 时放开截断上限
        return render.render_full(rec, detail=detail)

    # 定义 task_row：任务表里该轨迹的一行摘要——先给失败定性（哪一类），再给关键计数
    def task_row(self, task_id, rec, tr):
        # 三份原始数据：meta（运行状态）、verdict（评分）、diag（行为诊断统计）
        meta, verdict, diag = rec.get("meta") or {}, rec.get("verdict") or {}, rec.get("diag") or {}
        # 任务元数据（domain 字段）
        task = rec.get("task") or {}
        # 轨迹消息列表（数 assistant 步用）
        msgs = (rec.get("traj") or {}).get("messages") or []
        # verifier 的错误文本
        err = str(verdict.get("error") or "")
        # 失败分类（见 briefs._CLASSES 的四分法）：
        if diag.get("no_payload"):
            # 没写任何 payload：终止/步数预算/管道问题，不是工程问题
            kind = "NO-SUBMISSION(nothing was written)"
        elif (verdict.get("status") or "ok") != "ok":
            # verifier 自身出错（基础设施）：不构成对设计的证据
            kind = f"INFRA({verdict.get('status')}) -- not evidence about the design"
        elif verdict.get("passed"):
            # 通过
            kind = "passed"
        elif "literal" in err.lower() or "validation" in err.lower() or "PAYLOAD" in err:
            # payload 写了但 verifier 用不了（非纯字面量/schema 字段错/类型错）——设计可能在最后一步被扔掉
            kind = "MALFORMED(design discarded at the last step)"
        elif float(verdict.get("valid") or 0.0) < 1.0:
            # 解析得了但违反硬约束
            kind = "INVALID-DESIGN(violated a hard constraint)"
        else:
            # 有效但没过性能门槛
            kind = "UNDER-TARGET(valid but missed the gate)"
        # 附加标记：从未通过且一次代码都没执行过——"算了就交"家族的强信号
        flag = " NEVER-RAN-ANY-CODE" if (not verdict.get("passed")
                                          and (diag.get("n_code_exec") or 0) == 0) else ""
        # 拼一行：任务 | 学科 | 分类 | 通过率 | combined_score | valid | 状态 | 步数 | 代码执行次数(失败数) | payload 写入次数
        return (f"{task_id} | {task.get('domain')} | {kind}{flag} | pass_rate={tr.mean:.2f} "
                f"score={float(verdict.get('combined_score') or 0.0):.3f} "
                f"valid={verdict.get('valid')} | status={meta.get('status')} | "
                f"steps={sum(1 for m in msgs if m.get('role') == 'assistant')} | "
                f"code_exec={diag.get('n_code_exec')}({diag.get('n_code_exec_failed')} err) "
                f"payload_writes={diag.get('n_payload_writes')}")

    # ---- gates -------------------------------------------------------------
    # —— 闸门 ——
    # 定义 smoke：评估前的活性检查：编译 scaffold -> 注册表取 agent + 模板契约哨兵检查 ->
    # 清目录实跑 smoke 任务（k=1）并打分 -> 无缺失、无白卷、均分>0 才放行
    def smoke(self, root, runs_dir, job, ids):
        # 统一转 Path
        root, runs_dir = Path(root), Path(runs_dir)
        # worktree 内 eng 域目录
        eng = root / "domains" / "eng"
        # 第 1 关：语法编译整个 harness 目录
        comp = subprocess.run([AGENT_PY, "-m", "compileall", "-q", str(self.harness_dir(root))],
                              capture_output=True, text=True)
        # 编译失败：判死
        if comp.returncode != 0:
            return False, {"stage": "compile", "err": (comp.stdout + comp.stderr)[-1500:]}
        # 第 2 关：内联脚本——从引擎注册表取 react_toolbelt_agent 实现，检查系统提示词非空，
        # 并用 SENTINEL 哨兵渲染任务模板，确认四个关键占位（工作区/提示/契约/交付物）
        # 与 PAYLOAD 交付契约都还在（防止编辑悄悄删掉契约）
        code = (
            "import sys, os; sys.path.insert(0, '../../third_party/archipelago'); os.environ['RRSI_HARNESS_MODULE'] = 'harness_eng.main'\n"
            "from runner.agents.registry import get_agent_impl\n"
            "from harness_eng.prompts import SYSTEM_PROMPT, TASK_TEMPLATE\n"
            "impl = get_agent_impl('react_toolbelt_agent')\n"
            "assert callable(impl), 'agent impl not callable'\n"
            "assert SYSTEM_PROMPT.strip(), 'empty system prompt'\n"
            "body = TASK_TEMPLATE.format(workspace='SENTINEL-WS', prompt_file='SENTINEL-PROMPT',\n"
            "                            schema_file='SENTINEL-SCHEMA', candidate_file='SENTINEL-CAND')\n"
            "for tok in ('SENTINEL-WS', 'SENTINEL-PROMPT', 'SENTINEL-SCHEMA', 'SENTINEL-CAND'):\n"
            "    assert tok in body, f'task template drops {tok}'\n"
            "assert 'PAYLOAD' in body, 'deliverable contract lost from task template'\n"
            "print('CTOR OK')\n")
        # 执行第 2 关脚本
        ctor = subprocess.run([AGENT_PY, "-c", code], cwd=str(eng), capture_output=True, text=True)
        # 退出码非 0 或没看到 CTOR OK：判死
        if ctor.returncode != 0 or "CTOR OK" not in ctor.stdout:
            return False, {"stage": "ctor", "err": (ctor.stderr or ctor.stdout)[-1500:]}
        # 第 3 关：删同名作业目录，实跑 smoke 任务（每任务 1 trial，含冻结评分）
        subprocess.run(["rm", "-rf", str(runs_dir / "jobs" / job)])
        self._run_and_verify(root, runs_dir, job, ids, 1, job)
        # 取 smoke 打分结果
        per, extra = self.score(runs_dir, job, ids, 1)
        # 缺失 trial 总数
        missing = sum(tr.missing for tr in per.values())
        # 放行条件：无缺失、零白卷、平均 combined_score > 0（三项缺一不可）
        ok = missing == 0 and extra["no_payload_rate"] == 0.0 and extra["mean_combined_score"] > 0.0
        # 返回 (是否存活, 详情)
        return ok, {"stage": "run", "smoke_ids": ids, "missing": missing, **extra}


# 模块级单例：load_domain 约定从 adapter 模块读取 DOMAIN
DOMAIN = EngDomain()
