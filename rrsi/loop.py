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
"""One round of RRSI: Algorithm 1 (proposal side) then Algorithm 2 (selection
side), for round t = 0..T-1 of one domain.

    run.round(t)
      1. F_t   <- Analyze(H_t, D_evolve)                     analyst over the incumbent's own eval
      2. b_t   <- annealed edit budget                         schedule.edit_budget
      3. sigma_t, T_t, U_t, E_t, B_t                            history.stall_flag / History
      4. for each of m variants (own worktree + branch):
             C_t^(v) ~ P_reg(. | H_t, F_t, L_t, b_t, E_t)      propose.propose
             tag edits (component, hypothesis, diff)
             Critic(H_t, C_t^(v)) with bounded repair          critic.review
             liveness smoke (compile / construct / few tasks)   domain.smoke
      5. Evaluate(H', D_evolve, k) for the screened set        evaluate.evaluate (parallel optional)
      6. admissibility, argmax, S*, history records, B_t       select.select_round
      7. fast-forward evolve/<domain> to the winner, update frontier

Everything written under runs/<domain>/ is resume-safe: a round that crashed
after evaluating variant A reuses A's eval.json when re-run.
"""
# 【模块说明（中文）】RRSI 的一轮：先 Algorithm 1（提案侧）再 Algorithm 2（选择侧），
# 对应某 domain 的第 t 轮（t = 0..T-1）。主循环步骤：
#   1. F_t <- Analyze(H_t, D_evolve)：分析师基于现任自己的评估产出三透镜报告
#   2. b_t <- 退火编辑预算（schedule.edit_budget）
#   3. 取 sigma_t（停滞标志）、T_t（已尝试）、U_t（未尝试）、E_t（探索指令）、B_t（修剪集）
#   4. 对 m 个变体各起一个 worktree + 分支：起草（propose.propose）、打标签（组件/假设/diff）、
#      critic 泄漏筛查（有界修复 bounded repair）、活性 smoke（编译/构造/少量任务）
#   5. 对通过筛选的候选并行 Evaluate(H', D_evolve, k)
#   6. Algorithm 2：可接纳性、argmax、S*、历史记录、B_t（select.select_round）
#   7. 把 evolve/<domain> 快进到获胜者，更新 frontier
# runs/<domain>/ 下的一切都可断点续跑：评估完变体 A 后崩溃的一轮，重跑时会复用 A 的 eval.json。

# 延迟类型注解求值
from __future__ import annotations

# json：frontier/报告/决策等所有 JSON 产物
import json
# os：normpath 规范化 harness 相对路径
import os
# time：日志时间戳
import time
# ThreadPoolExecutor：并行评估多个候选
from concurrent.futures import ThreadPoolExecutor
# Path：全部路径对象
from pathlib import Path

# gitops（别名 G）：worktree/分支/快进等 git 底层操作
from . import gitops as G
# analyze / load_digests：分析师入口与摘要回读
from .analyst import analyze, load_digests
# calibrate / write：噪声带校准与落盘
from .calibrate import calibrate as _calibrate, write as _write_cal
# normalize：按 diff 校正编辑的组件标签
from .components import normalize
# RRSIConfig：超参数对象
from .config import RRSIConfig
# review：critic 泄漏筛查
from .critic import review
# EvalResult / evaluate：评估结果结构与评估入口
from .evaluate import EvalResult, evaluate
# History / exploration / stall_flag：编辑历史与探索指令
from .history import History, exploration, stall_flag
# propose：提案者入口
from .propose import propose
# edit_budget：退火编辑预算 b_t
from .schedule import edit_budget
# Candidate / select_round：候选结构与 Algorithm 2 入口
from .selection import Candidate, select_round

# VARIANT_LABELS：变体标签字母表——每轮最多 8 个候选（cfg.m <= 8）
VARIANT_LABELS = "ABCDEFGH"


def log(domain: str, msg: str) -> None:  # 定义 log：统一前缀的进度日志
    print(f"[rrsi:{domain}] {time.strftime('%H:%M:%S')} {msg}", flush=True)  # 带 domain 名与时间戳，flush 保证实时可见（子进程日志依赖它）


class Run:  # Run 类：一个 domain 演化的路径、状态与 git 管道（主循环的宿主）
    """Paths, state and git plumbing of one domain's evolution."""

    def __init__(self, domain, cfg: RRSIConfig, repo: Path, runs_root: Path):  # 构造：绑定 domain/配置/仓库根/运行根
        self.domain, self.cfg, self.repo = domain, cfg, Path(repo)  # 保存三元组
        self.runs = Path(runs_root) / domain.name  # runs/<domain>/：本 domain 的全部产物
        self.jobs = self.runs / "jobs"  # jobs/：各次评估（base、r3A、heldout_*）的 eval.json
        self.wt_root = self.runs / "wt"  # wt/：候选与临时 worktree 的挂载点
        self.frontier_path = self.runs / "frontier.json"  # frontier：现任、S*、轨迹、配置
        self.calibration_path = self.runs / "calibration.json"  # 噪声带校准结果
        self.global_analysis = self.runs / "global_analysis.json"  # 跨轮保留的失败模式/成功习惯（稳定命名用）
        self.attribution_path = self.runs / "attribution.jsonl"  # 归因记分板（预测命中率的 JSONL）
        self.history = History(self.runs / "history.jsonl")  # 编辑历史 L_t
        self.branch = f"evolve/{domain.name}"  # 本 domain 的演化分支
        self.harness_rel = os.path.normpath(f"domains/{domain.name}/{domain.harness_path}")  # harness 相对路径（diff/提交/树哈希都用它）
        for d in (self.runs, self.jobs, self.wt_root, self.runs / "logs"):  # 逐个确保目录存在
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ state ---
    # —— 状态层：frontier 与噪声带 ——
    def frontier(self) -> dict:  # 定义 frontier：读取 frontier.json
        if not self.frontier_path.exists():  # 还没跑 baseline
            raise SystemExit(f"no {self.frontier_path}; run `baseline` first")  # 明确报错并给出补救命令
        return json.loads(self.frontier_path.read_text())  # 返回反序列化后的 frontier

    def save_frontier(self, fr: dict) -> None:  # 定义 save_frontier：写回 frontier.json
        self.frontier_path.write_text(json.dumps(fr, indent=1))  # indent=1 便于人工审计

    def delta(self) -> float:  # 定义 delta：解析噪声带 delta 的三级来源
        if self.cfg.delta is not None:  # 1) 配置显式给定（论文里三个实例的固定值）
            return float(self.cfg.delta)
        if self.calibration_path.exists():  # 2) 读校准结果（calibrate.py 产物）
            return float(json.loads(self.calibration_path.read_text())["delta"])
        raise SystemExit("no noise band: set cfg.delta or run `calibrate`")  # 3) 两者皆无：报错

    def eval_path(self, job: str) -> Path:  # 定义 eval_path：某评估作业的 eval.json 路径
        return self.jobs / job / "eval.json"  # jobs/<job>/eval.json

    def incumbent_eval(self, fr: dict) -> EvalResult:  # 定义 incumbent_eval：现任 harness 的评估结果
        return EvalResult.load(self.eval_path(fr["incumbent"]["job"]))  # 按 frontier 记录的 job 名回读

    # -------------------------------------------------------------- git ---
    # —— git 层 ——
    def ensure_branch(self) -> None:  # 定义 ensure_branch：保证演化分支存在
        G.ensure_branch(self.repo, self.branch, "HEAD")  # 不存在则从 HEAD 创建

    def checkout(self, name: str, start: str, branch: str | None = None) -> Path:  # 定义 checkout：建一个 worktree
        """Worktree `name` at `start`; on a new branch when `branch` is given,
        detached otherwise."""
        # docstring：在 start 处建名为 name 的 worktree；给了 branch 就开新分支，否则游离检出
        path = self.wt_root / name  # worktree 路径 runs/<domain>/wt/<name>
        if branch:  # 候选评估用：带分支（可提交、可被快进引用）
            return G.worktree_add(self.repo, path, branch, start)
        G.worktree_remove(self.repo, path)  # 临时用途：先清掉同名旧 worktree
        path.parent.mkdir(parents=True, exist_ok=True)  # 父目录就绪
        G.git(self.repo, "worktree", "add", "--detach", str(path), start, check=True)  # --detach：游离 HEAD（baseline/heldout/smoke 只读用）
        return path  # 返回路径

    def harness_tree(self, ref: str) -> str:  # 定义 harness_tree：某引用下 harness 子树的哈希
        return G.tree_hash(self.repo, ref, self.harness_rel)  # 内容指纹：harness 变没变由它判定，与提交历史无关

    # --------------------------------------------------------- evidence ---
    # —— 证据层：轨迹挑选与归因 ——
    def build_traces(self, job: str, per_task: dict) -> dict:  # 定义 build_traces：挑选送入分析师的轨迹
        """Worst trial of the lowest-scoring tasks plus the best trial of the
        highest-scoring ones (success habits the proposer must not break)."""
        # docstring：最低分任务的最差 trial + 最高分任务的最好 trial（提案者不许破坏的成功习惯）
        ranked = sorted(per_task.items(), key=lambda kv: kv[1].mean)  # 按任务均分升序
        fails = [t for t, _ in ranked[:self.cfg.n_fail_traces]]  # 最低分的 n_fail_traces 个任务（失败透镜）
        wins = [t for t, _ in ranked[-self.cfg.n_success_traces:] if t not in fails]  # 最高分的 n_success_traces 个（成功透镜；与 fails 去重）
        traces = {}  # task_id -> 原始 trial 记录
        for tid in fails + wins:  # 遍历选中的任务
            tr = per_task[tid]  # 该任务结果
            if not tr.rewards:  # 没有任何 trial 记录
                continue  # 跳过
            idx = (tr.rewards.index(min(tr.rewards)) if tid in fails  # 失败任务取最差 trial；成功任务取最好 trial
                   else tr.rewards.index(max(tr.rewards)))
            rec = self.domain.load_trial(self.runs, job, tid, idx)  # 由 domain 读回原始轨迹
            if rec is not None:  # 读到了
                traces[tid] = rec  # 收进集合
        return traces  # 返回分析师的证据集

    def scoreboard(self) -> list:  # 定义 scoreboard：读取归因记分板的最近 20 行
        if not self.attribution_path.exists():  # 从未归因过
            return []
        rows = [json.loads(l) for l in self.attribution_path.read_text().splitlines()  # 逐行解析
                if l.strip()]  # 跳过空行
        return rows[-20:]  # 只回看最近 20 条（记分板太长会淹没上下文）

    def attribute(self, t: int, variant: str, edits: list, inc: EvalResult,  # 定义 attribute：把候选的预测与实测对账（写归因 JSONL）
                  cand: EvalResult) -> list:
        thr = self.domain.regression_threshold(self.cfg.k)  # 单任务"回退"阈值（默认 1/k）
        rows = []  # 归因行
        for e in edits:  # 逐编辑对账
            pred = [str(p) for p in (e.get("predicted_affected") or [])  # 该编辑预测受影响的任务
                    if str(p) in inc.per_task and str(p) in cand.per_task]  # 只保留两次评估都有的任务（可比较）
            hits = [p for p in pred if cand.per_task[p].mean > inc.per_task[p].mean]  # 命中：预测任务确实变好
            drops = [x for x in inc.per_task if x in cand.per_task  # 实测回退：均值下降超阈值
                     and cand.per_task[x].mean - inc.per_task[x].mean <= -thr]
            rows.append({"t": t, "variant": variant, "edit_id": e.get("id"),  # 记录轮次/变体/编辑 id
                         "component": e.get("component"),  # 组件
                         "hypothesis": str(e.get("hypothesis"))[:160],  # 假设截 160 字符
                         "n_predicted": len(pred), "predicted_hit": hits,  # 预测数与命中明细
                         "hit_rate": round(len(hits) / len(pred), 2) if pred else None,  # 命中率（无预测为 None）
                         "unpredicted_regressions": [d for d in drops if d not in pred][:12]})  # 未被预测到的回退（最贵的学习信号）截 12 条
        with open(self.attribution_path, "a") as f:  # 追加写归因 JSONL
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows  # 返回（供日志）

    # ----------------------------------------------------------- baseline --
    # —— baseline：H_0 评估与 frontier 播种 ——
    def baseline(self, job: str = "base") -> EvalResult:  # 定义 baseline：评估 H_0 并播种 frontier
        """Evaluate H_0 (the tip of evolve/<domain>) and seed the frontier."""
        # docstring：评估 H_0（evolve/<domain> 的分支尖）并播种 frontier
        self.ensure_branch()  # 确保演化分支存在
        wt = self.checkout("incumbent", self.branch)  # 游离检出分支尖到临时 worktree
        ids = self.domain.evolve_ids()  # evolve 集任务 id
        log(self.domain.name, f"baseline: {len(ids)} tasks x k={self.cfg.k} -> job {job}")  # 宣告评估规模
        ev = evaluate(self.domain, wt, self.runs, job, ids, self.cfg.k,  # 正式评估 H_0（k trials/任务）
                      log_prefix=f"{job}")
        ev.save(self.eval_path(job))  # 落盘 jobs/base/eval.json（后续每轮的现任基线）
        if ev.missing > self.cfg.invalid_missing_frac * ev.n_expected:  # 缺失 trial 超比例
            raise SystemExit(f"baseline invalid: {ev.missing}/{ev.n_expected} trials missing")  # 判定基线无效——噪声带与比较全部失真
        commit = G.rev(self.repo, self.branch)  # H_0 的提交哈希
        fr = {"domain": self.domain.name,  # 组装 frontier
              "incumbent": {"t": 0, "commit": commit,  # 现任：第 0 轮
                            "harness_tree": self.harness_tree(self.branch),  # harness 树哈希（内容指纹）
                            "job": job, "S": ev.S, "C": ev.C, "extra": ev.extra},  # 分数/成本/附加聚合
              "S_star": ev.S,  # 历史最好分（noise-adjusted floor 的锚点）
              "trajectory": [{"t": 0, "S": ev.S, "C": ev.C, "commit": commit, "job": job}],  # 分数轨迹（停滞检测的输入）
              "config": self.cfg.dump()}  # 配置快照（复现用）
        self.save_frontier(fr)  # 落盘
        self.history.append({"t": 0, "variant": "-", "edit_id": None, "component": None,  # 历史里记一条 BASELINE（a=1，但无编辑）
                             "hypothesis": "H_0 baseline", "outcome": "BASELINE",
                             "S": ev.S, "C": ev.C, "accepted": True,
                             "delta_S": None, "delta_C": None, "bundle": 0})
        log(self.domain.name, f"baseline S={ev.S:.4f} C={ev.C} missing={ev.missing}"  # 汇报基线结果
            f"/{ev.n_expected} {json.dumps(ev.extra)[:200]}")
        return ev  # 返回评估结果

    def calibrate(self, jobs: list[str]) -> dict:  # 定义 calibrate：对给定评估作业估噪声带
        evals = [EvalResult.load(self.eval_path(j)) for j in jobs]  # 回读各作业的 eval.json
        cal = _calibrate(evals, z=self.cfg.delta_z)  # 交给 calibrate.py 计算法（bootstrap 或重复评估）
        cal["jobs"] = jobs  # 记录所用作业（审计）
        _write_cal(self.calibration_path, cal)  # 落盘 calibration.json
        log(self.domain.name, f"calibrated delta={cal['delta']:.5f} "  # 汇报 delta 与方法
            f"(sd_null {cal['sd_null']:.5f}, z={cal['z']}, {cal['method']})")
        return cal  # 返回校准详情

    def heldout(self, label: str, ids: list[str], ref: str | None = None,  # 定义 heldout：在 held-out/OOD 集上评估某引用
                k: int | None = None) -> EvalResult:
        ref = ref or self.branch  # 默认评估现任（分支尖）
        wt = self.checkout(f"heldout_{label}", ref)  # 临时 worktree 检出
        job = f"heldout_{label}"  # 作业名
        ev = evaluate(self.domain, wt, self.runs, job, ids, k or self.cfg.k,  # 评估（不参与选择，纯测量）
                      log_prefix=job)
        ev.save(self.eval_path(job))  # 落盘
        log(self.domain.name, f"heldout {label} @ {G.rev(self.repo, ref)}: S={ev.S:.4f} "  # 汇报
            f"C={ev.C} missing={ev.missing}/{ev.n_expected} {json.dumps(ev.extra)[:200]}")
        return ev  # 返回结果

    # -------------------------------------------------------------- round --
    # —— 主循环：一轮的完整流程 ——
    def round(self, t: int, dry_run: bool = False) -> None:  # 定义 round：第 t 轮（Alg.1 + Alg.2）
        d, cfg = self.domain, self.cfg  # 局部别名
        fr = self.frontier()  # 读 frontier
        inc = fr["incumbent"]  # 现任条目
        if inc["harness_tree"] != self.harness_tree(self.branch):  # 分支上的 harness 内容与 frontier 记录不符
            raise SystemExit(f"harness tree of {self.branch} != frontier "  # 说明有人手动改了分支——拒绝继续（防状态漂移）
                             f"({self.harness_tree(self.branch)} vs {inc['harness_tree']})")
        if len(fr["trajectory"]) < t + 1:  # 轨迹长度不够（前面的轮还没跑）
            raise SystemExit(f"round {t} needs trajectory up to t={t}; have "  # 强制按顺序执行
                             f"{len(fr['trajectory'])} entries (run earlier rounds)")
        delta = self.delta()  # 本轮用的噪声带
        rdir = self.runs / f"r{t}"  # 本轮目录 runs/<domain>/r<t>/
        rdir.mkdir(exist_ok=True)
        ids = d.evolve_ids()  # evolve 集
        inc_ev = self.incumbent_eval(fr)  # 现任评估结果（所有比较的基线）
        log(d.name, f"=== round {t}/{cfg.T} incumbent t={inc['t']} {inc['commit']} "  # 宣告开轮与关键状态
            f"S={inc_ev.S:.4f} C={inc_ev.C} S*={fr['S_star']:.4f} delta={delta:.5f} ===")

        # 1) F_t <- Analyze(H_t, D_evolve)
        # 步骤 1：分析师产出三透镜报告 F_t
        traces = self.build_traces(inc["job"], inc_ev.per_task)  # 挑选失败/成功轨迹
        if len(traces) < 0.5 * min(len(ids), cfg.n_fail_traces + cfg.n_success_traces):  # 可用轨迹过少（评估产物损坏）
            raise SystemExit(f"only {len(traces)} traces available from {inc['job']}")  # 中止——证据不足的轮次没有意义
        report_path = rdir / "analysis_report.json"  # 报告落盘路径
        if report_path.exists():  # 已有报告（断点续跑）
            report = json.loads(report_path.read_text())  # 直接复用
            log(d.name, "reusing analysis_report.json")
        else:  # 首次分析
            prior = (json.loads(self.global_analysis.read_text())  # 跨轮的全局分析（稳定命名先验）
                     if self.global_analysis.exists() else {})
            report = analyze(d, traces, inc_ev.per_task, rdir,  # 调分析师（内部派发 digester）
                             prior_modes=prior.get("failure_modes"),
                             prior_habits=prior.get("success_habits"),
                             model=cfg.analyst_model)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))  # 落盘报告
        self.global_analysis.write_text(json.dumps(  # 更新全局分析（供下一轮做命名先验）
            {"failure_modes": report.get("failure_modes"),
             "success_habits": report.get("success_habits")}, ensure_ascii=False, indent=1))
        log(d.name, f"top modes: {[m.get('mode') for m in report.get('failure_modes', [])[:5]]}")  # 汇报 top 失败模式
        if dry_run:  # 演练模式：只做分析
            log(d.name, "dry-run: stopping before propose")
            return

        # 2-3) b_t, sigma_t, T_t, U_t, E_t, B_t
        # 步骤 2-3：预算与历史汇总量
        budget = edit_budget(t, cfg.T, cfg.b_min, cfg.b_max)  # 退火编辑预算 b_t（schedule.py）
        traj = [x["S"] for x in fr["trajectory"]]  # 分数轨迹
        sigma = stall_flag(traj, t, cfg.w, delta)  # 停滞标志 sigma_t（Eq. explore）
        tried = self.history.tried()  # 已尝试组件 T_t
        explore = exploration(t, sigma, tried, cfg.m_draft)  # 探索指令 E_t
        prune = self.history.prune_set(t, cfg.n_prune)  # 修剪集 B_t（含要拆除的机制）
        hist_rows = self.history.render()  # 给提案者的紧凑历史视图
        digests = load_digests(rdir)  # 本轮的分析师摘要
        skill_md, patterns_md = d.constitution(self.repo)  # 宪法两件套（SKILL.md + PATTERNS.md）
        log(d.name, f"b_t={budget} sigma_t={sigma} untried={explore['untried']} "  # 汇报本轮正则化状态
            f"prune={[p['component'] for p in prune]}")
        (rdir / "directives.json").write_text(json.dumps(  # 把本轮全部指令落盘（可审计）
            {"t": t, "b_t": budget, "sigma_t": sigma, "tried": sorted(tried),
             "explore": explore, "prune_set": prune, "delta": delta,
             "S_star": fr["S_star"]}, indent=1))

        # 4) draw and screen m candidates
        # 步骤 4：起草并筛选 m 个候选
        cands: list[Candidate] = []  # 候选列表
        for v in range(cfg.m):  # 逐个起草（相互独立）
            vid = VARIANT_LABELS[v]  # 变体标签 A/B/...
            reserved = bool(sigma and explore["untried"] and v >= cfg.m - cfg.m_draft)  # 保留槽位：停滞且排最后的 m_draft 个变体
            cands.append(self._draft(t, vid, rdir, report, digests, traces, inc_ev,  # 起草（内部走 propose->critic->smoke）
                                     budget, explore, reserved, prune, hist_rows,
                                     skill_md, patterns_md))

        # 5) Evaluate(H', D_evolve, k) for the screened set, in parallel
        # 步骤 5：对通过闸门的候选并行评估
        live = [c for c in cands if c.gate_failure is None]  # 只有通过 critic+smoke 的候选值得花评估
        with ThreadPoolExecutor(max_workers=max(1, cfg.eval_parallel)) as ex:  # 线程池（eval_parallel 个候选同时评）
            list(ex.map(lambda c: self._evaluate(t, c, rdir, ids), live))  # 阻塞等待全部完成（异常在 _evaluate 内部消化）

        # 6) Algorithm 2: admissibility, argmax, S*, history
        # 步骤 6：选择侧裁决
        counts = self.history.incumbent_component_counts()  # 现任各组件的被接受编辑数（novelty 判据）
        winner, decisions = select_round(cands, inc_ev, fr["S_star"], delta, cfg,  # Algorithm 2：地板/成本规则/guards/argmax
                                         counts, guard_fn=d.guards)
        (rdir / "decisions.json").write_text(json.dumps(  # 全部裁决落盘（可审计）
            [x.to_json() for x in decisions], indent=1))
        for c, dec in zip(cands, decisions):  # 把结局写进编辑历史
            if self.history.has(t, c.variant):  # 已写过（断点续跑）
                continue                      # resumed round: already recorded
            if c.ev is None:  # 没有测量（闸门失败/评估无效）
                outcome = c.gate_failure or "not_evaluated"  # 结局即失败名
                self.history.append_candidate(t, c.variant, c.edits, outcome, None, None,  # 记录但不进 T_t/g_t（delta_S=None）
                                              False, None, None, c.diff_path, c.detail)
                continue
            outcome = ("ACCEPTED" if c is winner else  # 有测量：ACCEPTED / LOST（可接纳但落选）/ REJECTED（不可接纳）
                       ("LOST" if dec.admissible else "REJECTED"))
            self.history.append_candidate(t, c.variant, c.edits, outcome, dec.delta_S,  # 逐编辑记录测量结果
                                          dec.delta_C, c is winner, dec.S, dec.C,
                                          c.diff_path, dec.reason)
            self.attribute(t, c.variant, c.edits, inc_ev, c.ev)  # 预测对账写归因记分板
            log(d.name, f"{c.variant}: S={dec.S:.4f} dS={dec.delta_S:+.4f} "  # 汇报每个候选的裁决
                f"dC={dec.delta_C:+.3f} nu={dec.novelty} -> {outcome}: {dec.reason}")

        # 7) H_{t+1}
        # 步骤 7：更新现任
        if winner is not None:  # 有可接纳获胜者
            G.fast_forward(self.repo, self.branch, winner.commit)  # 快进演化分支到获胜提交（现任永远是一个提交）
            new_S, new_C = winner.ev.S, winner.ev.C  # 新现任的分数/成本
            fr["incumbent"] = {"t": t + 1, "commit": winner.commit,  # 更新现任条目
                               "harness_tree": self.harness_tree(winner.commit),
                               "job": winner.ev.job, "S": new_S, "C": new_C,
                               "extra": winner.ev.extra, "variant": winner.variant}
            fr["S_star"] = max(fr["S_star"], new_S)  # S* 只升不降（地板随之抬高）
            log(d.name, f"ACCEPTED r{t}{winner.variant} -> {winner.commit} "  # 汇报接受
                f"S={new_S:.4f} S*={fr['S_star']:.4f}")
        else:  # 无可接纳候选
            new_S, new_C = inc_ev.S, inc_ev.C  # H_{t+1} = H_t（正则化刹住搜索）
            log(d.name, f"no admissible candidate; H_{t+1} = H_{t}")
        traj_entry = {"t": t + 1, "S": new_S, "C": new_C,  # 轨迹追加 t+1 条目（轮已落定的标志）
                      "commit": fr["incumbent"]["commit"], "job": fr["incumbent"]["job"]}
        fr["trajectory"] = [x for x in fr["trajectory"] if x["t"] <= t] + [traj_entry]  # 先清掉旧的 t+1 条目（readjudicate 重写场景）再追加
        self.save_frontier(fr)  # 落盘 frontier——此后本轮算 settled
        for c in cands:  # 清理本轮全部候选 worktree
            G.worktree_remove(self.repo, self.wt_root / f"r{t}{c.variant}")

    def readjudicate(self, t: int) -> None:  # 定义 readjudicate：用存储的测量重跑 Algorithm 2（不花新评估）
        """Re-run Algorithm 2 on the STORED measurements of round t, after a
        change of delta or of the acceptance weights. No new evaluation is
        spent: every candidate keeps the (S', C') it was measured at against
        the incumbent H_t it was drafted from. Rounds after t must have been
        removed first, since their candidates were drafted from the old H_{t+1}."""
        # docstring：在 delta 或接受权重变化后，对第 t 轮"已存储的测量"重跑 Algorithm 2。
        # 不花任何新评估：每个候选保留它被测量时的 (S', C')，对照它起草时的现任 H_t。
        # t 之后的轮必须先删掉，因为它们的候选是从旧的 H_{t+1} 起草的。
        d, cfg = self.domain, self.cfg  # 局部别名
        fr = self.frontier()  # 读 frontier
        if len(fr["trajectory"]) > t + 2:  # t 之后还有轮
            raise SystemExit(f"rounds after {t} exist in the frontier; remove them first")  # 拒绝（防止以旧基础重写新历史）
        inc_entry = fr["trajectory"][t]  # H_t 的轨迹条目
        inc_job = inc_entry.get("job") or ("base" if t == 0 else None)  # 找到 H_t 的评估作业名（t=0 兜底 base）
        if not inc_job:  # 找不到
            raise SystemExit(f"trajectory[{t}] has no job; cannot locate H_{t}'s evaluation")  # 报错
        inc_ev = EvalResult.load(self.eval_path(inc_job))  # 回读 H_t 评估
        S_star = max(x["S"] for x in fr["trajectory"][:t + 1])  # 重算截至 t 的 S*（历史最好）
        delta = self.delta()  # 当前噪声带（可能已改）
        rdir = self.runs / f"r{t}"  # 本轮目录
        cands = []  # 从磁盘重建候选
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir() and len(p.name) == 1):  # 单字母子目录 = 变体 A..H
            prep = (json.loads((vdir / "prep.json").read_text())  # 读候选的准备信息
                    if (vdir / "prep.json").exists() else {})
            c = Candidate(vdir.name, prep.get("edits") or [], diff_path=prep.get("diff_path"),  # 重建 Candidate
                          branch=prep.get("branch"), commit=prep.get("commit"),
                          gate_failure=prep.get("gate_failure"), detail=prep.get("detail", ""))
            if (vdir / "eval.json").exists():  # 有存储的评估
                c.ev = EvalResult.load(vdir / "eval.json")  # 回读（不重测）
            elif not c.gate_failure:  # 既无评估也无闸门失败记录
                c.gate_failure = "eval_invalid"  # 视为评估无效
            cands.append(c)  # 收集
        counts = {k: 0 for k in self.history.incumbent_component_counts()}  # 重建截至 t 的组件计数（排除 t 之后的接受）
        for r in self.history.records():  # 手工累加：t 轮之前被接受的记录
            if r.get("accepted") and r.get("t", 10**9) < t and r.get("component") in counts:
                counts[r["component"]] += 1
        winner, decisions = select_round(cands, inc_ev, S_star, delta, cfg, counts,  # 用新 delta/权重重新裁决
                                         guard_fn=d.guards)
        (rdir / "decisions.json").write_text(json.dumps(  # 覆写决策文件
            [x.to_json() for x in decisions], indent=1))
        self.history.replace_round(t)  # 删掉第 t 轮旧的历史记录
        for c, dec in zip(cands, decisions):  # 重新落账
            if c.ev is None:  # 无测量
                self.history.append_candidate(t, c.variant, c.edits, c.gate_failure or "not_evaluated",  # 记失败
                                              None, None, False, None, None, c.diff_path, c.detail)
                continue
            outcome = ("ACCEPTED" if c is winner else ("LOST" if dec.admissible else "REJECTED"))  # 结局重判
            self.history.append_candidate(t, c.variant, c.edits, outcome, dec.delta_S, dec.delta_C,  # 重写逐编辑记录
                                          c is winner, dec.S, dec.C, c.diff_path,
                                          f"[re-adjudicated delta={delta:.5f}] " + dec.reason)  # 理由前缀标明重裁
            log(d.name, f"r{t}{c.variant}: S={dec.S:.4f} dS={dec.delta_S:+.4f} dC={dec.delta_C:+.3f} "  # 汇报
                f"-> {outcome}: {dec.reason}")
        if winner is not None:  # 重裁后有获胜者
            r = G.git(self.repo, "merge-base", "--is-ancestor", inc_entry["commit"], winner.commit)  # 校验获胜提交确从 H_t 派生
            if r.returncode != 0:  # 不是后代
                raise SystemExit(f"{winner.commit} does not descend from H_{t} {inc_entry['commit']}")  # 拒绝（防历史分叉）
            G.git(self.repo, "update-ref", f"refs/heads/{self.branch}", winner.commit, check=True)  # 直接移动分支指针（worktree 已删，无法快进）
            new = {"t": t + 1, "commit": winner.commit, "harness_tree": self.harness_tree(winner.commit),  # 新现任条目
                   "job": winner.ev.job, "S": winner.ev.S, "C": winner.ev.C,
                   "extra": winner.ev.extra, "variant": winner.variant}
            fr["incumbent"] = new
            fr["S_star"] = max(S_star, winner.ev.S)  # S* 更新
            entry = {"t": t + 1, "S": winner.ev.S, "C": winner.ev.C, "commit": winner.commit,  # 轨迹条目
                     "job": winner.ev.job}
            log(d.name, f"re-adjudicated r{t}: ACCEPTED {winner.variant} -> {winner.commit} "  # 汇报
                f"S={winner.ev.S:.4f} S*={fr['S_star']:.4f}")
        else:  # 重裁后仍无获胜者
            G.git(self.repo, "update-ref", f"refs/heads/{self.branch}", inc_entry["commit"], check=True)  # 分支指回 H_t
            fr["incumbent"] = {"t": t + 1, "commit": inc_entry["commit"],  # 现任条目指向 H_t
                               "harness_tree": self.harness_tree(inc_entry["commit"]),
                               "job": inc_job, "S": inc_ev.S, "C": inc_ev.C, "extra": inc_ev.extra}
            fr["S_star"] = S_star  # S* 回退到截至 t 的值
            entry = {"t": t + 1, "S": inc_ev.S, "C": inc_ev.C, "commit": inc_entry["commit"],  # 轨迹条目
                     "job": inc_job}
            log(d.name, f"re-adjudicated r{t}: no admissible candidate; H_{t+1} = H_{t}")
        fr["trajectory"] = fr["trajectory"][:t + 1] + [entry]  # 截断轨迹到 t 并追加新条目
        self.save_frontier(fr)  # 落盘

    def reevaluate(self, t: int, variants: list[str] | None = None) -> None:  # 定义 reevaluate：基础设施故障后的重测 + 重裁
        """Re-measure the stored (committed) candidates of round t whose evaluation
        was invalid or corrupted by an infrastructure failure, then re-adjudicate
        the round. Rounds after t must have been removed first."""
        # docstring：重测第 t 轮中评估无效/被基础设施故障损坏的已提交候选，然后重新裁决该轮。
        # t 之后的轮必须先删掉。
        d = self.domain  # 局部别名
        rdir = self.runs / f"r{t}"  # 本轮目录
        ids = d.evolve_ids()  # evolve 集
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir() and len(p.name) == 1):  # 遍历变体目录
            if variants and vdir.name not in variants:  # 指定了变体子集
                continue  # 只处理指定的
            prep = (json.loads((vdir / "prep.json").read_text())  # 读准备信息
                    if (vdir / "prep.json").exists() else {})
            if not prep.get("commit") or not G.branch_exists(self.repo, prep.get("branch", "")):  # 没有已提交的候选可重测
                log(d.name, f"r{t}{vdir.name}: no committed candidate to re-evaluate")  # 跳过
                continue
            (vdir / "eval.json").unlink(missing_ok=True)  # 删掉可能损坏的旧评估
            wt = self.wt_root / f"r{t}{vdir.name}"  # 重建 worktree
            G.worktree_remove(self.repo, wt)
            G.git(self.repo, "worktree", "add", str(wt), prep["branch"], check=True)  # 检出候选分支
            c = Candidate(vdir.name, prep.get("edits") or [], diff_path=prep.get("diff_path"),  # 重建 Candidate
                          branch=prep["branch"], commit=prep["commit"])
            self._evaluate(t, c, rdir, ids)  # 重测（走同一套有效性检查）
            log(d.name, f"r{t}{vdir.name}: re-evaluated -> "  # 汇报重测结果
                + (f"S={c.ev.S:.4f} C={c.ev.C} missing={c.ev.missing}" if c.ev else
                   f"{c.gate_failure}: {c.detail}"))
            G.worktree_remove(self.repo, wt)  # 清理 worktree
        self.readjudicate(t)  # 最后重新裁决整轮

    # ------------------------------------------------------------ helpers --
    # —— 私有辅助：起草与评估 ——
    def _draft(self, t, vid, rdir, report, digests, traces, inc_ev, budget, explore,  # 定义 _draft：起草一个候选变体（propose->critic->smoke->commit）
               reserved, prune, hist_rows, skill_md, patterns_md) -> Candidate:
        d, cfg = self.domain, self.cfg  # 局部别名
        vdir = rdir / vid  # 本变体的产物目录 r<t>/<A>/
        vdir.mkdir(exist_ok=True)
        branch = f"{d.name}/r{t}{vid}"  # 候选分支名 <domain>/r<t><变体>
        wt = self.wt_root / f"r{t}{vid}"  # 候选 worktree 路径
        prep_path = vdir / "prep.json"  # 准备信息（断点续跑的锚）
        # resume: candidate already drafted and committed in an earlier attempt
        # 断点续跑：候选在更早的尝试中已起草并提交
        if prep_path.exists():  # 有准备记录
            prep = json.loads(prep_path.read_text())  # 读入
            if prep.get("gate_failure"):  # 之前已判定闸门失败
                return Candidate(vid, prep.get("edits") or [], gate_failure=prep["gate_failure"],  # 直接复用失败结论（不重复起草）
                                 detail=prep.get("detail", ""),
                                 diff_path=prep.get("diff_path"))
            if prep.get("commit") and G.branch_exists(self.repo, branch):  # 之前已提交且分支还在
                G.worktree_remove(self.repo, wt)
                G.git(self.repo, "worktree", "add", str(wt), branch, check=True)  # 重新挂载 worktree（评估用）
                log(d.name, f"{vid}: resuming committed candidate {prep['commit']}")
                return Candidate(vid, prep["edits"], diff_path=prep["diff_path"],  # 返回现成候选
                                 branch=branch, commit=prep["commit"])
        G.worktree_add(self.repo, wt, branch, self.branch)  # 全新起草：从演化分支头开新 worktree + 分支
        hdir = d.harness_dir(wt)  # worktree 内的 harness 目录
        variant_brief = (f"You are variant {vid} of round {t}. {cfg.m} variants are "  # 变体简报：独立起草、同台竞争
                         f"drafted independently from the same incumbent this round "
                         f"and each is evaluated on the full evolve set; the best "
                         f"admissible one becomes H_{t + 1}.")

        def finish(gate_failure: str, detail: str = "", edits=None, diff_path=None):  # 定义 finish：闸门失败的统一收尾
            prep_path.write_text(json.dumps(  # 写 prep.json 记录失败（续跑时不再重试）
                {"gate_failure": gate_failure, "detail": detail, "edits": edits or [],
                 "diff_path": diff_path}, ensure_ascii=False, indent=1))
            G.worktree_remove(self.repo, wt, branch)  # 删 worktree 与分支（失败候选不留痕）
            log(d.name, f"{vid}: {gate_failure} {detail[:200]}")  # 汇报
            return Candidate(vid, edits or [], gate_failure=gate_failure, detail=detail,  # 返回失败候选
                             diff_path=diff_path)

        prop = propose(d, hdir, report, hist_rows, skill_md, patterns_md, budget,  # 提案者在 worktree 内就地起草（Alg.1 主体）
                       explore, reserved, prune, traces=traces, per_task=inc_ev.per_task,
                       findings=digests, scoreboard=self.scoreboard(),
                       variant_brief=variant_brief, model=cfg.proposer_model)
        (vdir / "proposal.json").write_text(json.dumps(prop, ensure_ascii=False, indent=1))  # 提案落盘
        if prop["status"] != "done" or prop["n_edits"] == 0:  # 没交付或没实际改动
            return finish("no_proposal", str(prop.get("reason") or prop["status"]))  # 失败收尾

        # Critic(H_t, H') with bounded repair
        # critic 泄漏筛查 + 有界修复循环
        verdict, diff = None, ""  # 初始化裁决与 diff
        for attempt in range(1 + cfg.repair_rounds):  # 首审 + 至多 repair_rounds 次修复
            diff = G.diff_with_new_files(wt, self.harness_rel)  # 取工作区 diff（含新文件全文）
            (vdir / "diff.patch").write_text(diff)  # diff 落盘（审计）
            verdict = review(d, diff, prop.get("mechanism") or "",  # critic 审查
                             prop.get("targets_mode") or "", edits=prop.get("edits"),
                             model=cfg.critic_model)
            if verdict.get("verdict") == "accept":  # critic 通过
                # tag validation (l' must be evidenced by d'): a declared structural
                # component with nothing in the diff is re-tagged from the diff, and
                # a reserved exploration slot must still be honoured afterwards.
                # 标签校验（l' 必须有 d' 的证据）：声明为结构组件但 diff 里没有内容的，
                # 按 diff 重新打标签；保留探索槽位随后仍须被满足。
                tagged = [normalize(e.get("component"), diff, d.component_signals)  # 逐编辑按 diff 校正标签
                          for e in (prop.get("edits") or [])]
                # A declared tag without diff evidence is simply re-tagged to what
                # the diff reads as (that is what the history records and what the
                # novelty term counts); it is bounced only when the re-tag breaks a
                # reserved exploration slot, since arguing labels with the proposer
                # costs a candidate and buys nothing.
                # 无 diff 证据的声明标签直接改成 diff 实际读出的组件（历史记录与 novelty
                # 都以此为准）；只有当重打标签会破坏保留探索槽位时才打回——与提案者争论
                # 标签只会白赔一个候选。
                retagged = [(e.get("id"), str(e.get("component", "")).lower(), c)  # 收集被改标签的编辑
                            for e, c in zip(prop.get("edits") or [], tagged)
                            if c != str(e.get("component", "")).lower()]
                for eid, decl, real in retagged:  # 逐条汇报改标签
                    log(d.name, f"{vid}: edit {eid} declared '{decl}' but the diff reads as "
                        f"'{real}'; re-tagged")
                if reserved and explore.get("untried") and not any(  # 重打标签后槽位落空
                        c in explore["untried"] for c in tagged):
                    verdict = {"verdict": "reject",  # 改判拒绝（理由写明槽位要求）
                               "reasons": [f"this variant holds a RESERVED EXPLORATION SLOT: at "
                                           f"least one edit must be on a never-exercised component "
                                           f"from {explore['untried']}, judged by the DIFF, and "
                                           f"none is"],
                               "risk_notes": verdict.get("risk_notes")}
            (vdir / f"critic_a{attempt}.json").write_text(json.dumps(verdict, indent=1))  # 每次裁决落盘
            if verdict.get("verdict") == "accept" or attempt >= cfg.repair_rounds:  # 通过或修复次数用尽
                break  # 退出循环
            log(d.name, f"{vid}: critic objections (repair {attempt + 1}/"  # 汇报反对意见
                f"{cfg.repair_rounds}): {str(verdict.get('reasons'))[:200]}")
            prop = propose(d, hdir, report, hist_rows, skill_md, patterns_md, budget,  # 带着反对意见重新提案（修复轮）
                           explore, reserved, prune, traces=traces,
                           per_task=inc_ev.per_task, findings=digests,
                           scoreboard=self.scoreboard(), variant_brief=variant_brief,
                           repair_brief={"reasons": verdict.get("reasons"),
                                         "risk_notes": verdict.get("risk_notes"),
                                         "your_declared_edits": prop.get("edits")},
                           model=cfg.proposer_model)
            (vdir / f"proposal_r{attempt + 1}.json").write_text(  # 修复提案落盘
                json.dumps(prop, ensure_ascii=False, indent=1))
            if prop["status"] != "done":  # 修复提案也没交付
                break  # 放弃
        (vdir / "critic.json").write_text(json.dumps(verdict, indent=1))  # 最终裁决落盘
        edits = prop.get("edits") or []  # 最终编辑清单
        if not verdict or verdict.get("verdict") != "accept":  # 仍被拒
            return finish("critic_reject", str((verdict or {}).get("reasons"))[:600],  # 失败收尾
                          edits, str(vdir / "diff.patch"))

        # tag (l', h', d'): validate the declared component against the diff
        # 标签终校：声明组件对照 diff
        diff = G.diff_with_new_files(wt, self.harness_rel)  # 再取一次最终 diff
        (vdir / "diff.patch").write_text(diff)  # 落盘
        for e in edits:  # 逐编辑校正标签（以 diff 为准）
            e["component"] = normalize(e.get("component"), diff, d.component_signals)
        commit = G.commit_path(wt, self.harness_rel,  # 提交候选（只提交 harness 路径）
                               f"r{t}{vid}: {prop.get('mechanism')}")

        # liveness smoke: does the candidate run at all (not a selection rule)
        # 活性 smoke：候选到底能不能跑（不是选择规则，不参与打分）
        ok, detail = d.smoke(wt, self.runs, f"r{t}{vid}_smoke",  # 跑少量任务的存活检查
                             d.smoke_ids(inc_ev.per_task))
        (vdir / "smoke.json").write_text(json.dumps({"ok": ok, **detail}, indent=1))  # 结果落盘
        if not ok:  # 活性检查失败
            return finish("smoke_fail", json.dumps(detail)[:600], edits,  # 失败收尾
                          str(vdir / "diff.patch"))
        prep_path.write_text(json.dumps(  # 写 prep.json：候选已就绪（续跑锚点）
            {"commit": commit, "branch": branch, "edits": edits,
             "diff_path": str(vdir / "diff.patch"), "mechanism": prop.get("mechanism")},
            ensure_ascii=False, indent=1))
        log(d.name, f"{vid}: {len(edits)} edit(s) on "  # 汇报起草完成
            f"{[e['component'] for e in edits]} -> {commit}")
        return Candidate(vid, edits, diff_path=str(vdir / "diff.patch"),  # 返回就绪候选
                         branch=branch, commit=commit)

    def _evaluate(self, t: int, c: Candidate, rdir: Path, ids: list[str]) -> None:  # 定义 _evaluate：评估一个候选（含重试与有效性判定）
        d, cfg = self.domain, self.cfg  # 局部别名
        job = f"r{t}{c.variant}"  # 作业名
        ep = rdir / c.variant / "eval.json"  # 候选目录内的评估产物
        if ep.exists():  # 已有评估（断点续跑）
            c.ev = EvalResult.load(ep)  # 直接复用
            log(d.name, f"{c.variant}: reusing eval.json")
            return
        wt = self.wt_root / job  # 候选 worktree（_draft 已挂载）
        ev = None  # 评估结果
        for attempt in range(2):  # 至多两次尝试（基础设施抖动重试一次）
            try:  # 评估可能抛异常
                ev = evaluate(d, wt, self.runs, job, ids, cfg.k, log_prefix=job)  # 正式评估（domain runner 自动补缺 trial）
            except Exception as e:  # noqa: BLE001
                c.gate_failure, c.detail = "eval_invalid", f"evaluation crashed: {e!r}"[:600]  # 评估崩溃：记失败
                log(d.name, f"{c.variant}: {c.detail}")
                return
            if ev.missing <= cfg.invalid_missing_frac * ev.n_expected:  # 缺失比例可接受
                break  # 有效，退出重试
            log(d.name, f"{c.variant}: {ev.missing}/{ev.n_expected} trials missing "  # 缺失过多：报基础设施问题
                f"(infrastructure){'; retrying once' if attempt == 0 else ''}")
        if ev.missing > cfg.invalid_missing_frac * ev.n_expected:  # 重试后仍无效
            c.gate_failure = "eval_invalid"  # 判定失败（不进入选择）
            c.detail = f"{ev.missing}/{ev.n_expected} trials missing (infrastructure) after a retry"
            return
        ev.save(ep)  # 落盘到候选目录
        ev.save(self.eval_path(job))  # 同时落盘到 jobs/（续跑/readjudicate 用）
        c.ev = ev  # 挂到候选上
