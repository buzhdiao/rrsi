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
# 模块说明:单任务 EngDesign 验证器适配器。由 build_engdesign_bench.py 放进每个
# 生成任务的 frontier_eval/ 目录,由 run_eval.sh 携代理候选调用;写出带顶层
# combined_score 与 valid 的 metrics.json 供 bench/verify.py 读取。
# 它适配的上游契约(EngDesign 每任务目录):
#   * output_structure.py: pydantic 模型 Response_structure(reasoning, config, ...)
#   * evaluate.py: evaluate_llm_response(resp) -> (passed, details, score, score_max_or_conf)
# 代理候选是定义 dict PAYLOAD(别名 SUBMISSION/submission)的小 python 文件,
# 即 Response_structure 的 kwargs,如 PAYLOAD = {"reasoning": "...", "config": {...}}
# 归一化计分(跨任务可比):combined_score = score/score_max 截到 [0,1];
# valid:评测无硬错误为 1.0,否则 0.0。score_max 取第四个返回值(当它像上限:
# 数值、>= score、> 1),否则退回量规解析,再退回 100。
"""Single-task EngDesign verifier adapter.

Dropped into each generated task's `frontier_eval/` by build_engdesign_bench.py
and invoked by `run_eval.sh` with the agent's candidate. It writes `metrics.json`
with a top-level `combined_score` and `valid`, which bench/verify.py reads.

Contract it adapts (upstream EngDesign, per task dir):
  * output_structure.py : pydantic `Response_structure(reasoning, config, ...)`
  * evaluate.py         : `evaluate_llm_response(resp) -> (passed, details, score, score_max_or_conf)`

The agent's candidate is a small python file defining a dict `PAYLOAD` (aliases:
`SUBMISSION`, `submission`) = the `Response_structure` kwargs, e.g.
    PAYLOAD = {"reasoning": "...", "config": {...}}

Scoring (normalized, cross-task-comparable):
  combined_score = score / score_max   (clamped to [0,1])
  valid          = 1.0 if it evaluated without a hard error, else 0.0
score_max is the 4th return value when it looks like a max (numeric, >= score,
> 1), else falls back to a rubric parse, else 100.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


def _tail(text: str, limit: int = 12000) -> str:
    # 保留文本尾部:超长时只留最后 limit 字符(错误信息通常在末尾)
    return text if len(text) <= limit else text[-limit:]


def _write_json(path: Path, obj: Any) -> None:
    # 把对象写成 JSON 文件:先建父目录,default=str 兜底不可序列化对象
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


@contextlib.contextmanager
def _pushd(path: Path):
    # 临时切换工作目录的上下文管理器:评测脚本常按相对路径读任务文件
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        # 无论成败都恢复原目录
        os.chdir(old)


def _load_module(name: str, path: Path, extra_paths: list[Path]) -> Any:
    # 按文件路径加载 python 模块:临时把 extra_paths 插到 sys.path 最前,加载完恢复
    original = list(sys.path)
    # 记录同名模块的旧状态,加载后按需还原,避免污染 sys.modules
    prev = sys.modules.get(name)
    try:
        sys.path = [str(p) for p in extra_paths] + original
        # 由文件路径构造模块 spec
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec: {path}")
        # 创建模块对象并注册进 sys.modules,再真正执行模块代码
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # 恢复 sys.path 与 sys.modules 的原状
        sys.path = original
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def _load_payload(candidate_path: Path) -> dict[str, Any]:
    """Extract the PAYLOAD dict WITHOUT executing the candidate.

    The candidate is graded in a sandbox that also holds the task's evaluate.py /
    solution.txt, so executing it (runpy/exec) would let a candidate import or
    open those to recover the gold answer (verified leak). We therefore parse the
    file and literal-eval ONLY the assigned value — ast.literal_eval runs no
    imports, calls, or I/O, so it cannot read the grader/solution files.
    Candidates must provide literal values (numbers, strings, lists, dicts).
    """
    # 读候选源码并解析成 AST(不执行!)
    src = candidate_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # 允许的顶层变量名(主名 + 别名)
    keys = ("PAYLOAD", "SUBMISSION", "submission", "RESPONSE", "response")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in keys:
                    # 只对赋值右值做 literal_eval:纯字面量求值,不会执行 import/
                    # 函数调用/IO,因此候选无法借机读走评分器/答案文件
                    val = ast.literal_eval(node.value)  # literals only — safe
                    if isinstance(val, dict):
                        return val
    # 找不到合法字面量 dict 则报错
    raise ValueError("candidate must define a LITERAL dict named PAYLOAD "
                     "(no imports/calls/file I/O; literal values only)")


def _normalize_payload(section: dict[str, Any]) -> dict[str, Any]:
    """Pass PAYLOAD through as Response_structure kwargs, essentially untouched.

    EngDesign schemas are heterogeneous: some nest the answer under `config`
    (e.g. AB_01's RadiomicsOutput), some are flat (NS_PA_SS_* = reasoning + a
    `code` string), and some carry extra required top-level siblings beyond
    config (XY_01 = config + tetromino_pattern + ...). An earlier version forced
    every payload into a fixed `{reasoning, config}` shape, which dropped the
    siblings and broke the flat schemas outright. We now only guarantee a
    `reasoning` key and hand everything else to the task's own schema, which is
    the single source of truth for field names/nesting.
    """
    # PAYLOAD 必须是 dict(即 Response_structure 的 kwargs)
    if not isinstance(section, dict):
        raise TypeError("PAYLOAD must be a dict of Response_structure kwargs")
    # 浅拷贝后只保证有 reasoning 键:字段名/嵌套结构以任务自身 schema 为唯一真相
    payload = dict(section)
    payload.setdefault("reasoning", "")
    return payload


def _safe_float(v: Any) -> float | None:
    # 宽松转 float:bool 映射 1.0/0.0,数值直接转,数字字符串 strip 后转,其余 None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _rubric_max(benchmark_dir: Path) -> float | None:
    """Best-effort max-points parse from rubrics.txt / rubics.txt."""
    # 尽力从量规文件解析满分:匹配 "N points/pts/分" 模式
    for name in ("rubrics.txt", "rubics.txt", "rubric.txt"):
        p = benchmark_dir / name
        if not p.is_file():
            continue
        # 提取所有分值数字
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:points|pts|分)", p.read_text("utf-8", "ignore"), re.I)]
        if nums:
            # 多个子项则求和,单个则直接用
            return sum(nums) if len(nums) > 1 else nums[0]
    return None


def _resolve_max(score: float | None, fourth: Any, benchmark_dir: Path) -> float:
    # 解析满分(score_max):优先用 evaluate 的第四个返回值(当它像上限)
    f = _safe_float(fourth)
    if f is not None and f > 1 and (score is None or f >= score):
        return f
    # 否则退回量规文件解析出的满分
    rm = _rubric_max(benchmark_dir)
    if rm and rm > 0:
        return rm
    # 都没有则用默认 100
    return 100.0


def main() -> int:
    # 命令行入口:候选路径、任务目录、输出文件与超时
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--benchmark-dir", default=".")
    ap.add_argument("--metrics-out", default="metrics.json")
    ap.add_argument("--artifacts-out", default="artifacts.json")
    ap.add_argument("--task-timeout-s", default=180.0, type=float)
    args = ap.parse_args()

    # 规范化各路径:相对路径一律锚定在任务目录上
    bdir = Path(args.benchmark_dir).expanduser().resolve()
    cand = Path(args.candidate).expanduser()
    cand = cand if cand.is_absolute() else (bdir / cand)
    metrics_out = Path(args.metrics_out)
    metrics_out = metrics_out if metrics_out.is_absolute() else (bdir / metrics_out)
    artifacts_out = Path(args.artifacts_out)
    artifacts_out = artifacts_out if artifacts_out.is_absolute() else (bdir / artifacts_out)

    # 指标的失败缺省值:0 分且无效
    metrics: dict[str, Any] = {"combined_score": 0.0, "valid": 0.0}
    # 产物记录:候选与任务目录路径
    artifacts: dict[str, Any] = {"candidate_path": str(cand), "benchmark_dir": str(bdir)}

    try:
        # 无执行地取出 PAYLOAD 字面量并规范化成 schema kwargs
        payload = _normalize_payload(_load_payload(cand))
        # 加载任务的响应 schema 模块与评测模块
        out_mod = _load_module("engdesign_output", bdir / "output_structure.py", [bdir])
        eval_mod = _load_module("engdesign_evaluate", bdir / "evaluate.py", [bdir])
        # 校验两个模块暴露了必需的入口
        if not hasattr(out_mod, "Response_structure"):
            raise AttributeError("output_structure.py has no Response_structure")
        if not hasattr(eval_mod, "evaluate_llm_response"):
            raise AttributeError("evaluate.py has no evaluate_llm_response")

        # 用 PAYLOAD 作为 kwargs 实例化任务的响应结构
        resp = out_mod.Response_structure(**payload)
        # 捕获评测过程的 stdout/stderr,并在任务目录内运行
        sout, serr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(sout), contextlib.redirect_stderr(serr), _pushd(bdir):
            ret = eval_mod.evaluate_llm_response(resp)

        # 上游返回值为 (passed, details, score, score_max_or_conf):不足四个补 None
        passed, details, score, fourth = (list(ret) + [None] * 4)[:4]
        # 分数转 float;解析满分
        score_f = _safe_float(score)
        score_max = _resolve_max(score_f, fourth, bdir)
        # 归一化到 [0,1] 的跨任务可比分数
        combined = 0.0 if score_f is None else max(0.0, min(1.0, score_f / score_max))

        # 写最终指标
        metrics = {
            "combined_score": combined,
            "valid": 1.0 if score_f is not None else 0.0,
            "passed": bool(passed),
            "raw_score": score_f,
            "score_max": score_max,
        }
        # 记录评测产物:details、原始返回值与评测日志尾部
        artifacts.update({
            "details": details, "raw_return": [str(passed), str(score), str(fourth)],
            "eval_stdout": _tail(sout.getvalue()), "eval_stderr": _tail(serr.getvalue()),
        })
    except Exception as exc:  # noqa: BLE001
        # 任何异常都记成 0 分且无效,并保留错误与 traceback 供诊断
        metrics = {"combined_score": 0.0, "valid": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        artifacts["traceback"] = _tail(traceback.format_exc())

    # 落盘 metrics 与 artifacts,并向 stdout 打印摘要
    _write_json(metrics_out, metrics)
    _write_json(artifacts_out, artifacts)
    print(json.dumps({k: metrics.get(k) for k in ("combined_score", "valid", "passed", "error")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
