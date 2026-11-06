#!/usr/bin/env python3
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
# 模块说明:为工程设计(engineering-design)实例构建 EngDesign 基准树。对官方
# EngDesign-Open 检出的每个任务目录,生成 <out>/benchmarks/<EngDesignOpen_域>/<任务>/,
# 内含完整上游任务(验证器要加载 evaluate.py 与 output_structure.py)、种子候选
# submission/payload.py,以及 bench/verify.py 评分所需的 frontier_eval/ 接线:
# run_eval.sh 调用 evaluate_submission.py,后者把 payload 解析成字面量、实例化任务的
# Response_structure 并运行任务自带的 evaluate_llm_response。copy_files.txt 把代理
# 能看到的文件限制在题面、响应 schema、payload 存根和图片;评分器/答案/量规虽保留在
# 树里但绝不进入工作区。需要 MATLAB 或 Webots 的任务被跳过(67 个中的 6 个),
# 剩下 data/split_engd.json 中的 61 个任务。
"""Build the EngDesign benchmark tree used by the engineering-design instance.

For every task directory of an official EngDesign-Open checkout
(https://github.com/AGI4Engineering/EngDesign, `EngDesign-Open/<task_id>/`) this
writes `<out>/benchmarks/<EngDesignOpen_Domain>/<task_id>/` holding the full
upstream task (so the verifier can load `evaluate.py` and
`output_structure.py`), a seed candidate `submission/payload.py`, and a
`frontier_eval/` folder with the grading wiring that bench/verify.py runs:
`run_eval.sh` calls `evaluate_submission.py`, which parses the payload as a
literal, instantiates the task's `Response_structure` and runs the task's own
`evaluate_llm_response`. `copy_files.txt` restricts what the AGENT sees to the
prompt, the response schema, the payload stub and the images; graders,
solutions and rubrics stay in the tree but are never copied into a workspace.

Tasks that need MATLAB or Webots are skipped (six of the 67 EngDesign-Open
tasks), which leaves the 61 tasks of data/split_engd.json.

Usage:
  python3 domains/eng/scripts/engdesign/build_engdesign_bench.py \
      --engdesign-open /path/to/EngDesign/EngDesign-Open \
      --out domains/eng/engdesign_bench [--tasks AB_01 XY_01] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# 本脚本同目录下放着 evaluate_submission.py 与 run_eval.sh,构建时复制进每个任务
_ASSETS = Path(__file__).resolve().parent   # evaluate_submission.py and run_eval.sh live alongside

# never copied into the generated task dir at all
# 生成任务目录时完全不复制的内容
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
# present in the task dir (grader may need none of these) but WITHHELD from the
# agent's container via copy_files — these leak the answer / rubric. Matched by
# exact name or glob; tolerant of upstream's filename variants.
# NB: output_structure.py is NOT withheld — it is the response *schema* (field
# names/types the answer must conform to), leaks nothing about the solution, and
# upstream EngDesign gives it to the model. Withholding it forced the agent to
# guess field names. The grader/solution/rubric files below stay withheld.
# 不复制进任务目录、且通过 copy_files 对代理隐藏的文件(会泄露答案/量规):
# 评分器、参考答案、rubric 文件、日志与 pyc;支持精确名与 glob,兼容上游文件名变体。
# 注意 output_structure.py 不隐藏——它是响应 schema(答案须符合的字段名/类型),
# 不泄露解法,且上游 EngDesign 本来就提供给模型;隐藏它会逼代理瞎猜字段名。
_WITHHOLD_GLOBS = (
    "evaluate.py", "run_eval.py",
    "solution.*", "reference.txt", "references.txt",
    "rubric.txt", "rubrics.txt", "rubics.txt",
    "logs", "logs/*", "*.pyc",
)

# Tasks that cannot be graded on a headless host without proprietary software.
# They are recorded in the manifest under "skipped".
# 无头服务器上没有专有软件就无法评分的任务;会记录在清单的 "skipped" 里
_SKIP_TASKS = {
    "RS_01": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "RS_02": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "RS_03": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "Ziheng_02": "MATLAB-typed answers (matlab.double); baseline logs score 0",
    "XZ_03": "needs Webots simulator (subprocess) — not installable headless",
    "XZ_04": "needs Webots simulator (subprocess) — not installable headless",
}

# 种子候选 payload 模板:EVOLVE-BLOCK 内是进化算法可以改写的初始提交
_SEED_PAYLOAD = '''\
"""EngDesign submission for {task_id}.

Define a module-level dict `PAYLOAD` whose keys are the kwargs of the task's
`Response_structure` (see output_structure.py for the exact fields/nesting —
some tasks nest the answer under `config`, others are flat). `PAYLOAD` MUST be a
pure literal (only numbers/strings/lists/dicts/bools/None) — the grader parses
it with ast.literal_eval, so NO imports, function calls, comprehensions, or file
I/O. Only this file is graded.
"""

# EVOLVE-BLOCK-START
PAYLOAD = {{
    "reasoning": "TODO: solve the task described in LLM_prompt.txt",
    # TODO: add the remaining fields required by Response_structure
    # (see output_structure.py — provide literal values only).
}}
# EVOLVE-BLOCK-END
'''


# leading boilerplate labels the upstream files prepend before the real topic,
# e.g. "Main domain topic: Operating system" / "Domain: Computer Architecture"
# 上游题面在真实主题前会加的套话标签,如 "Main domain topic: Operating system" /
# "Domain: Computer Architecture",抽域时要去掉
_LABEL_RE = re.compile(r"^\s*(main\s+)?(domain(\s+topic)?|topic)\s*:\s*", re.I)


def _domain(task_dir: Path, _allow_sibling: bool = True) -> str:
    # 从 domain_topic.txt 推导粗粒度域名并压成 EngDesignOpen_<Slug> 形式
    p = task_dir / "domain_topic.txt"
    first = ""
    if p.is_file():
        # 取文件第一个非空行作为主题描述
        for ln in p.read_text("utf-8", "ignore").splitlines():
            if ln.strip():
                first = ln.strip()
                break
    if not first:
        # no domain_topic.txt — borrow from a sibling sharing the author prefix
        # (e.g. JY_02/JY_03 inherit JY_01's "Computer Vision")
        # 没有 domain_topic.txt 时,向共享作者前缀的兄弟任务借用
        # (如 JY_02/JY_03 继承 JY_01 的 "Computer Vision")
        if _allow_sibling and "_" in task_dir.name:
            prefix = task_dir.name.rsplit("_", 1)[0]
            for sib in sorted(task_dir.parent.iterdir()):
                if (sib.is_dir() and sib != task_dir
                        and sib.name.rsplit("_", 1)[0] == prefix
                        and (sib / "domain_topic.txt").is_file()):
                    # 递归借用时禁止再次借(防连锁),借用结果若仍是兜底值则继续找
                    d = _domain(sib, _allow_sibling=False)
                    if d != "EngDesignOpen_Engdesign":
                        return d
        # 实在拿不到就用兜底值
        first = "EngDesign"
    # drop a leading label ("Main domain topic:" etc.) so we keep the value
    # 去掉开头的标签("Main domain topic:" 等),只保留取值部分
    first = _LABEL_RE.sub("", first).strip() or first
    # coarse domain = text before the first real separator: ':' '/' '(' ',' or a
    # space-flanked dash. A bare hyphen ("Multi-Agent", "Anti-Aliasing") is kept.
    # 粗粒度域 = 第一个真实分隔符之前的部分:':' '/' '(' ',' 或带空格的连字符;
    # 纯连字符词("Multi-Agent"、"Anti-Aliasing")要保留
    coarse = re.split(r"[:/(,]|\s-\s|\s-|-\s", first, 1)[0].strip() or first
    # 压成标题式 slug,非字母数字字符全部去除
    slug = re.sub(r"[^0-9A-Za-z]+", "", coarse.title())
    return f"EngDesignOpen_{slug}" if slug else "EngDesignOpen"


def _is_withheld(rel: str) -> bool:
    # 判断相对路径是否命中隐藏规则:整路径或首段目录名匹配任一 glob 即隐藏
    from fnmatch import fnmatch
    return any(fnmatch(rel, g) or fnmatch(rel.split("/")[0], g) for g in _WITHHOLD_GLOBS)


def _copy_task_tree(src: Path, dst: Path) -> None:
    # 复制任务目录树:跳过 _SKIP_DIRS 中的目录,只复制文件并保留元数据
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.is_dir():
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def _agent_visible(src: Path) -> list[str]:
    """Files shipped into the agent's container: the prompt, images, and the
    seed candidate. Everything else (graders, solutions, rubrics) is withheld."""
    # 代理容器里只放:题面、响应 schema、种子候选 payload
    out = ["LLM_prompt.txt", "output_structure.py", "submission/payload.py"]
    img = src / "images"
    if img.is_dir():
        # 有图片目录则一并放入
        out.append("images")
    return out


def build_one(task_dir: Path, out_root: Path, env_name: str, dry: bool) -> dict:
    # 构建单个任务:返回信息 dict,dry 模式只报告不落盘
    task_id = task_dir.name
    # 推导域并拼出基准内相对路径
    domain = _domain(task_dir)
    rel = f"{domain}/{task_id}"
    dst = out_root / "benchmarks" / domain / task_id
    info = {"task_id": task_id, "domain": domain, "rel": rel,
            "withheld": [], "agent_files": _agent_visible(task_dir)}

    # what would be withheld (for reporting / the README)
    # 统计将被隐藏的文件清单(用于报告/README)
    info["withheld"] = sorted({
        r.relative_to(task_dir).as_posix() for r in task_dir.rglob("*")
        if r.is_file() and _is_withheld(r.relative_to(task_dir).as_posix())
    })

    if dry:
        # dry-run:只报告不写任何文件
        return info

    if dst.exists():
        # 目标已存在则先删掉,保证幂等重建
        shutil.rmtree(dst)
    _copy_task_tree(task_dir, dst)

    # surface the task text as Task.md for tooling that reads Task.md/README.md
    # rather than LLM_prompt.txt
    # 把题面同步成 Task.md,方便只认 Task.md/README.md 的工具
    prompt_txt = (task_dir / "LLM_prompt.txt")
    if prompt_txt.is_file():
        (dst / "Task.md").write_text(prompt_txt.read_text("utf-8", "ignore"), encoding="utf-8")

    # seed candidate
    # 写种子候选 payload(EVOLVE-BLOCK 内为初始可进化提交)
    sub = dst / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "payload.py").write_text(_SEED_PAYLOAD.format(task_id=task_id), encoding="utf-8")

    # frontier_eval metadata
    # 写 frontier_eval/ 元数据:候选目的地、初始程序、评测命令、工作目录
    fe = dst / "frontier_eval"
    fe.mkdir(parents=True, exist_ok=True)
    (fe / "candidate_destination.txt").write_text("submission/payload.py\n")
    (fe / "initial_program.txt").write_text("submission/payload.py\n")
    (fe / "eval_command.txt").write_text("bash frontier_eval/run_eval.sh {python} {benchmark} {candidate}\n")
    (fe / "eval_cwd.txt").write_text(".\n")
    # 代理可见文件清单与验证器 venv 名
    (fe / "copy_files.txt").write_text("\n".join(_agent_visible(task_dir)) + "\n")
    (fe / "env_name.txt").write_text(env_name + "\n")
    # 复制评测适配器与启动脚本,并赋予执行权限
    shutil.copy2(_ASSETS / "evaluate_submission.py", fe / "evaluate_submission.py")
    shutil.copy2(_ASSETS / "run_eval.sh", fe / "run_eval.sh")
    (fe / "run_eval.sh").chmod(0o755)
    return info


def main(argv: list[str]) -> int:
    # 命令行入口:解析参数
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engdesign-open", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--env-name", default="engdesign")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # 校验上游 EngDesign-Open 源目录
    src_root = args.engdesign_open.expanduser().resolve()
    if not src_root.is_dir():
        print(f"error: {src_root} not a dir", file=sys.stderr)
        return 2
    out_root = args.out.expanduser().resolve()

    # 任务目录 = 源根下带 evaluate.py 的非 __ 开头目录
    task_dirs = [d for d in sorted(src_root.iterdir())
                 if d.is_dir() and not d.name.startswith("__")
                 and (d / "evaluate.py").is_file()]
    if args.tasks:
        # 指定 --tasks 时只保留子集
        want = {t.strip("/") for t in args.tasks}
        task_dirs = [d for d in task_dirs if d.name in want]
    if not task_dirs:
        print("no matching tasks", file=sys.stderr)
        return 2

    # 先摘出需要专有软件、无法评分的任务并打印跳过原因
    skipped = [{"task_id": d.name, "reason": _SKIP_TASKS[d.name]}
               for d in task_dirs if d.name in _SKIP_TASKS]
    task_dirs = [d for d in task_dirs if d.name not in _SKIP_TASKS]
    for s in skipped:
        print(f"[skip] {s['task_id']:43s} {s['reason']}")

    # 逐任务构建(或 dry-run 报告)
    infos = []
    for d in task_dirs:
        info = build_one(d, out_root, args.env_name, args.dry_run)
        infos.append(info)
        tag = "[dry]" if args.dry_run else "[ok]"
        print(f"{tag} {info['rel']:48s} agent_files={info['agent_files']} "
              f"withheld={len(info['withheld'])}")

    if not args.dry_run:
        # 写清单:env 名、任务信息、跳过原因
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "engdesign_manifest.json").write_text(
            json.dumps({"env_name": args.env_name, "tasks": infos,
                        "skipped": skipped}, indent=2), encoding="utf-8")
        # 汇总输出
        print(f"\nwrote {len(infos)} tasks ({len(skipped)} skipped) -> {out_root}/benchmarks/")
        print(f"manifest -> {out_root}/engdesign_manifest.json")
        print(f"\nNOTE: the verifiers run under the grading venv .venvs/{args.env_name} "
              f"(see domains/eng/README.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
