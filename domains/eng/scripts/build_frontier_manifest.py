#!/usr/bin/env python3
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
# 模块说明:为 Frontier-Engineering 基准构建任务清单(manifest),供
# bench/run_tasks_frontier.py 与 bench/verify_frontier.py 读取。每个任务的
# frontier_eval/ 目录声明了候选文件、评测命令、工作目录与代理可见文件;但多数任务
# 的 frontier_eval/ 里并不包含验证器要用的 venv 名称——需要按上游 runner 的解析
# 顺序(env_name.txt > v1.yaml 精确 label > 域多数票 > conf/task/*.yaml > 全局默认)
# 推导。本脚本把这一解析一次性做掉并写进清单,让 runner 与评分器对"哪个任务用哪个
# venv"共享同一事实来源。
"""Build the Frontier-Eng task manifest read by bench/run_tasks_frontier.py and
bench/verify_frontier.py.

Every task of a Frontier-Engineering checkout
(https://github.com/EinsiaLab/Frontier-Engineering, `benchmarks/<Domain>/<Task>/`)
carries a `frontier_eval/` folder naming its candidate file, its eval command,
its working directory and the files the agent may see. The venv the verifier
must run under is NOT in that folder for most tasks: it is resolved the way the
benchmark's own runner resolves it, from `frontier_eval/conf/batch/v1.yaml`,
then per domain, then `frontier_eval/conf/task/*.yaml`, then the global default
`frontier-eval-driver`. This script does that resolution once and writes it
down, so the runner and the grader share one source of truth for which venv
grades which task.

Usage:
  python3 domains/eng/scripts/build_frontier_manifest.py \
      --frontier-root domains/eng/Frontier-Engineering \
      --out domains/eng/data/frontier_manifest.json [--tasks Domain/Task ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 全局默认验证器 venv 名称(与上游 frontier_eval/conf/task/unified.yaml 的默认一致)
_DEFAULT_ENV = "frontier-eval-driver"   # frontier_eval/conf/task/unified.yaml default


def _read(p: Path) -> str:
    # 读取文件全文并去首尾空白;任何失败(文件不存在等)都返回空串——缺文件是常态
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_lines(p: Path) -> list[str]:
    # 逐行读取:去空行与 # 注释行并 strip,用于解析 copy_files.txt 之类的清单
    return [ln.strip() for ln in _read(p).splitlines() if ln.strip() and not ln.strip().startswith("#")]


class EnvResolver:
    """Resolve a task's verifier venv (`env_name`) with transparent provenance:
    task env_name.txt > exact v1.yaml label > majority env of the domain in
    v1.yaml > conf/task/<name>.yaml > global default."""

    def __init__(self, froot: Path) -> None:
        # 记录 Frontier-Engineering 检出(checkout)的根目录
        self.froot = froot
        # 精确映射:"域/任务" label -> venv 名,来自 v1.yaml
        self.exact: dict[str, str] = {}
        # 域 -> 该域多数任务使用的 venv 名(多数票结果)
        self.domain_major: dict[str, str] = {}
        # 归一化名称 -> venv 名,来自 conf/task/*.yaml
        self.conf: dict[str, str] = {}
        # 读取批配置 v1.yaml 全文,准备按任务块解析
        txt = _read(froot / "frontier_eval" / "conf" / "batch" / "v1.yaml")
        # 每个域一个计数器,统计域内各 venv 出现次数,用于求多数票
        per_domain: dict[str, Counter] = defaultdict(Counter)
        # v1.yaml 中每个任务块以 "- name:" 开头,按它切块逐个解析
        for block in re.split(r"\n- name:", txt):
            # 块内必须带 label(即"域/任务"标识),否则跳过
            lm = re.search(r"label:\s*(\S+)", block)
            if not lm:
                continue
            label = lm.group(1).strip()
            # venv 名可能在 env_name= 或 python_path=uv-env: 两种写法中出现
            em = (re.search(r"env_name=(\S+)", block)
                  or re.search(r"python_path=uv-env:(\S+)", block))
            env = em.group(1).strip() if em else _DEFAULT_ENV
            if "/" in label:
                # 带域前缀的 label 记入精确映射,同时给所在域的计数器加一
                self.exact[label] = env
                per_domain[label.split("/")[0]][env] += 1
        # 对每个域取出现次数最多的 venv,作为该域的多数票结果
        for dom, c in per_domain.items():
            self.domain_major[dom] = c.most_common(1)[0][0]
        # 再解析 conf/task/*.yaml:提取每个文件的首个 name: 与末次 env_name: 字段
        for y in sorted((froot / "frontier_eval" / "conf" / "task").glob("*.yaml")):
            name, env = None, None
            for ln in _read(y).splitlines():
                m = re.match(r"\s*name:\s*(\S+)", ln)
                if m and name is None:
                    name = m.group(1).strip().strip("\"'")
                m = re.match(r"\s*env_name:\s*(\S+)", ln)
                if m:
                    env = m.group(1).strip().strip("\"'")
            if env:
                # 以文件名 stem 与 yaml 内 name 两种键(归一化后)登记该 venv
                for k in {y.stem, name}:
                    if k:
                        self.conf[re.sub(r"[^0-9a-z]+", "", k.lower())] = env

    def resolve(self, rel: Path) -> tuple[str, str]:
        # 返回 (venv 名, 来源说明);按五级优先级依次尝试,保证结果可追溯
        label = rel.as_posix()
        # 1) 任务自带 env_name.txt:优先级最高
        task_env = _read(self.froot / "benchmarks" / rel / "frontier_eval" / "env_name.txt")
        if task_env:
            return task_env, "task:env_name.txt"
        # 2) v1.yaml 中的精确 label 匹配
        if label in self.exact:
            return self.exact[label], f"v1:{label}"
        # 3) 该域在 v1.yaml 中的多数票 venv
        dom = rel.parts[0]
        if dom in self.domain_major:
            return self.domain_major[dom], f"v1-domain:{dom}"
        # 4) conf/task/*.yaml:依次用任务名/全路径/域名(归一化)查表
        for c in (rel.name, "_".join(rel.parts), dom):
            k = re.sub(r"[^0-9a-z]+", "", c.lower())
            if k in self.conf:
                return self.conf[k], f"conf:{c}"
        # 5) 全局默认 venv
        return _DEFAULT_ENV, "default"


def discover_tasks(froot: Path) -> list[Path]:
    """Task roots: the parent of each frontier_eval/ folder that names an eval command."""
    # 任务根 = benchmarks/ 下每个含 eval_command.txt 的 frontier_eval/ 目录的父目录
    return [fe.parent for fe in sorted((froot / "benchmarks").rglob("frontier_eval"))
            if fe.is_dir() and (fe / "eval_command.txt").exists()]


def main(argv: list[str]) -> int:
    # 命令行入口:解析参数
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontier-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset by rel path, e.g. KernelEngineering/MLA")
    args = ap.parse_args(argv)

    # 校验 Frontier-Engineering 检出根目录
    froot = args.frontier_root.expanduser().resolve()
    if not (froot / "benchmarks").is_dir():
        print(f"error: {froot} has no benchmarks/ dir", file=sys.stderr)
        return 2
    # 构建 venv 解析器并发现全部任务根
    resolver = EnvResolver(froot)
    roots = discover_tasks(froot)
    if args.tasks:
        # 指定了 --tasks 时只保留子集(按相对路径匹配)
        want = {t.strip("/") for t in args.tasks}
        roots = [r for r in roots if r.relative_to(froot / "benchmarks").as_posix() in want]

    # 逐任务构建清单条目
    entries = []
    for task_root in roots:
        # 任务相对路径与其 frontier_eval/ 目录
        rel = task_root.relative_to(froot / "benchmarks")
        fe = task_root / "frontier_eval"
        # 解析该任务验证器使用的 venv 及其来源
        env_name, env_src = resolver.resolve(rel)
        entries.append({
            # task_id:把相对路径压平成 frontier__域__任务 的合法标识符
            "task_id": re.sub(r"[^0-9A-Za-z_]+", "_", "frontier__" + "__".join(rel.parts)),
            # 域名取路径第一段
            "domain": rel.parts[0],
            "frontier": {
                "task_rel": rel.as_posix(),
                # 候选文件目的地,缺失时退回 initial_program.txt
                "candidate_destination": _read(fe / "candidate_destination.txt") or _read(fe / "initial_program.txt"),
                # 评测命令与工作目录(默认当前目录)
                "eval_command": _read(fe / "eval_command.txt"),
                "eval_cwd": _read(fe / "eval_cwd.txt") or ".",
                # 解析出的 venv 名与来源(供排查)
                "env_name": env_name,
                "env_name_source": env_src,
                # 允许代理看到的文件清单
                "copy_files": _read_lines(fe / "copy_files.txt"),
            },
        })
        # 逐任务打印解析结果
        print(f"[ok] {rel.as_posix():55s} env={env_name} ({env_src})")

    # 写出清单 JSON(确保父目录存在)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # 统计各 venv 被多少任务需要,打印汇总
    envs = Counter(e["frontier"]["env_name"] for e in entries)
    print(f"\nwrote {len(entries)} tasks -> {args.out}; venvs needed under {froot}/.venvs: "
          + ", ".join(f"{k} ({v})" for k, v in envs.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
