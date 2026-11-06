#!/usr/bin/env python3
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
# 模块说明:Harvey LAB 的进化/保留集(evolve/held-out)划分:120 个进化任务 +
# 40 个保留任务。划分文件不随仓库分发:首次需要时从 Harvey LAB 检出确定性生成并
# 写到 data/split_workspace.json。生成器是确定性的,同一检出必得同一划分;论文所用
# 检出以 rrsi.json 的 harvey_lab_commit 锁定,检出 commit 不同会告警——上游对判据
# 或文档的后续修改可能让任务跨过资格线或分层边界。
# 资格线:任务有判据与文档、判据数不超过 100(100+ 判据全过分必为零且是评审成本
# 炸弹)、文档包不超过 20MB。
# 分层:按执业领域(practice area)按比例分配;域内按判据数(交付复杂度的代理
# 指标)排序后等步长抽取,先抽保留集(使其覆盖复杂度全谱),进化集取其余。
"""Evolve / held-out split of Harvey LAB: 120 evolve tasks + 40 held-out tasks.

The split is not distributed with this repository. It is generated from a
Harvey LAB checkout (https://github.com/harveyai/harvey-labs) the first time
the workspace domain needs it and written to data/split_workspace.json. The
generator is deterministic, so one checkout always yields one split; the
checkout the paper's split was generated from is pinned in rrsi.json as
`harvey_lab_commit`, and a checkout at a different commit is reported as a
warning because later upstream edits to a task's criteria or documents can
move it across the eligibility or stratification boundaries.

Eligibility: the task has criteria and documents, at most 100 criteria (an
all-pass score on 100+ criteria is a guaranteed zero and a judge-cost bomb)
and a document bundle of at most 20MB.

Stratification: proportional per practice area; within an area tasks are
ordered by criteria count (a proxy for deliverable complexity) and picked by
even stride, held-out first (so the held-out set spans the complexity range),
evolve from the remainder.

Usage: HARVEY_LAB_ROOT=/path/to/harvey-labs python3 split_workspace.py
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

# 本目录与默认输出路径
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "split_workspace.json")
# 目标规模:进化 120、保留 40
N_EVOLVE, N_HELDOUT = 120, 40
# 资格上限:判据数 100、文档包 20MB
MAX_CRIT = 100
MAX_DOC_BYTES = 20 * 1024 * 1024


def _cfg() -> dict:
    # 读取本域配置 rrsi.json
    return json.load(open(os.path.join(HERE, "rrsi.json")))


def lab_root() -> str:
    # Harvey LAB 检出根:环境变量优先,其次配置,最后相对缺省
    return os.environ.get("HARVEY_LAB_ROOT", _cfg().get("harvey_lab_root") or "harvey-labs")


def checkout_commit(root: str) -> str:
    # 读取检出的当前 git commit;失败(非 git 仓库等)返回空串
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def stride_pick(items, n):
    # 等步长(even stride)抽取 n 个:在有序列表上均匀取索引,保证覆盖全谱
    if n >= len(items):
        return list(items), []
    # 计算等距索引;碰撞时用未被占用的最小索引补齐
    idx = {round(i * (len(items) - 1) / max(1, n - 1)) for i in range(n)}
    while len(idx) < n:
        idx.add(min(i for i in range(len(items)) if i not in idx))
    picked = [items[i] for i in sorted(idx)]
    rest = [x for i, x in enumerate(items) if i not in idx]
    return picked, rest


def build_split(root: str) -> tuple:
    """Return (split dict, per-area report) for the Harvey LAB checkout at root."""
    # 扫描 tasks/ 目录收集候选任务
    tasks_dir = os.path.join(root, "tasks")
    # 按执业领域(路径第一段)分组
    cands = defaultdict(list)
    for dirpath, _, filenames in os.walk(tasks_dir):
        # 必须同时有 task.json 与 documents/ 目录
        if "task.json" not in filenames or not os.path.isdir(os.path.join(dirpath, "documents")):
            continue
        rel = os.path.relpath(dirpath, tasks_dir)
        try:
            tj = json.load(open(os.path.join(dirpath, "task.json")))
        except Exception:
            continue
        crit = tj.get("criteria") or []
        # 资格过滤:有判据且不超过上限
        if not crit or len(crit) > MAX_CRIT:
            continue
        # 资格过滤:文档包大小不超过上限
        docs = os.path.join(dirpath, "documents")
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(docs) for f in fs)
        if size > MAX_DOC_BYTES:
            continue
        # 记录候选:任务相对路径与判据数
        cands[rel.split("/")[0]].append((rel, len(crit)))

    # 无任何合格任务则报错退出
    total = sum(len(v) for v in cands.values())
    if not total:
        raise SystemExit(f"no eligible Harvey LAB tasks under {tasks_dir}")
    split = {"evolve": [], "heldout": []}
    report = []
    for area in sorted(cands):
        # 域内按(判据数, 路径)排序——判据数是交付复杂度的代理指标
        pool = sorted(cands[area], key=lambda x: (x[1], x[0]))
        ids = [t for t, _ in pool]
        # 按域规模比例分配配额(保留集可为 0,进化集至少 1)
        q_ho = max(0, round(N_HELDOUT * len(pool) / total))
        q_ev = max(1, round(N_EVOLVE * len(pool) / total))
        # 先等步长抽保留集,再从其余抽进化集
        ho, rest = stride_pick(ids, q_ho)
        ev, _ = stride_pick(rest, q_ev)
        split["heldout"] += ho
        split["evolve"] += ev
        report.append(f"{area}: ev {len(ev)} ho {len(ho)} / {len(pool)}")

    # trim to exact sizes deterministically
    # 确定性地裁剪到精确规模(按 id 排序后截断)
    split["evolve"] = sorted(split["evolve"])[:N_EVOLVE]
    split["heldout"] = sorted(split["heldout"])[:N_HELDOUT]
    allids = split["evolve"] + split["heldout"]
    # 断言两集合无交集
    assert len(allids) == len(set(allids)), "overlap between evolve and heldout"
    out = {
        "mode": "workspace_train_v1",
        "tasks": split,
        "smoke": split["evolve"][:2],   # cheap pipe checks come from the evolve set
        "meta": {"n": {k: len(v) for k, v in split.items()},
                 "harvey_lab_commit": checkout_commit(root)},
    }
    return out, report


def write_split(path: str = OUT, root=None) -> dict:
    """Generate the split from the checkout at root (default: HARVEY_LAB_ROOT) and write it."""
    # 检出根缺省取 lab_root()
    root = root or lab_root()
    # 校验检出存在
    if not os.path.isdir(os.path.join(root, "tasks")):
        raise SystemExit(f"no Harvey LAB checkout at {root} (set HARVEY_LAB_ROOT)")
    # 检出 commit 与论文锁定 commit 不一致时告警(上游编辑可能移动分层边界)
    pin, head = _cfg().get("harvey_lab_commit") or "", checkout_commit(root)
    if pin and head and head != pin:
        print(f"[split_workspace] warning: checkout is at {head[:9]}; the split reported in "
              f"the paper was generated at {pin[:9]}", file=sys.stderr)
    # 构建划分并写出
    out, report = build_split(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    # 打印元数据与各域分配报告
    print(json.dumps(out["meta"], indent=1))
    print("; ".join(report))
    return out


if __name__ == "__main__":
    write_split()
