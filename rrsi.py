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

#!/usr/bin/env python3
# shebang：允许 ./rrsi.py 直接执行
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
"""RRSI command line.

  python3 rrsi.py --domain eng baseline                # evaluate H_0, seed the frontier
  python3 rrsi.py --domain eng calibrate [--jobs base,base2]   # noise band delta
  python3 rrsi.py --domain eng round --t 3 [--dry-run] # one round (Alg. 1 + Alg. 2)
  python3 rrsi.py --domain eng run [--start 0]         # driver: rounds until T
  python3 rrsi.py --domain eng readjudicate --t 3      # re-apply Alg. 2 to round 3's stored measurements
  python3 rrsi.py --domain eng reevaluate --t 3        # re-measure round 3's candidates (infra failure), then re-adjudicate
  python3 rrsi.py --domain workspace heldout --label champ [--ref <commit>] [--set heldout]
  python3 rrsi.py --domain eng smoke                   # liveness check of the incumbent
  python3 rrsi.py --domain eng status                  # frontier + history summary

Hyperparameters come from domains/<domain>/rrsi.json; any of them can be
overridden on the command line (--T, --k, --m, --b-min, --b-max, --w,
--m-draft, --delta, --beta0, --beta1, --w-s, --w-c, --w-n, --n-prune).
"""
# 【模块说明（中文）】RRSI 命令行入口。子命令一览：
#   baseline=评估 H_0 并播种 frontier；calibrate=估噪声带 delta；
#   round=跑一轮（Alg.1 + Alg.2）；run=驱动器连续跑到 T 轮；
#   readjudicate=对存储测量重跑 Alg.2；reevaluate=重测某轮候选后重裁；
#   heldout=在 held-out/OOD 集上评估；smoke=现任活性检查；status=状态摘要。
# 超参数来自 domains/<domain>/rrsi.json，全部可在命令行覆盖。

# argparse：命令行解析
import argparse
# json：状态输出的序列化
import json
# sys：路径注入与退出码
import sys
# Path：路径对象
from pathlib import Path

# ROOT：仓库根（本文件的父目录）
ROOT = Path(__file__).resolve().parent
# 把仓库根塞进 sys.path：允许从任意工作目录 `python3 /path/to/rrsi.py` 运行
sys.path.insert(0, str(ROOT))

# gitops（别名 G）：git 状态查询（status 子命令用）
from rrsi import gitops as G                 # noqa: E402
# RRSIConfig：超参数加载
from rrsi.config import RRSIConfig          # noqa: E402
# load_domain：动态加载 domain 适配器
from rrsi.domain import load_domain          # noqa: E402
# drive：顺序驱动器
from rrsi.driver import drive                # noqa: E402
# Run：主循环宿主（全部子命令的核心）
from rrsi.loop import Run                    # noqa: E402
# budget_table：退火调度表（status 展示用）
from rrsi.schedule import budget_table       # noqa: E402
# （上面这些 import 在 sys.path 注入之后，故标注 noqa: E402 关闭"模块级 import 不在文件头"的 lint）

# OVERRIDES：可在命令行覆盖的超参数名清单（与 RRSIConfig 字段一一对应）
OVERRIDES = ["T", "k", "m", "b_min", "b_max", "w", "m_draft", "delta", "delta_z",
             "beta0", "beta1", "w_s", "w_c", "w_n", "n_prune", "eval_parallel"]


def main():  # 定义 main：CLI 主函数
    ap = argparse.ArgumentParser(description=__doc__,  # 帮助文本直接用模块 docstring
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # 保留原文格式
    ap.add_argument("--domain", required=True)  # domain 名（coding/workspace/eng）必填
    ap.add_argument("--runs", default=str(ROOT / "runs"))  # 运行根目录，默认 <repo>/runs
    for name in OVERRIDES:  # 为每个可覆盖超参数生成命令行开关
        typ = int if name in ("T", "k", "m", "b_min", "b_max", "w", "m_draft",  # 整型参数集合
                              "n_prune", "eval_parallel") else float  # 其余为浮点
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)  # --b-min 风格命名；None 表示未提供
    sub = ap.add_subparsers(dest="cmd", required=True)  # 子命令（必选）
    sub.add_parser("baseline").add_argument("--job", default="base")  # baseline：可指定作业名
    sub.add_parser("calibrate").add_argument("--jobs", default="base")  # calibrate：可给多个评估作业（逗号分隔）
    p = sub.add_parser("round")  # round：单轮
    p.add_argument("--t", type=int, required=True)  # 轮号必填
    p.add_argument("--dry-run", action="store_true")  # 演练模式：只做分析
    sub.add_parser("run").add_argument("--start", type=int, default=0)  # run：起始轮
    sub.add_parser("readjudicate", help="re-run Algorithm 2 on a stored round after a "  # readjudicate：重裁
                   "delta / weight change (no new evaluation)").add_argument("--t", type=int, required=True)
    p = sub.add_parser("reevaluate", help="re-measure a round's committed candidates after an "  # reevaluate：重测+重裁
                       "infrastructure failure, then re-adjudicate")
    p.add_argument("--t", type=int, required=True)  # 轮号
    p.add_argument("--variants", default="", help="e.g. A,B (default: all)")  # 可指定变体子集
    p = sub.add_parser("heldout")  # heldout：held-out/OOD 评估
    p.add_argument("--label", required=True)  # 标签（作业名后缀）
    p.add_argument("--ref", default=None, help="commit/branch to evaluate (default incumbent)")  # 被评估的提交/分支
    p.add_argument("--set", default="heldout", choices=["heldout", "evolve"])  # 任务集选择
    sub.add_parser("smoke")  # smoke：活性检查
    sub.add_parser("status")  # status：状态摘要
    args = ap.parse_args()  # 解析

    domain = load_domain(args.domain)  # 动态加载 domain 适配器
    cfg = RRSIConfig.load(domain.root / "rrsi.json",  # 从 domain 配置加载超参数
                           **{k: getattr(args, k) for k in OVERRIDES})  # 命令行覆盖项（None 表示未提供）
    run = Run(domain, cfg, ROOT, Path(args.runs))  # 构建主循环宿主

    if args.cmd == "baseline":  # 分发：baseline
        run.baseline(args.job)
        run.calibrate([args.job])  # 顺带用基线评估做噪声带校准
    elif args.cmd == "calibrate":  # calibrate：可基于多个基线评估
        run.calibrate([j.strip() for j in args.jobs.split(",") if j.strip()])
    elif args.cmd == "round":  # round：跑单轮
        run.round(args.t, dry_run=args.dry_run)
    elif args.cmd == "readjudicate":  # readjudicate：重裁
        run.readjudicate(args.t)
    elif args.cmd == "reevaluate":  # reevaluate：重测（可只测部分变体）后重裁
        run.reevaluate(args.t, [v for v in args.variants.split(",") if v] or None)
    elif args.cmd == "run":  # run：交给驱动器（子进程逐轮）
        drive(Path(__file__), args.domain, Path(args.runs), cfg.T, args.start,
              extra_args=[a for k in OVERRIDES if getattr(args, k) is not None  # 把显式覆盖参数透传给每轮子进程
                          for a in ("--" + k.replace("_", "-"), str(getattr(args, k)))])
    elif args.cmd == "heldout":  # heldout：选择任务集
        ids = domain.heldout_ids() if args.set == "heldout" else domain.evolve_ids()
        if not ids:  # 该集为空
            sys.exit(f"domain {domain.name} has no {args.set} split")  # 报错退出
        run.heldout(args.label, ids, ref=args.ref)
    elif args.cmd == "smoke":  # smoke：现任活性检查
        run.ensure_branch()  # 确保分支存在
        wt = run.checkout("smoke", run.branch)  # 游离检出分支尖
        ok, detail = domain.smoke(wt, run.runs, "smoke", domain.smoke_ids())  # 跑少量任务
        print(json.dumps({"ok": ok, **detail}, indent=1))  # JSON 输出结果
        sys.exit(0 if ok else 1)  # 退出码反映活性
    elif args.cmd == "status":  # status：打印状态摘要
        fr = run.frontier()  # 读 frontier
        print(json.dumps({k: v for k, v in fr.items() if k != "config"}, indent=1))  # 除 config 外全部打印
        print("b_t schedule:", budget_table(cfg.T, cfg.b_min, cfg.b_max))  # 退火调度表
        print(f"delta = {run.delta():.5f}")  # 当前噪声带
        for r in run.history.render(60):  # 最近 60 条历史
            print(json.dumps(r, ensure_ascii=False))
        print("evolve branch:", G.rev(ROOT, run.branch), "tree", run.harness_tree(run.branch))  # 分支尖与 harness 树哈希


if __name__ == "__main__":  # 直接执行入口
    main()  # 启动 CLI
