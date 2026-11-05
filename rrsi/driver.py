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
"""Sequential driver: baseline + calibration if missing, then rounds t..T-1.

A round is settled once the frontier trajectory has an entry for t+1.
`touch runs/<domain>/STOP` stops after the current round. Consecutive
infrastructure failures stop the driver so a broken environment cannot burn
the whole budget.
"""
# 【模块说明（中文）】顺序驱动器：缺 baseline + 校准就先补上，然后从 t 跑到 T-1 轮。
# 一轮"落定（settled）"的判据：frontier 轨迹里出现 t+1 的条目。
# `touch runs/<domain>/STOP` 会在当前轮结束后停下。连续的基础设施失败会终止驱动器，
# 防止坏掉的环境烧光全部预算。

# 延迟类型注解求值
from __future__ import annotations

# json：读 frontier.json 判断进度
import json
# subprocess：以子进程方式逐轮调用 rrsi.py（进程隔离：一轮崩溃不拖垮驱动器）
import subprocess
# sys：sys.executable（当前解释器）与退出
import sys
# time：日志时间戳
import time
# Path：运行目录路径
from pathlib import Path

# MAX_CONSECUTIVE_INFRA：连续基础设施失败的上限——超过即熔断停机
MAX_CONSECUTIVE_INFRA = 3


def settled_rounds(frontier_path: Path) -> int:  # 定义 settled_rounds：从 frontier 读出已落定的轮数
    if not frontier_path.exists():  # baseline 还没跑
        return -1  # 返回 -1（特殊值：驱动器据此决定先跑 baseline）
    fr = json.loads(frontier_path.read_text())  # 读 frontier
    return len(fr["trajectory"]) - 1          # trajectory has entries 0..settled
    # 轨迹含 t=0..settled 的条目，条目数减 1 即最新落定轮号


def drive(entry: Path, domain: str, runs_dir: Path, T: int, start: int = 0,  # 定义 drive：主循环——从 start 轮跑到 T-1
          extra_args: list[str] | None = None) -> None:
    run_dir = runs_dir / domain  # 本 domain 的运行目录
    fr = run_dir / "frontier.json"  # frontier 文件（进度的事实来源）
    stop = run_dir / "STOP"  # STOP 哨兵文件：存在即请求优雅停机
    logs = run_dir / "logs"  # 每轮日志目录
    logs.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
    base_args = [sys.executable, str(entry), "--domain", domain,  # 子进程命令前缀：当前解释器 + rrsi.py + domain
                 "--runs", str(runs_dir)] + list(extra_args or [])  # + 命令行覆盖参数（--T 等透传给每一轮）
    if not fr.exists():  # 还没有 frontier
        print(f"[driver:{domain}] baseline + calibrate", flush=True)  # 提示先做基线
        with open(logs / "baseline.log", "a") as lf:  # baseline 日志（追加模式，多次重启不覆盖）
            r = subprocess.run(base_args + ["baseline"], stdout=lf, stderr=subprocess.STDOUT)  # stderr 并入 stdout 全部落盘
        if r.returncode != 0 or not fr.exists():  # 失败或 frontier 仍未生成
            print(f"[driver:{domain}] baseline failed (rc={r.returncode})")  # 报告失败
            sys.exit(1)  # 退出——没有基线一切免谈
    if not (run_dir / "calibration.json").exists():  # 还没有噪声带校准
        with open(logs / "calibrate.log", "a") as lf:  # 校准日志
            subprocess.run(base_args + ["calibrate"], stdout=lf, stderr=subprocess.STDOUT)  # 补跑 calibrate（用 baseline 的 eval）
    infra = 0  # 连续失败计数器（成功一轮即清零）
    for t in range(start, T):  # 逐轮推进
        if stop.exists():  # 用户请求停止
            print(f"[driver:{domain}] STOP present; exiting", flush=True)  # 说明停止原因
            return  # 优雅退出（当前轮已完成才会看到 STOP，不砍正在跑的轮）
        if settled_rounds(fr) >= t + 1:  # 该轮已落定（断点续跑：跳过）
            print(f"[driver:{domain}] round {t} already settled", flush=True)  # 打印跳过信息
            continue  # 下一轮
        log = logs / f"r{t}.log"  # 本轮日志文件
        print(f"[driver:{domain}] === round {t} ({time.strftime('%H:%M:%S')}) -> {log}",  # 宣告开轮与日志位置
              flush=True)
        with open(log, "a") as lf:  # 追加模式写本轮日志
            r = subprocess.run(base_args + ["round", "--t", str(t)],  # 子进程跑单轮（崩溃只损失这一轮）
                               stdout=lf, stderr=subprocess.STDOUT)  # 输出全部落盘
        if r.returncode != 0 or settled_rounds(fr) < t + 1:  # 退出码非 0 或轮未落定
            infra += 1  # 失败计数 +1
            print(f"[driver:{domain}] round {t} did not settle (rc={r.returncode}, "  # 报告未落定及当前连续失败数
                  f"{infra}/{MAX_CONSECUTIVE_INFRA})", flush=True)
            if infra >= MAX_CONSECUTIVE_INFRA:  # 连续 3 次失败
                print(f"[driver:{domain}] too many consecutive failures; stopping")  # 熔断
                sys.exit(1)  # 停机：坏环境不该继续烧钱
        else:  # 本轮成功
            infra = 0  # 清零连续失败计数
            b = json.loads(fr.read_text())["incumbent"]  # 读新现任
            print(f"[driver:{domain}]   incumbent t={b['t']} {b['commit']} S={b['S']:.4f}",  # 报告现任轮次/提交/分数
                  flush=True)
    print(f"[driver:{domain}] all rounds settled", flush=True)  # 全部轮次完成
