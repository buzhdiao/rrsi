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
"""Unit tests for the method core: no API, no benchmark.

    python3 -m pytest -q tests/        or        python3 tests/test_core.py
"""
# 【模块说明（中文）】方法核心的单元测试：不调 LLM API、不跑真实基准，只验证
# RRSI 循环中"纯计算"的骨架——退火编辑预算（schedule）、评估聚合与判据加权
# （evaluate）、噪声带 delta 的自助法校准（calibrate）、成本规则与整轮选择
# （selection）、编辑历史的摘要/修剪窗口/停滞标志/探索指令（history）、组件
# 归一化与新颖度（components）、配置加载与 JSON 往返（config）。
# 既可用 pytest 跑，也可 `python3 tests/test_core.py` 走文件尾部的极简 runner。

# json：配置往返测试中写/读临时配置文件
import json
# math：余弦退火的期望值、-inf 哨兵比较
import math
# sys：把仓库根塞进 sys.path，并为 runner 提供退出码
import sys
# tempfile：为 History/配置测试提供一次性目录
import tempfile
# Path：跨平台的路径构造
from pathlib import Path

# 把仓库根加到 sys.path 头部：无需安装包即可 `import rrsi`（noqa: E402 也因此而来）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# calibrate：噪声带 delta 的校准入口
from rrsi.calibrate import calibrate                       # noqa: E402
# components：组件归一化 normalize 与新颖度 novelty
from rrsi.components import normalize, novelty              # noqa: E402
# config：带 CLI 覆盖与 notes 兜底的配置加载
from rrsi.config import RRSIConfig                         # noqa: E402
# evaluate：单任务结果 TaskResult、总评估 EvalResult 与聚合 aggregate
from rrsi.evaluate import EvalResult, TaskResult, aggregate  # noqa: E402
# history：编辑历史 L_t 及其停滞标志 stall_flag、探索指令 exploration
from rrsi.history import History, exploration, stall_flag   # noqa: E402
# schedule：逐轮编辑预算 edit_budget 与整张预算表 budget_table
from rrsi.schedule import budget_table, edit_budget         # noqa: E402
# selection：候选 Candidate、成本规则 cost_rule 与整轮选择 select_round
from rrsi.selection import Candidate, cost_rule, select_round  # noqa: E402


def _ev(job, rewards_by_task, k, tokens=1000.0):  # 测试夹具 _ev：由 {任务: 奖励列表} 一步造出聚合评估
    """构造每个任务的 TaskResult（奖励列表 + 等长 token 计数）并聚合成
    EvalResult；token 计数默认每次 1000，让成本规则测试可控。"""
    per = {t: TaskResult(rewards=list(r), tokens=[tokens] * len(r))  # 每任务：奖励与同长度的固定 token 计数
           for t, r in rewards_by_task.items()}
    return aggregate(job, k, per)  # 作业名 + 每任务 k 次 + 逐任务结果 -> EvalResult


def test_schedule_matches_eq_anneal():  # 测试：编辑预算调度必须逐点等于余弦退火公式
    """edit_budget 应逐点等于论文 Eq. 的余弦退火；预算表从 bmax 单调不增
    收敛到 bmin（早期探索放宽、后期收敛收紧）。"""
    T, bmin, bmax = 20, 1, 4  # 总轮数与预算上下界
    for t in range(T):  # 逐轮核对公式
        expect = math.ceil(bmin + (bmax - bmin) * 0.5 * (1 + math.cos(math.pi * t / T)))  # 期望值：t=0 取 bmax、t=T 取 bmin 的余弦退火再向上取整
        assert edit_budget(t, T, bmin, bmax) == expect  # 调度器输出与公式完全一致
    tab = budget_table(T, bmin, bmax)  # 整张预算表
    assert tab[0] == bmax and tab == sorted(tab, reverse=True)  # 起点最宽松，且整表单调不增
    assert edit_budget(T, T, bmin, bmax) == bmin  # 最后一轮收敛到下界 bmin


def test_estimator_and_weights():  # 测试：总分估计器 Ŝ 的均匀与判据加权两种形态
    """均匀权重下 S 就是逐 trial 的平均；给了 weights 则按判据加权聚合。"""
    ev = _ev("j", {"a": [1, 0], "b": [1, 1]}, 2)  # 两个任务各 2 次 trial：4 次里 3 次成功
    assert abs(ev.S - 0.75) < 1e-9 and ev.n_expected == 4 and ev.missing == 0  # S=0.75，期望 4 次、无缺失
    per = {"a": TaskResult(rewards=[0.5, 1.0], weights=[10, 10]),  # 判据 a：权重合计 20，均值 0.75
           "b": TaskResult(rewards=[0.0, 0.0], weights=[90, 90])}  # 判据 b：权重合计 180，均值 0
    ev = aggregate("w", 2, per)  # 加权聚合
    assert abs(ev.S - 15 / 200) < 1e-9          # 按判据加权的得分比例：(20*0.75 + 180*0) / 200


def test_calibration_bootstrap_and_repeats():  # 测试：噪声带 delta 的自助法校准（单次与多次评估）
    """单次评估走 bootstrap 重抽样；多次独立评估可合并校准，delta 恒非负。"""
    rng_rewards = {f"t{i}": [i % 2, (i + 1) % 2] for i in range(40)}  # 40 个任务，每个恰一成一败（确定性伪随机）
    ev = _ev("base", rng_rewards, 2)  # 基础评估：S=0.5
    cal = calibrate([ev], z=2.0, reps=300)  # 单评估：对任务自助重抽样 300 次估带宽
    assert cal["delta"] > 0 and cal["method"].startswith("bootstrap")  # 单评估必然走 bootstrap，带宽为正
    ev2 = _ev("base2", {f"t{i}": [1, 1 if i % 3 else 0] for i in range(40)}, 2)  # 第二次评估：每 3 个任务错 1 个
    cal2 = calibrate([ev, ev2], z=2.0, reps=100)  # 两次独立评估合并校准
    assert cal2["n_evals"] == 2 and cal2["delta"] >= 0  # 记录评估次数；带宽非负（保守下限 0）


def test_cost_rule_both_branches():  # 测试：成本规则 c 的增益分支与中性分支
    """增益超噪声带走 L1 预算（Eq. tokenbudget）；中性编辑只许更便宜或引入新结构。"""
    cfg = RRSIConfig(beta0=0.1, beta1=40.0, w_s=100.0, w_c=15.0, w_n=0.5)  # 成本模型系数：固定项、分数斜率与打分权重
    ok, _ = cost_rule(0.05, 0.10 + 40 * 0.05 - 0.01, 0, 0.02, cfg)      # 增益在预算内：ΔS=0.05 过带，成本贴着上限之下
    assert ok
    ok, _ = cost_rule(0.05, 0.10 + 40 * 0.05 + 0.01, 0, 0.02, cfg)      # 超预算：成本略高于 beta0+beta1*ΔS 即拒绝
    assert not ok
    ok, _ = cost_rule(0.0, -0.10, 0, 0.02, cfg)                          # 中性且更便宜：放行（白赚的简化）
    assert ok
    ok, _ = cost_rule(0.0, 0.10, 0, 0.02, cfg)                           # 中性且更贵：拒绝（没收益就不许加成本）
    assert not ok
    ok, _ = cost_rule(0.0, 0.0, 1, 0.02, cfg)                            # 中性、零成本但引入新结构（nov=1）：放行
    assert ok


def test_selection_floor_argmax_and_sstar():  # 测试：整轮选择——下限闸门、argmax、L1 成本与不可补偿护栏
    """S' 低于 S* 减下限的候选不可取用；获胜者是可取用者中目标函数的 argmax；
    增益候选太贵被 L1 成本规则拦下；domain 护栏一票否决、不能被分数补偿。"""
    cfg = RRSIConfig(beta0=0.1, beta1=40.0, w_s=100.0, w_c=15.0, w_n=0.5)  # 与成本测试同一组系数
    inc = _ev("inc", {f"t{i}": [1, 1] if i < 5 else [0, 0] for i in range(10)}, 2)   # 现任：10 任务过 5 个，S=0.5
    a = Candidate("A", [{"id": "C1", "component": "prompt"}],
                  ev=_ev("A", {f"t{i}": [1, 1] if i < 7 else [0, 0] for i in range(10)}, 2))  # 候选 A：S=0.7，prompt 编辑
    b = Candidate("B", [{"id": "C1", "component": "skill"}],
                  ev=_ev("B", {f"t{i}": [1, 1] if i < 6 else [0, 0] for i in range(10)}, 2))  # 候选 B：S=0.6，skill 编辑
    c = Candidate("C", [{"id": "C1", "component": "config"}],
                  ev=_ev("C", {f"t{i}": [1, 1] if i < 3 else [0, 0] for i in range(10)}, 2))  # 候选 C：S=0.3，低于下限必被拦
    d = Candidate("D", [], gate_failure="critic_reject")  # 候选 D：被 critic 拒绝，没有评估结果
    S_star, delta = 0.55, 0.05                      # S*=0.55 高于现任总分（取自更早的峰值）；带宽 0.05
    win, decs = select_round([a, b, c, d], inc, S_star, delta, cfg, {}, guard_fn=None)  # 跑整轮选择（无护栏）
    assert win is a  # 获胜者是 argmax：可取用者里 A 的目标值最高
    by = {x.variant: x for x in decs}  # 按变体名索引裁决记录
    assert by["A"].admissible and by["B"].admissible  # A、B 均通过闸门（S' - 下限之上）
    assert not by["C"].admissible and "floor" in by["C"].reason  # C 被 S*-下限闸门拦下
    assert not by["D"].admissible and by["D"].reason == "critic_reject"  # D 因 critic 拒绝出局（未测量即不算证据）
    # 增益候选若太贵，会被 L1 成本规则拦下
    exp = Candidate("E", [{"id": "C1", "component": "prompt"}],
                    ev=_ev("E", {f"t{i}": [1, 1] if i < 7 else [0, 0] for i in range(10)}, 2,
                           tokens=1000.0 * (1 + 0.1 + 40 * 0.2 + 0.5)))  # 成本恰好越过 beta0+beta1*ΔS 的上限
    win2, decs2 = select_round([exp], inc, S_star, delta, cfg, {})
    assert win2 is None and "cost rule" in decs2[0].reason  # 无获胜者，拒绝理由指向成本规则
    # domain 护栏不可被补偿
    win3, decs3 = select_round([a], inc, S_star, delta, cfg, {},
                               guard_fn=lambda i, c: ["valid rate fell"])  # 护栏报告"有效率下降"这类 domain 红线
    assert win3 is None and "guard" in decs3[0].reason  # 即便 S' 更高也一票否决（护栏不可补偿）


def test_history_summaries_prune_and_explore():  # 测试：编辑历史 L_t 的摘要、修剪窗口、停滞标志与探索指令
    """tried/组件计数/新颖度、yield_g 的滑动窗口与 -inf 哨兵、prune_set 携带
    要拆除的现任机制、stall_flag 的窗口判定、exploration 的保留槽位文案。"""
    with tempfile.TemporaryDirectory() as td:  # 一次性目录放 history.jsonl
        h = History(Path(td) / "history.jsonl")  # 内存 + JSONL 落盘的编辑历史
        h.append_candidate(0, "A", [{"id": "C1", "component": "prompt", "hypothesis": "h1"}],
                           "ACCEPTED", 0.03, 0.05, True, 0.53, 1000, None)  # t=0：prompt 编辑被接受，ΔS=+0.03
        h.append_candidate(1, "A", [{"id": "C1", "component": "prompt", "hypothesis": "h2"}],
                           "REJECTED", -0.01, 0.0, False, 0.52, 1000, None)  # t=1：prompt 编辑被拒，ΔS=-0.01
        h.append_candidate(1, "B", [{"id": "C1", "component": "skill", "hypothesis": "h3"},
                                    {"id": "C2", "component": "memory", "hypothesis": "h4"}],
                           "REJECTED", -0.02, 0.3, False, 0.51, 1300, None)  # t=1：候选 B 捆绑 skill+memory 两编辑，整体被拒
        h.append_candidate(2, "A", [{"id": "C1", "component": "config", "hypothesis": "h5"}],
                           "critic_reject", None, None, False, None, None, None, "leak")  # t=2：config 编辑被 critic 以"泄漏"拒绝，无任何测量
        assert h.tried() == {"prompt", "skill", "memory"}       # 已试组件集合：被 critic 拒绝的 config 不算"试过"
        g = h.yield_g(t=3, n_prune=4)  # t=3 的组件牵引 g：滑动窗口内结果对每个组件的拉力
        assert g["prompt"] == 0.03 and g["skill"] == -0.02  # prompt 由被接受的 +0.03 主导；skill 拉着 -0.02
        # 滑动窗口：t=5、n_prune=4 时，t=0 那条被接受的 prompt 编辑已滑出窗口
        g5 = h.yield_g(t=5, n_prune=4)
        assert g5["prompt"] == -0.01  # 窗口里只剩被拒的那条，牵引转负
        assert h.yield_g(t=6, n_prune=4)["prompt"] == -math.inf    # 窗口内近期什么都没有：用 -inf 表示"毫无近期牵引"
        prune = {p["component"]: p for p in h.prune_set(t=5, n_prune=4)}  # 修剪集：按组件列出要拆的现任机制
        assert set(prune) == {"prompt", "skill", "memory"}  # 三个被拒/被撤的组件都进入修剪集
        assert prune["prompt"]["accepted_edits_in_incumbent"][0]["hypothesis"] == "h1"  # 修剪指令携带要拆除的机制本体（h1 的 prompt 编辑）
        assert h.incumbent_component_counts()["prompt"] == 1  # 现任机制里 prompt 组件出现 1 次
        assert novelty(["client_tool", "prompt"], h.incumbent_component_counts()) == 1  # 新颖度=未试组件数：client_tool 没试过（prompt 试过）
        assert novelty(["skill"], {"skill": 2}) == 0  # skill 已在现任中出现：新颖度为 0
        assert h.has(1, "B") and not h.has(5, "A")  # has：按（轮, 变体）查历史存在性
    traj = [0.50, 0.53, 0.53, 0.535, 0.60]  # 一条总分轨迹（含平台期与跳升）
    assert stall_flag(traj, 3, 3, 0.02) == 0       # 窗口 3：0.535 - 0.50 = 0.035 > 0.02，未停滞
    assert stall_flag(traj, 3, 2, 0.02) == 1       # 窗口 2：0.535 - 0.53 = 0.005 <= 0.02，判停滞
    assert stall_flag(traj, 4, 2, 0.02) == 0       # 窗口 2：0.60 - 0.53 = 0.07，跳升解除停滞
    assert stall_flag(traj, 1, 3, 0.02) == 0       # 历史不足一个窗口：不轻言停滞
    e = exploration(3, 1, {"prompt"}, 1)  # 探索指令：t=3、已停滞 1 轮、试过 {prompt}、保留 1 个槽位
    assert e["sigma"] == 1 and "prompt" not in e["untried"] and "RESERVED" in e["text"]  # 停滞标志、未试组件清单、保留槽位文案齐全


def test_component_normalization():  # 测试：组件归一化 normalize 的证据优先级链
    """声明的组件要经 diff 证据核实；声明缺失/非法时按 diff 路径与 domain
    模式推断；一切证据皆无则回落到 prompt（最保守的组件标签）。"""
    assert normalize("Skill", "", []) == "prompt"          # 空 diff 毫无证据：即使声明"Skill"也不可信，回落 prompt
    assert normalize("Skill", "+++ b/x/skills/y/SKILL.md", []) == "skill"  # diff 路径证据支持声明：把别名规范化为 skill
    assert normalize("bogus", "+++ b/harness/skills/x/SKILL.md", []) == "skill"  # 声明非法（不在词表 K）：改按 diff 路径推断出 skill
    assert normalize(None, "+++ b/harness/resum.py\n+ keep_last = 5",
                     [("context_mgmt", [r"resum\.py"])]) == "context_mgmt"  # 无声明：domain 专属模式命中 resum.py，归为 context_mgmt
    assert normalize(None, "nothing structural", []) == "prompt"  # 找不到任何结构性证据：默认按 prompt（指令编辑）处理


def test_config_roundtrip():  # 测试：配置加载的优先级与评估结果的 JSON 往返
    """RRSIConfig.load：文件覆盖默认、CLI 覆盖文件、未知键落进 notes 不报错；
    EvalResult 的 to_json/from_json 往返无损。"""
    with tempfile.TemporaryDirectory() as td:  # 一次性目录放临时配置
        p = Path(td) / "rrsi.json"  # 临时配置文件路径
        p.write_text(json.dumps({"T": 7, "k": 3, "beta1": 12.5, "custom": "x"}))  # 写入已知键 + 一个未知键 custom
        cfg = RRSIConfig.load(p, T=None, k=5)  # T 未给 CLI 值→用文件值 7；k 给了 CLI 值→覆盖文件值 3
        assert cfg.T == 7 and cfg.k == 5 and cfg.beta1 == 12.5 and cfg.notes["custom"] == "x"  # 未知键落进 notes，不炸
        ev = EvalResult.from_json(aggregate("j", 1, {"a": TaskResult(rewards=[1.0])}).to_json())  # 聚合 -> JSON -> 回读 的往返
        assert ev.S == 1.0  # 往返后分数无损


if __name__ == "__main__":  # 直接运行本文件时的极简测试 runner（不依赖 pytest）
    import inspect  # inspect：判定对象是否为函数
    fails = 0  # 失败计数
    for name, fn in list(globals().items()):  # 枚举本模块全局命名空间中的全部 test_ 函数
        if name.startswith("test_") and inspect.isfunction(fn):
            try:  # 逐个执行，单点失败不中断
                fn()
                print(f"ok   {name}")  # 通过就打印 ok
            except Exception as e:  # noqa: BLE001
                fails += 1  # 兜底捕获任何异常：记一次失败，继续跑其余测试
                print(f"FAIL {name}: {e!r}")  # 打印异常的 repr 便于定位
    sys.exit(1 if fails else 0)  # 有失败则退出码 1（供 CI 判定）
