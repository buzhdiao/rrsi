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
"""RRSI: regularized test-time harness evolution.

Domain-agnostic implementation of the two algorithms in the paper
"Regularized Recursive Self-Improvement of Agent Harnesses":

  Algorithm 1 (proposal side)   rrsi.loop.propose_round
  Algorithm 2 (selection side)  rrsi.selection.select_round

Everything benchmark-specific (how a harness is run, scored, rendered and
screened for leakage) lives behind the Domain interface in rrsi.domain and is
implemented under domains/<name>/adapter.py.
"""
# 【模块说明（中文）】RRSI 包：正则化的测试时 harness 进化。
# 论文《Regularized Recursive Self-Improvement of Agent Harnesses》两个算法的
# 与基准无关（domain-agnostic）实现：
#   Algorithm 1（提案侧）  rrsi.loop（起草->筛查->评估的一轮主循环）
#   Algorithm 2（选择侧）  rrsi.selection.select_round（地板/成本规则/argmax）
# 一切基准相关的东西（harness 如何运行、打分、渲染轨迹、泄漏筛查）都藏在
# rrsi.domain 的 Domain 接口之后，由 domains/<name>/adapter.py 实现。

# __all__：包的公开模块清单（from rrsi import * 的白名单，也是模块地图）
__all__ = ["config", "schedule", "history", "components", "select",
           "evaluate", "calibrate", "propose", "critic", "analyst",
           "digester", "loop", "driver", "domain", "gitops", "llm"]
