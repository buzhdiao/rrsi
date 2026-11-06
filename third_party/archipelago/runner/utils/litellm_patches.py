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

"""Monkey-patches for litellm bugs. Remove when upgrading past the fix."""

# 从包元数据导入 version 函数，用于读取已安装 litellm 的版本号。
from importlib.metadata import version as _pkg_version
# 导入 typing.Any：参数 dict 的通用值类型注解。
from typing import Any

# 导入 LiteLLM 内部 OpenAI GPT-5 参数映射配置类，待打补丁的目标。
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
# 导入 packaging 的 Version：做语义化版本号比较。
from packaging.version import Version

# litellm <= 1.82.1 incorrectly rejects xhigh reasoning_effort for gpt-5.4.
# The hardcoded check only allows gpt-5.1-codex-max and gpt-5.2, but OpenAI's
# model card confirms gpt-5.4 supports xhigh. Patch until litellm ships the fix.
# litellm 1.82.1 及更早版本会错误地拒绝 gpt-5.4 使用 xhigh 推理努力档位。
# 其硬编码白名单只允许 gpt-5.1-codex-max 与 gpt-5.2，但 OpenAI 模型卡确认
# gpt-5.4 支持 xhigh。在 litellm 发布修复前先用此补丁。
# 记录最后一个存在该 bug 的 litellm 版本。
_LITELLM_MAX_BUGGY_VERSION = Version("1.82.1")

# 已安装的 litellm 版本仍在有 bug 的范围内时，才应用猴子补丁。
if Version(_pkg_version("litellm")) <= _LITELLM_MAX_BUGGY_VERSION:
    # 保存原始的参数映射方法，供补丁函数内部回退调用。
    _original_map = OpenAIGPT5Config.map_openai_params

    def _patched_map(
        self: OpenAIGPT5Config,
        non_default_params: dict[str, Any],
        optional_params: dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> dict[str, Any]:
        # Temporarily make gpt-5.4 look like gpt-5.2 so the xhigh check passes.
        # 临时把模型名伪装成 gpt-5.2，使 xhigh 白名单检查通过。
        fake_model = False
        # 从非默认参数或可选参数中取出 reasoning_effort（推理努力档位）。
        reasoning_effort = non_default_params.get(
            "reasoning_effort"
        ) or optional_params.get("reasoning_effort")
        if "gpt-5.4" in model and reasoning_effort == "xhigh":
            # 命中"gpt-5.4 + xhigh"组合：标记已伪装，并替换模型名。
            fake_model = True
            model = model.replace("gpt-5.4", "gpt-5.2")

        # 调用原始实现完成真正的参数映射校验。
        result = _original_map(
            self, non_default_params, optional_params, model, drop_params
        )

        # Restore the real model — callers may inspect optional_params["model"] downstream.
        # 还原真实模型名——下游调用方可能检查 result 中的 model 字段。
        if fake_model:
            result["model"] = result.get("model", "").replace("gpt-5.2", "gpt-5.4")

        # 返回映射结果。
        return result

    # 用补丁函数替换类上的原方法（类型检查器会报赋值类型不符，故忽略）。
    OpenAIGPT5Config.map_openai_params = _patched_map  # type: ignore[assignment]
