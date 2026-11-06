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

# 导入 json：解析与写出 asciinema 录制文件的 JSON 行。
import json
# 导入 Path：录制文件与临时文件的路径操作。
from pathlib import Path
# 导入 TextIO：输出文件句柄的类型注解。
from typing import TextIO


class AsciinemaHandler:
    """
    Handles operations related to asciinema recordings, including marker management.
    """

    def __init__(self, markers: list[tuple[float, str]], recording_path: Path):
        """
        Initialize the AsciinemaHandler.

        Args:
            markers: List of tuples containing (timestamp, label) for each marker
            recording_path: Path to the asciinema recording file
        """
        # 把标记按时间戳升序排序（空列表保持为空），便于按序插入。
        self._markers = sorted(markers, key=lambda x: x[0]) if markers else []
        # 保存录制文件路径。
        self._recording_path = recording_path

    def merge_markers(self) -> None:
        """
        Merge asciinema markers into a recording.

        Inserts marker events into an asciinema recording file at specified timestamps.
        Markers are added as special events with type 'm' in the recording format.
        The original recording is preserved until the merge is successful.

        In the future Asciinema might support adding markers via RCP:
        https://discourse.asciinema.org/t/add-markers-with-title-from-cli/861
        """
        # 无标记或录制文件不存在：无需合并，直接返回。
        if not self._markers or not self._recording_path.exists():
            return

        # Create a temporary file in the same directory as the recording
        # 在录制文件同目录创建临时文件。
        temp_path = self._recording_path.with_suffix(".tmp")
        # 先把合并结果写入临时文件（保证原文件在成功前不被破坏）。
        self._write_merged_recording(temp_path)
        # 合并成功后原子替换原录制文件。
        temp_path.replace(self._recording_path)

    def _write_merged_recording(self, output_path: Path) -> None:
        """
        Write a new recording file with markers merged in at the correct timestamps.
        """
        # 标记游标：下一个待插入标记的下标。
        marker_index = 0

        # 同时打开原录制文件（读）与输出文件（写）。
        with (
            open(self._recording_path) as input_file,
            open(output_path, "w") as output_file,
        ):
            # Preserve header
            # 原样保留 asciinema 文件首行头部（格式版本声明）。
            output_file.write(input_file.readline())

            for line in input_file:
                # 逐行处理录制事件，按需插入标记并推进游标。
                marker_index = self._process_recording_line(
                    line, output_file, marker_index
                )

            # Add any remaining markers at the end
            # 把晚于所有事件的剩余标记追加到文件末尾。
            self._write_remaining_markers(output_file, self._markers[marker_index:])

    def _process_recording_line(
        self,
        line: str,
        output_file: TextIO,
        marker_index: int,
    ) -> int:
        """Process a single line from the recording, inserting markers as needed."""
        # 非 JSON 事件行（如空行）：原样写出，不插标记。
        if not line.startswith("["):
            output_file.write(line)
            return marker_index

        try:
            # 解析事件行：asciinema 格式为 [时间戳, 事件类型, 数据]。
            data = json.loads(line)
            # 取出事件时间戳。
            timestamp = float(data[0])

            # Insert any markers that should appear before this timestamp
            # 把时间不晚于本事件的所有标记依次插入到它之前。
            while (
                marker_index < len(self._markers)
                and self._markers[marker_index][0] <= timestamp
            ):
                self._write_marker(output_file, self._markers[marker_index])
                marker_index += 1

        except (json.JSONDecodeError, ValueError, IndexError):
            # If we can't parse the line, preserve it as-is
            # 无法解析的行原样保留，不做任何处理。
            pass

        # 写出原事件行。
        output_file.write(line)
        # 返回更新后的标记游标。
        return marker_index

    def _write_marker(self, output_file: TextIO, marker: tuple[float, str]) -> None:
        """Write a single marker event to the output file."""
        # 拆出标记的时间戳与标签。
        marker_time, marker_label = marker
        # 构造 asciinema 标记事件：类型为 "m"。
        marker_data = [marker_time, "m", marker_label]
        # 以 JSON 行形式写出该标记事件。
        output_file.write(json.dumps(marker_data) + "\n")

    def _write_remaining_markers(
        self, output_file: TextIO, markers: list[tuple[float, str]]
    ) -> None:
        """Write any remaining markers that come after all recorded events."""
        # 逐个写出所有晚于录制结束时间的标记。
        for marker in markers:
            self._write_marker(output_file, marker)
