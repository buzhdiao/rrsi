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
"""FastMCP streamable-http gateway: the tool ENVIRONMENT for the EngDesign
meta-harness. Frozen — this is the benchmark's world, not the agent's scaffold,
and the proposer never edits it.

Surface:
  - filesystem reads (read_text_file, list_files, get_directory_tree, search_files)
  - write_file
  - read_pdf (pymupdf)
  - code_exec (shell + the scientific python stack, run inside a bubblewrap jail)

No web_search: EngDesign ships every input it needs inside the workspace.

TWO CONTAINMENT RULES, both load-bearing:

1. Every path argument must resolve inside WORKSPACE_BASE. The agent's workspace
   holds only the files the task's own `copy_files.txt` exposes (prompt, response
   schema, images, the payload stub). The pristine benchmark tree — which holds
   `evaluate.py`, `solution.txt`, `rubrics.txt` and `reference.txt` right next to
   the prompt — lives outside it and is never reachable.

2. `code_exec` is a SHELL, so a path check on its arguments is worth nothing: the
   command itself can cat, grep or glob anywhere the process can reach. It
   therefore runs under `bwrap` with the workspace as the only writable mount,
   the system runtime read-only, and no network at all. Without this the whole
   evolution is unsound — a scaffold that learns to grep the filesystem for
   `solution.txt` would climb the metric while learning nothing, and would be
   invisible in the score.

Stateless with respect to task, so one gateway safely serves all concurrent
trials.
"""

# 【模块说明（中文）】FastMCP streamable-http 网关：EngDesign 元 harness 的工具环境。
# 冻结件——这是基准的"世界"，不是 agent 的 scaffold，proposer 永不编辑它。
# 工具面：文件读取（read_text_file/list_files/get_directory_tree/search_files）、
# write_file、read_pdf（pymupdf）、code_exec（shell + 科学 Python 栈，跑在 bubblewrap
# 牢笼里）。没有 web_search：EngDesign 需要的输入全在工作区里。
# 两条承重的封锁规则：
# 1) 一切路径参数必须解析到 WORKSPACE_BASE 之内。工作区只放任务自己的 copy_files.txt
#    暴露的文件（题面/响应 schema/图片/payload 存根）；原始基准树（evaluate.py、
#    solution.txt、rubrics.txt、reference.txt 就在题面旁边）在墙外，永远够不到。
# 2) code_exec 是 SHELL，对它的参数做路径检查毫无意义：命令本身能 cat/grep/glob 到
#    进程可达的任何地方。所以它跑在 bwrap 下：工作区是唯一可写挂载、系统运行时只读、
#    完全断网。没有这条，整个演化就不成立——学会 grep solution.txt 的 scaffold 会长分
#    却什么都没学，而且分数上看不出来。
# 对任务无状态：一个网关安全地服务所有并发 trial。

# os：环境变量与 os.walk/realpath
import os
# shutil：which("bwrap") 探测
import shutil
# subprocess：执行 shell 命令（code_exec）
import subprocess
# Path：路径对象
from pathlib import Path

# FastMCP：MCP 服务器框架
from fastmcp import FastMCP

# 监听主机（默认本机回环）
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
# 监听端口（默认 8994）
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8994"))
# code_exec 里的 python（科学栈所在）
CODE_EXEC_PYTHON = os.environ.get("CODE_EXEC_PYTHON", "/opt/conda/bin/python3")
# 单次 code_exec 的超时秒数
CODE_EXEC_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT", "600"))
# WORKSPACE_BASE：牢笼的墙——realpath 消符号链接后作为一切路径判定的根
WORKSPACE_BASE = os.path.realpath(
    os.environ.get("WORKSPACE_BASE",
                   os.path.join(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))), "workspaces")))
# SANDBOX：是否启用 bwrap 牢笼（默认开；ENGD_SANDBOX=0/false 关——仅调试用）
SANDBOX = os.environ.get("ENGD_SANDBOX", "1") not in ("0", "false", "")

# 单次读取/输出返回给 agent 的最大字符数（防上下文爆炸）
MAX_READ_CHARS = 200_000

# mcp：FastMCP 服务器实例（工具面与给 agent 的说明文字）
mcp = FastMCP(
    name="engdesign-gateway",
    instructions=(
        "Tools for engineering design tasks: read the problem statement and "
        "response schema, run shell + scientific python to compute and check a "
        "design, and write the submission file."
    ),
)


# ---------------------------------------------------------------------------- #
# Containment
# ---------------------------------------------------------------------------- #
# —— 封锁 ——
# 定义 _inside：解析路径并只在落在 WORKSPACE_BASE 内时返回，否则 None（越界）
def _inside(path: str) -> Path | None:
    """Resolve `path` and return it only if it lands inside WORKSPACE_BASE."""
    # realpath 消解符号链接与 ..，防止穿越
    try:
        p = Path(os.path.realpath(path))
    except Exception:  # noqa: BLE001
        return None
    # relative_to 抛 ValueError 即在墙外
    try:
        p.relative_to(WORKSPACE_BASE)
    except ValueError:
        return None
    # 墙内路径
    return p


# 定义 _denied：越界时的标准拒绝文案（提醒 agent 一切材料都在工作区内）
def _denied(path: str) -> str:
    return (f"Error: path is outside the workspace and cannot be accessed: "
            f"{path}. Everything this task provides is inside your workspace.")


# ---------------------------------------------------------------------------- #
# Filesystem
# ---------------------------------------------------------------------------- #
# —— 文件系统工具 ——
# 工具 read_text_file：读 UTF-8 文本文件（墙内），超长截断
@mcp.tool
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents. `path` is an absolute
    path inside your workspace. Use for .txt/.md/.json/.csv/.py and other
    plaintext; for PDFs use read_pdf, for binary formats use code_exec."""
    # 封锁检查
    p = _inside(path)
    if p is None:
        return _denied(path)
    # 必须是文件
    if not p.is_file():
        return f"Error: not a file: {path}"
    # 读文本（非法字节替换不炸）
    data = p.read_text(encoding="utf-8", errors="replace")
    # 超长截断并标注
    if len(data) > MAX_READ_CHARS:
        return data[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return data


# 工具 list_files：列目录（目录带 / 后缀，文件带字节数）
@mcp.tool
def list_files(path: str = ".") -> str:
    """List the entries of a directory inside your workspace, one per line,
    with a trailing '/' on directories and a byte size on files."""
    # 封锁检查
    p = _inside(path)
    if p is None:
        return _denied(path)
    # 必须是目录
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    # 排序后逐项格式化
    out = []
    for e in sorted(p.iterdir()):
        out.append(f"{e.name}/" if e.is_dir() else f"{e.name}  {e.stat().st_size}")
    # 空目录明说
    return "\n".join(out) or "(empty)"


# 工具 get_directory_tree：递归缩进树（跳过 __pycache__/.git，限 max_entries 条）
@mcp.tool
def get_directory_tree(path: str = ".", max_entries: int = 500) -> str:
    """Recursively list a directory inside your workspace as an indented tree,
    stopping after `max_entries` entries."""
    # 封锁检查
    root = _inside(path)
    if root is None:
        return _denied(path)
    # 必须是目录
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    # 输出行与条目计数
    lines, n = [], 0
    # os.walk 遍历（top-down 便于原地剪枝）
    for dirpath, dirnames, filenames in os.walk(root):
        # 剪掉噪声目录并保持排序
        dirnames[:] = [d for d in sorted(dirnames) if d not in ("__pycache__", ".git")]
        # 当前深度决定缩进
        depth = len(Path(dirpath).relative_to(root).parts)
        # 目录行
        lines.append("  " * depth + Path(dirpath).name + "/")
        # 文件行（计数超限即截断返回）
        for f in sorted(filenames):
            n += 1
            if n > max_entries:
                lines.append("  " * (depth + 1) + f"[truncated at {max_entries} entries]")
                return "\n".join(lines)
            lines.append("  " * (depth + 1) + f)
    # 返回完整树
    return "\n".join(lines)


# 工具 search_files：按文件名子串搜索（rglob，最多 500 条命中）
@mcp.tool
def search_files(path: str, name_contains: str) -> str:
    """Find files under a workspace directory whose NAME contains a substring."""
    # 封锁检查
    root = _inside(path)
    if root is None:
        return _denied(path)
    # 必须是目录
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    # 递归匹配文件名（大小写不敏感）
    hits = [str(p) for p in root.rglob("*")
            if p.is_file() and name_contains.lower() in p.name.lower()]
    # 排序截断；无命中明说
    return "\n".join(sorted(hits)[:500]) or "(no match)"


# 工具 write_file：向墙内路径写 UTF-8 文本（自动建父目录，覆盖写）
@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to an absolute path inside your workspace, creating
    parent directories as needed. Overwrites an existing file."""
    # 封锁检查
    p = _inside(path)
    if p is None:
        return _denied(path)
    # 写文件（失败返回错误而不是抛异常）
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    # 成功回执
    return f"Wrote {len(content)} chars to {p}"


# 工具 read_pdf：按页抽取墙内 PDF 文本（pymupdf，限页数与总字符）
@mcp.tool
def read_pdf(path: str, max_pages: int = 100) -> str:
    """Extract text from a PDF inside your workspace, page by page."""
    # 封锁检查
    p = _inside(path)
    if p is None:
        return _denied(path)
    # 必须是文件
    if not p.is_file():
        return f"Error: not a file: {path}"
    # pymupdf 缺失时的降级
    try:
        import fitz  # pymupdf
    except Exception as e:  # noqa: BLE001
        return f"Error: pymupdf unavailable ({e})"
    # 打不开的 PDF 明说
    try:
        doc = fitz.open(str(p))
    except Exception as e:  # noqa: BLE001
        return f"Error: cannot open pdf: {e}"
    # 逐页抽取
    parts = []
    for i, page in enumerate(doc):
        # 达到页数上限就停并标注
        if i >= max_pages:
            parts.append(f"[stopped after {max_pages} pages]")
            break
        parts.append(f"--- page {i + 1} ---\n{page.get_text()}")
    # 关闭文档句柄
    doc.close()
    # 拼接并截断
    text = "\n".join(parts)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return text


# ---------------------------------------------------------------------------- #
# Code execution (jailed)
# ---------------------------------------------------------------------------- #
# —— 代码执行（牢笼）——
# _RO_BINDS：bwrap 的只读绑定清单（系统运行时与 conda 只读）
_RO_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/etc",
             "/opt/conda")


# 定义 _bwrap_argv：组装 bwrap 命令行。
def _bwrap_argv(cwd: str) -> list[str]:
    """bwrap invocation: system runtime read-only, the WHOLE workspace base
    read-write, private /tmp, no network, dies with the gateway.

    The workspace base rather than the single task workspace is bound because
    one gateway serves every concurrent trial and each call names its own cwd;
    binding per-call keeps the mount set identical across trials, which matters
    because a champion and a candidate must run in the same environment.
    """
    # docstring：系统运行时只读、整个工作区根可写、私有 /tmp、全断网、随网关死。
    # 绑工作区根而非单个任务工作区，因为一个网关服务所有并发 trial，每次调用自带
    # cwd；按调用绑会破坏"冠军与候选必须在同一环境里跑"的可比性
    # 命令头部
    argv = ["bwrap"]
    # 存在的系统目录逐一只读绑定
    for b in _RO_BINDS:
        if os.path.exists(b):
            argv += ["--ro-bind", b, b]
    # 工作区根可写绑定；独立 /proc /dev；私有 /tmp；全部 namespace 隔离（断网）；
    # 父进程死则死；新会话；进入调用方指定的 cwd
    argv += [
        "--bind", WORKSPACE_BASE, WORKSPACE_BASE,
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--chdir", cwd,
    ]
    # 返回完整 argv
    return argv


# 工具 code_exec：执行 shell 命令（bwrap 牢笼内），返回 {success, output, returncode}
@mcp.tool
def code_exec(code: str, workdir: str | None = None) -> dict:
    """Execute a SHELL command. `code` is a shell command string, e.g.
    'ls -la', 'python script.py', or 'python -c "..."'. To run Python, write a
    script with write_file and run it, or use 'python -c'. Do NOT pass raw
    Python source directly as `code`.

    A scientific Python is on PATH (numpy, scipy, matplotlib, pandas, sympy,
    scikit-learn, scikit-image, opencv, control, cvxopt), along with iverilog,
    vvp, octave and ffmpeg. There is no network access. `workdir` sets the cwd
    and must be inside your workspace. Returns {success, output, returncode}."""
    # code 必填
    if not code:
        return {"success": False, "output": "Error: required parameter 'code'"}
    # workdir 封锁检查（给了就必须在墙内）
    cwd_p = _inside(workdir) if workdir else None
    if workdir and cwd_p is None:
        return {"success": False, "output": _denied(workdir)}
    # workdir 必须是已存在的目录
    if cwd_p is None or not cwd_p.is_dir():
        return {"success": False,
                "output": "Error: `workdir` must be an existing directory "
                          "inside your workspace."}
    # 转字符串路径
    cwd = str(cwd_p)

    # 受控环境变量：PATH 把科学 python 放最前；HOME 指向 cwd（防写穿真家目录）；
    # 关闭 matplotlib 显示；限制线程；不写 pyc
    env = {
        "PATH": os.path.dirname(CODE_EXEC_PYTHON) + os.pathsep
                + "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": cwd,
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # 牢笼开关：SANDBOX 开且系统有 bwrap 才加 bwrap 前缀；否则裸跑（仅调试）
    argv = (_bwrap_argv(cwd) if (SANDBOX and shutil.which("bwrap")) else [])
    # 追加要执行的 bash -lc 命令
    argv += ["/bin/bash", "-lc", code]
    try:
        # 运行（bwrap 模式下 cwd 由 --chdir 处理故传 None；超时按 CODE_EXEC_TIMEOUT）
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=CODE_EXEC_TIMEOUT, env=env,
                              cwd=(None if argv[0] == "bwrap" else cwd))
    except subprocess.TimeoutExpired:
        # 超时回执
        return {"success": False,
                "output": f"Error: timed out after {CODE_EXEC_TIMEOUT}s"}
    except Exception as e:  # noqa: BLE001
        # 其他失败回执
        return {"success": False, "output": f"Error: {e}"}
    # 拼接 stdout 与 stderr（stderr 有才附）
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    # 输出超长截断
    if len(out) > MAX_READ_CHARS:
        out = out[:MAX_READ_CHARS] + f"\n[output truncated at {MAX_READ_CHARS} chars]"
    # 返回结构化结果
    return {"success": proc.returncode == 0, "output": out,
            "returncode": proc.returncode}


# 启动入口：建工作区根、打印配置、以 streamable-http 起服务
if __name__ == "__main__":
    # 确保工作区根存在
    os.makedirs(WORKSPACE_BASE, exist_ok=True)
    # 打印启动信息
    print(f"[gateway] workspace base {WORKSPACE_BASE}  sandbox={SANDBOX}", flush=True)
    # 启动 FastMCP HTTP 服务
    mcp.run(transport="http", host=GATEWAY_HOST, port=GATEWAY_PORT)
