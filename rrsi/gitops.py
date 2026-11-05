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
"""Git plumbing for candidate isolation.

Each domain evolves on its own branch `evolve/<domain>`. Every candidate
harness of a round is drafted and evaluated in its own git worktree checked out
on a branch `<domain>/r<t><variant>` from the incumbent, so variants cannot see
each other's edits and can be evaluated concurrently (Alg. 2, "in parallel").
Accepting a candidate fast-forwards `evolve/<domain>` to its commit. The
harness tree hash (`git rev-parse <commit>:<harness path>`) is what the
frontier records, so commits outside the harness never look like a change.
"""
# 【模块说明（中文）】候选隔离用的 git 底层操作（git plumbing）。
# 每个 domain 在自己的分支 evolve/<domain> 上演化；每轮的每个候选 harness 都在
# 自己的 git worktree 中起草与评估，worktree 检出自现任分支 `<domain>/r<t><variant>`，
# 因此各变体互相看不到对方的编辑、且可并行评估（Alg. 2 的 "in parallel"）。
# 接受一个候选时把 evolve/<domain> 快进（fast-forward）到它的提交——现任永远是一个提交。
# frontier 记录的是 harness 树哈希（git rev-parse <commit>:<harness 路径>），
# 所以 harness 之外的提交永远不会被误判为"harness 变了"。

# 延迟类型注解求值
from __future__ import annotations

# shutil：强删 worktree 残留目录（git worktree remove 失败时的兜底）
import shutil
# subprocess：所有 git 命令的执行通道
import subprocess
# Path：仓库与 worktree 路径
from pathlib import Path

# AUTHOR：以固定身份提交（worktree 里没有 user.email/user.name 配置，内联 -c 传入保证 commit 不失败）
AUTHOR = ["-c", "user.email=rrsi@localhost", "-c", "user.name=rrsi"]


def sh(cmd: list[str], cwd: Path | str, check: bool = False) -> subprocess.CompletedProcess:  # 定义 sh：跑一条外部命令并捕获输出
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)  # 在 cwd 下执行，捕获 stdout/stderr 为文本（不继承终端）
    if check and r.returncode != 0:  # 要求成功但失败了
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}: {r.stderr[-800:]}")  # 抛错并带上 stderr 末尾 800 字符（错误详情通常在尾部）
    return r  # 返回 CompletedProcess（调用方自行检查 returncode）


def git(cwd: Path | str, *args: str, check: bool = False) -> subprocess.CompletedProcess:  # 定义 git：sh 的 git 专属封装
    return sh(["git", *AUTHOR, *args], cwd, check=check)  # 命令前统一拼上 git + 固定作者身份（所有提交可归因于 rrsi）


def head(cwd) -> str:  # 定义 head：当前 HEAD 的短哈希
    return git(cwd, "rev-parse", "--short", "HEAD").stdout.strip()  # rev-parse --short 取 7 位短哈希并去尾换行


def rev(cwd, ref: str) -> str:  # 定义 rev：任意引用（分支/提交）的短哈希
    return git(cwd, "rev-parse", "--short", ref).stdout.strip()  # 同上，用于 frontier 记录 commit


def tree_hash(cwd, ref: str, path: str) -> str:  # 定义 tree_hash：ref 下 path 子树的哈希
    return git(cwd, "rev-parse", f"{ref}:{path}").stdout.strip()[:12]  # <ref>:<path> 语法给出该目录的 tree 对象哈希（截 12 位）——harness 内容的指纹，与提交历史无关


def branch_exists(cwd, name: str) -> bool:  # 定义 branch_exists：本地分支是否存在
    return git(cwd, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}").returncode == 0  # --quiet 静默验证 refs/heads/<name>；返回码 0 即存在


def ensure_branch(cwd, name: str, start: str = "HEAD") -> None:  # 定义 ensure_branch：保证演化分支存在
    if not branch_exists(cwd, name):  # 分支还没有
        git(cwd, "branch", name, start, check=True)  # 从 start（默认 HEAD）创建它——baseline 前调用


def worktree_add(repo: Path, path: Path, branch: str, start: str) -> Path:  # 定义 worktree_add：在全新分支上建候选 worktree
    """Fresh worktree on a NEW branch `branch` at `start` (old one removed)."""
    # docstring：在 start 处以新分支 branch 建全新 worktree（旧的先删掉——保证重复起草从干净状态开始）
    worktree_remove(repo, path, branch)  # 清理同路径旧 worktree 与同名旧分支（断点续跑的"重来"语义）
    path.parent.mkdir(parents=True, exist_ok=True)  # 父目录（runs/<domain>/wt/）就绪
    git(repo, "worktree", "add", "-b", branch, str(path), start, check=True)  # -b 新建分支并检出：候选的独立工作副本
    return path  # 返回 worktree 路径


def worktree_remove(repo: Path, path: Path, branch: str | None = None) -> None:  # 定义 worktree_remove：删 worktree（可选连分支）
    if path.exists():  # 目录还在
        git(repo, "worktree", "remove", "--force", str(path))  # --force：不管有没有未提交改动都移除（候选失败即弃）
        shutil.rmtree(path, ignore_errors=True)  # 兜底强删残留目录（ignore_errors 防权限问题中断流程）
    git(repo, "worktree", "prune")  # prune：清掉 git 内部登记的失效 worktree 记录
    if branch and branch_exists(repo, branch):  # 指定了分支且它还存在
        git(repo, "branch", "-D", branch)  # -D 强删（未合并也要删——失败的候选不值得保留）


def is_clean(cwd, path: str) -> bool:  # 定义 is_clean：路径下是否有未提交改动
    return git(cwd, "status", "--porcelain", "--", path).stdout.strip() == ""  # porcelain 输出为空 = 干净


def revert_path(cwd, path: str) -> None:  # 定义 revert_path：把路径恢复到 HEAD 状态
    git(cwd, "reset", "-q", "--", path)  # 从暂存区撤下
    git(cwd, "checkout", "--", path)  # 检出覆盖已跟踪文件
    git(cwd, "clean", "-fdq", "--", path)  # 删除新增的未跟踪文件/目录


def diff_with_new_files(cwd: Path, path: str) -> str:  # 定义 diff_with_new_files：工作区 diff + 新文件的完整文本
    """Working-tree diff of `path` plus the full text of untracked files, so a
    module the proposer just created is reviewed by the critic in full."""
    # docstring：path 的工作区 diff 并含未跟踪文件全文——提案者刚新建的模块也要被 critic 完整审查
    git(cwd, "add", "-N", "--", path)  # -N（intent-to-add）：把新文件登记进索引，使 git diff 能输出其全文
    d = git(cwd, "diff", "--", path).stdout  # 输出统一 diff（含新文件 +N 后的全部内容）
    return d  # 返回 diff 文本（写盘 + 送 critic）


def commit_path(cwd, path: str, msg: str) -> str:  # 定义 commit_path：只提交 harness 路径的改动
    git(cwd, "add", "-A", "--", path)  # -A：该路径下的增/改/删全部入暂存
    git(cwd, "commit", "-q", "--allow-empty", "-m", msg, check=True)  # --allow-empty：没有实质改动也提交（保持每候选一个提交的恒定不变量）
    return head(cwd)  # 返回新提交的短哈希（写进 prep.json / frontier）


def fast_forward(repo: Path, branch: str, commit: str) -> None:  # 定义 fast_forward：把演化分支快进到获胜候选
    """Move refs/heads/<branch> to <commit> if it is a fast-forward."""
    # docstring：仅当 commit 是 branch 的快进（后代）时移动分支指针
    r = git(repo, "merge-base", "--is-ancestor", branch, commit)  # 判断 branch 是否为 commit 的祖先
    if r.returncode != 0:  # 不是祖先 = 不是快进
        raise RuntimeError(f"{commit} is not a fast-forward of {branch}")  # 拒绝：演化历史必须线性、可回放
    git(repo, "update-ref", f"refs/heads/{branch}", commit, check=True)  # update-ref 直接移指针：不等价于 checkout，不碰任何工作区
