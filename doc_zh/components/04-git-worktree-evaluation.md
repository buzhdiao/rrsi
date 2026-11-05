# 04 · git worktree 候选评估机制：evolve/&lt;domain&gt; 分支的 fast-forward

## 为什么用 git

RRSI 的每个候选 harness 必须同时满足：与现任隔离（变体之间互不干扰）、可并行评估、
可完整回放（任何一个历史候选都能再次检出）。git worktree 一次满足三条，而且不需要
容器：worktree 是同一仓库的独立工作目录，检出到不同分支互不污染。

## 分支拓扑

- 每个 domain 一条长期演化分支 `evolve/<domain>`（`rrsi/loop.py:127` 附近
  `self.branch`；由 `rrsi/gitops.py:88` 的 `ensure_branch` 从 HEAD 创建）。分支尖永远
  是**现任 harness**——一个真实的提交。
- 每轮每个候选一条短命分支 `<domain>/r<t><变体>`（如 `coding/r3A`），由
  `worktree_add` 从演化分支头检出（`rrsi/gitops.py:93`；调用点 `rrsi/loop.py:525`）。
  变体 A 看不到变体 B 的编辑，因此 `m` 个候选可以并行起草与评估（Alg. 2 的
  "in parallel"）。
- 临时检出（baseline / heldout / smoke）用 `--detach` 游离 HEAD（`rrsi/loop.py:159`
  附近），不产生分支。

## 提交与快进

提案者编辑完成后，`commit_path`（`rrsi/gitops.py:130`）只把 harness 路径
（`domains/<name>/<harness_path>`）加入暂存并提交，且 `--allow-empty`：即使没有实质
改动也保持"每候选一个提交"的不变量。候选获胜后，`fast_forward`
（`rrsi/gitops.py:136`）先用 `merge-base --is-ancestor` 校验获胜提交确为分支尖的
后代，再用 `update-ref` 移动分支指针（`rrsi/loop.py:373`）——不是 checkout，任何
工作区都不受影响。历史保持线性，`readjudicate` 才能安全重放。

## 内容指纹：tree hash

frontier 记录的不是提交哈希而是 harness **树哈希**：`git rev-parse <ref>:<harness路径>`
（`rrsi/gitops.py:80` 的 `tree_hash`，调用点 `rrsi/loop.py:163`）。提交里混入 harness
之外的改动（README、脚本）不会让树哈希变化；每轮开跑时 `Run.round` 都会比对分支上
的树哈希与 frontier 记录（`rrsi/loop.py:274-277`），不一致即拒绝继续——防止有人
手改分支造成状态漂移。

## 证据链的完整性

diff 的取法也有讲究：`diff_with_new_files`（`rrsi/gitops.py:121`）先 `git add -N`
（intent-to-add）再 diff，让提案者**刚新建的模块**以全文出现在 diff 里——critic 必须
能审到完整的新文件，而不是只看到"new file"占位。worktree 用完即删
（`worktree_remove`，`rrsi/gitops.py:102`，含 `prune` 与强删分支），仓库里留下的
只有演化分支与可从 prep.json / diff.patch 追溯的证据文件。
