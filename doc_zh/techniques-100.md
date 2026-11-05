# Python 高级技法与重要库 100 讲

本讲义从 RRSI 仓库的真实代码中提炼 100 条 Python 技法与重要库用法。每条给出**名称**、
**分类**、**仓库内用例**（文件引用，指向已加中文注释的当前代码）与 2-4 句要点。
按主题分十一类，编号连续（01-100）。

---

## 一、数据建模与 dataclass（01-10）

**01 · @dataclass 声明数据容器**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `TaskResult` / `EvalResult`、`rrsi/selection.py` 的 `Candidate` / `Decision`。
装饰器自动生成 `__init__`/`__repr__`/`__eq__`，字段即类型化属性。RRSI 用它承载"每任务 trial 结果""候选""裁决"等纯数据，替代手写构造器，字段一目了然。

**02 · field(default_factory=list) 可变默认值**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `TaskResult.weights`、`rrsi/config.py` 的 `RRSIConfig.notes`。
可变默认值必须经 `default_factory` 生成，否则会被所有实例共享。`notes: dict = field(default_factory=dict)` 让配置对象天然携带一个可安全追加的兜底字典。

**03 · __post_init__ 构造后规范化**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `TaskResult.__post_init__`。
构造完成后立即补齐派生不变量：没给权重就按 rewards 长度补全 1.0，没给 token 计数就补 None。调用方因此可以"只传 rewards"，规范化逻辑集中一处。

**04 · @property 派生量**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `TaskResult.mean`、`rrsi/selection.py` 的 `Candidate.components`。
把"加权平均奖励"这类可计算的量伪装成属性，调用点无需括号，也不用在落盘结构里冗余存储。`mean` 内部处理了分母为 0 的边界。

**05 · @classmethod 替代构造器**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `EvalResult.from_json` / `EvalResult.load`、`rrsi/config.py` 的 `RRSIConfig.load`。
从 dict / 文件构造实例的逻辑放在类方法里，序列化与反序列化对称成对。`EvalResult` 的 save/load/from_json 三件套是断点续跑的基础。

**06 · asdict 递归序列化**
分类：数据建模。用例：`rrsi/evaluate.py` 的 `EvalResult.to_json`（`asdict(r)`）、`rrsi/config.py` 的 `RRSIConfig.dump`。
`dataclasses.asdict` 递归把嵌套 dataclass 转成纯 JSON 类型。配置快照因此可以原样写进 frontier.json，实验复现不依赖记忆。

**07 · fields() 反射字段名集合**
分类：数据建模。用例：`rrsi/config.py` 的 `RRSIConfig.load`（`known = {f.name for f in fields(cls)}`）。
用 `dataclasses.fields()` 反射出全部合法字段名，JSON 里不认识的键自动分流进 `notes`。新增超参只需加字段，解析代码零改动。

**08 · __dict__ 直出 JSON**
分类：数据建模。用例：`rrsi/selection.py` 的 `Decision.to_json`。
无嵌套对象的 dataclass 不必绕 asdict，`return self.__dict__` 即可 JSON 化。轻量结构用轻量写法，也是"够用就好"的示范。

**09 · PEP 604 联合类型 int | None**
分类：类型注解。用例：`rrsi/config.py` 的 `delta: float | None`、`rrsi/selection.py` 的 `ev: EvalResult | None`。
3.10+ 的竖线联合类型替代 `Optional[float]`，与 `is None` 判空配合表达"缺省语义"。RRSI 里 `delta=None` 特指"去读校准文件"，类型本身就是文档。

**10 · from __future__ import annotations**
分类：类型注解。用例：仓库全部模块第一行 import。
注解延迟求值后，前向引用（如 `-> "EvalResult"`）与容器泛型 `list[dict]` 在旧运行时也不炸。RRSI 统一启用，注解可以放心写类型。

## 二、接口抽象与动态加载（11-16）

**11 · raise NotImplementedError 鸭子接口**
分类：面向对象。用例：`rrsi/domain.py` 的 `Domain.evolve_ids` / `run` / `smoke` 等。
不用 `abc.ABC`，基类方法体直接 `raise NotImplementedError`——实现即文档，未实现的方法在调用瞬间失败。配合类属性默认值（如 `guards` 返回 `[]`），形成"可选方法有默认、必选方法必实现"的轻量契约。

**12 · 类属性作为可覆盖默认**
分类：面向对象。用例：`rrsi/domain.py` 的 `critic_patterns: list = []`、`component_signals: list = []`、`source_exts` 集合。
子类直接声明同名类属性即可覆盖（见 `domains/*/adapter.py` 的 `critic_patterns`）。比构造函数传参更声明式，接口一览即知哪些行为可定制。

**13 · importlib.util 按路径动态加载模块**
分类：元编程。用例：`rrsi/domain.py` 的 `load_domain`（`spec_from_file_location` + `module_from_spec` + `exec_module`）。
domain 适配器不是常规包成员，按 `domains/<name>/adapter.py` 文件路径构造 spec 再执行。新增 domain 不需要动任何 `__init__` 或注册表。

**14 · sys.modules 预注册动态模块**
分类：元编程。用例：`rrsi/domain.py` 的 `sys.modules[spec.name] = mod`。
在 `exec_module` 之前先把空模块注册进 `sys.modules`，dataclass、pickle 等机制通过限定名查找模块时不会失联。动态加载的标准配套动作。

**15 · 导出单例约定（module.DOMAIN）**
分类：架构模式。用例：`rrsi/domain.py` 的 `dom = mod.DOMAIN`，及 `domains/coding/adapter.py:286` 的 `DOMAIN = CodingDomain()`。
适配器文件只需保证模块级存在一个 `DOMAIN` 对象，加载器再注入 `dom.root`。约定优于配置：没有注册、没有装饰器，"文件在+单例在"即插件生效。

**16 · 函数内 import 延迟依赖**
分类：工程习惯。用例：`rrsi/llm.py` 的 `_client_for`（`from anthropic import AnthropicVertex`）、`rrsi/calibrate.py` 的 `pooled`（`from .evaluate import TaskResult, aggregate`）。
重型 SDK 或可能成环的导入放进函数体，首次调用才付出导入成本。这让 `llm.py` 能被无凭据环境安全 import，也让模块依赖图保持单向。

## 三、文件系统与 pathlib（17-28）

**17 · Path.read_text / write_text 一行化 IO**
分类：文件系统。用例：全仓库，如 `rrsi/loop.py` 的 `save_frontier`、`rrsi/history.py` 的 `replace_round`。
读/写整个文本文件不再需要 open/with 三行。RRSI 的所有 JSON 产物都走这一对方法，代码密度显著降低。

**18 · mkdir(parents=True, exist_ok=True)**
分类：文件系统。用例：`rrsi/loop.py` 的 `Run.__init__`、`rrsi/analyst.py` 的 `prepare_rendered`、`rrsi/history.py` 的 `append`。
幂等建目录：父链一次建齐、已存在不报错。`append` 里每次写前都确保父目录在，容忍用户删目录不删进程的场景。

**19 · Path.rglob 递归遍历**
分类：文件系统。用例：`rrsi/propose.py` 的 `Workspace.list_files` / `dump`。
`rglob("*")` 递归枚举 harness 目录全部文件，配合 `p.is_file()` 与 `"__pycache__" not in p.parts` 过滤。生成器式的惰性枚举对大目录也稳。

**20 · Path.glob + sorted 稳定顺序**
分类：文件系统。用例：`rrsi/analyst.py` 的 `load_digests`（`sorted(dg.glob("*.json"))`）、`rrsi/digester.py` 的 glob 动作。
glob 的返回顺序没有承诺，任何"读入后聚合"的场景都先 sorted。摘要回读的顺序稳定，是分析可复现的小细节。

**21 · resolve() 路径囚笼**
分类：安全。用例：`rrsi/propose.py` 的 `Workspace._safe`、`rrsi/digester.py` 的 `_safe`。
拼接后立即 `resolve()` 消解 `..` 与符号链接，再做前缀检查——LLM 给的路径不可信，`../../etc/passwd` 必须在文件触碰之前被拦下。两个 `_safe` 是同一模式的两份实现。

**22 · startswith + os.sep 前缀判定**
分类：安全。用例：`rrsi/propose.py` 的 `Workspace._safe`（`str(p).startswith(str(self.dir) + os.sep)`）。
前缀检查必须拼上路径分隔符，否则 `/x/ab` 会被 `/x/a` 的前缀误放过。`os.sep` 保证 Windows 兼容。

**23 · lstrip("/") 与目录名前缀容错**
分类：字符串处理。用例：`rrsi/propose.py` 的 `Workspace._safe`（剥掉重复的 harness 目录名前缀）。
LLM 常给出带目录前缀或首斜杠的路径，先宽容地规范化再校验，而不是直接报错。对 LLM 协议层"严于律己、宽以待人"的典型取舍。

**24 · Path.unlink(missing_ok=True)**
分类：文件系统。用例：`rrsi/loop.py` 的 `reevaluate`（删可能损坏的旧 eval.json）。
删除不需要先 `exists()` 判断，`missing_ok=True` 把两步合成一步。重测流程因此对"文件在不在"幂等。

**25 · os.path.normpath 规范化相对路径**
分类：文件系统。用例：`rrsi/loop.py` 的 `Run.__init__`（`self.harness_rel`）。
`domains/<name>/<harness_path>` 三段拼接后统一规范化，后续 diff/commit/tree-hash 全部复用同一字符串。路径只算一次，处处一致。

**26 · 后缀白名单闸门**
分类：安全。用例：`rrsi/propose.py` 的 `Workspace.write`（`p.suffix not in self.exts`）。
新建文件的扩展名必须在 domain 声明的白名单里，阻止 LLM 写二进制或可执行脚本。闸门放在 IO 层而非提示词层，才真正可靠。

**27 · relative_to 输出相对路径**
分类：文件系统。用例：`rrsi/propose.py` 的 `Workspace.list_files` / `dump`（`p.relative_to(self.dir)`）。
给 LLM 与日志的路径一律相对于 harness 根，去掉机器相关的绝对前缀。提示词可移植、diff 可对比。

**28 · p.stat().st_size 元数据内联**
分类：文件系统。用例：`rrsi/propose.py` 的 `Workspace.list_files`（`f"... ({p.stat().st_size}B)"`）。
遍历时顺手取文件大小，让 LLM 在读文件前就能估计成本。一次 stat 调用换更聪明的读取决策。

## 四、JSON 与 JSONL（29-37）

**29 · JSONL 追加记录（append-only log）**
分类：数据格式。用例：`rrsi/history.py` 的 `History.append`、`rrsi/loop.py` 的 `attribute`（attribution.jsonl）。
一行一个 JSON 对象、`open(path, "a")` 追加写：崩溃不丢已写行，读取可流式分片。编辑历史与归因记分板都选 JSONL 而非单一大 JSON，正是要这种"只进不改"的可审计性。

**30 · JSONL 全量回读**
分类：数据格式。用例：`rrsi/history.py` 的 `History.records`（`splitlines()` + 逐行 `json.loads`）。
按行 split 后逐行解析并跳过空行，天然容忍文件尾部不完整的一行。配合 29 构成最小可用的持久化协议。

**31 · ensure_ascii=False 中文不转义**
分类：数据格式。用例：`rrsi/history.py` 的 `append`、`rrsi/analyst.py` 的摘要落盘。
序列化保留非 ASCII 原文，日志人可直接读。凡涉及自然语言内容（假设、失败模式）的 JSON 都该如此。

**32 · indent=1 产物风格**
分类：数据格式。用例：`rrsi/loop.py` 的 frontier/decisions/proposal 落盘、`rrsi/calibrate.py` 的 `write`。
统一的缩进风格让所有机器产物都方便 `diff` 与人工抽查。团队内约定一个缩进值并坚持到底，比"可读性最佳实践"更实际。

**33 · JSON 解析容错循环**
分类：健壮性。用例：`rrsi/propose.py` 的主循环、`rrsi/analyst.py` 的 `analyze`、`rrsi/digester.py` 的 `digest_task`。
LLM 输出按"可能坏"设计：`try json.loads` 失败就把原文回填进交互日志并报 `ERROR: invalid JSON`，让模型下一轮自我修正。错误信息回给能修复它的一方，是 agent 协议设计的核心。

**34 · 修复式 JSON 提取（围栏/前缀/截尾）**
分类：健壮性。用例：`rrsi/llm.py` 的 `extract_json`。
三层修复：剥 ``` 围栏、截掉 JSON 定界符之前的废话、截到最后的闭合括号。`re.S` 让 `.*?` 跨行匹配。宽容解析把"几乎对"的输出救回来，省掉整轮重试。

**35 · 环境变量解析为结构化配置**
分类：配置管理。用例：`rrsi/llm.py` 的 `_PROJECTS`（`RRSI_VERTEX_PROJECTS` 逗号拆分 + 列表推导）。
一个逗号分隔的环境变量变成项目字典列表，`if p.strip()` 顺手滤空段。凭据与部署差异走环境，代码保持纯净。

**36 · dict 合并展开 **observed**
分类：字典技巧。用例：`rrsi/calibrate.py` 的 `calibrate` 返回值（`..., **observed`）。
重复评估时才有的观测量（`S_per_eval`、`max_abs_diff`）用 `**` 按条件并进结果字典。一个返回结构同时服务两种模式，不必定义两个 schema。

**37 · 字段投影字典推导**
分类：字典技巧。用例：`rrsi/history.py` 的 `render`（`{k: r.get(k) for k in keys if r.get(k) is not None}`）。
把完整记录投影到固定字段白名单并剔除 None 键，控制给 LLM 的上下文体积。"给模型看什么"与"存了什么"解耦。

## 五、正则与文本处理（38-47）

**38 · (pattern, why) 元组表驱动黑名单**
分类：正则。用例：`rrsi/critic.py` 的 `GENERIC_PATTERNS`、`precheck`。
每条规则是"正则 + 拒绝理由"的元组，命中即回报理由。规则表与执行循环分离，domain 追加自己的模式（`domain.critic_patterns`）零成本。

**39 · re.compile 预编译热正则**
分类：正则。用例：`rrsi/components.py` 的 `_STRING_LINE`、`rrsi/digester.py` 的 `DENY_BASH`。
模块级 compile 一次、每次调用只 search。对每轮都要跑的安全检查，省掉重复编译，也让正则拥有一个可注释的常量名。

**40 · \b 词边界防误杀**
分类：正则。用例：`rrsi/digester.py` 的 `DENY_BASH`（`\brm\b`、`\bmv\b`）。
危险命令匹配必须加词边界，否则 `confirm` 会被 `rm` 命中、`remove` 会被 `move` 命中。安全正则的假阳性同样有代价——它拒绝的是合法工作。

**41 · re.split(r"[|;&]+") 多分隔符切分**
分类：正则。用例：`rrsi/digester.py` 的 `_bash_ok`（逐管道段检查白名单）。
一条 shell 命令按管道与命令分隔符切段，每段首词都必须在只读白名单里。`grep x; rm y` 这种"夹带"在第 2 段就被拦住。

**42 · re.search(r"\{.*\}", out, re.S) 抢救 JSON**
分类：正则。用例：`rrsi/critic.py` 的 `review`（整体解析失败时）。
DOTALL 模式下贪婪匹配第一个 `{` 到最后一个 `}`，从混入闲话的输出里抠出 JSON 对象。作为 34 的第二道保险，双重兜底后再降级为"按拒绝处理"。

**43 · re.match 行首步号解析**
分类：正则。用例：`rrsi/propose.py` 的 `_read_trace`（`r"\[step (\d+)\]"`）。
从渲染轨迹逐行抓取步号，实现 from_step/to_step 的切片过滤。抓不到步号的行（判分区等）按上下文标志保留。

**44 · raw 三引号字符串写正则**
分类：正则。用例：`rrsi/digester.py` 的 `DENY_BASH`（`r"(>>?|`|\$\(|..."`）。
`r"..."` 让 `\b`、`\$` 原样传给 re 模块，三引号容纳跨行长模式。安全规则这种"长而 Evil"的正则，raw + 三引号是唯一可读的写法。

**45 · count() 精确替换前置检查**
分类：字符串处理。用例：`rrsi/propose.py` 的 `Workspace.edit`（`src.count(old)` 判 0/1/多次）。
替换前统计 old 出现次数：0 次报"找不到"、多次报"请给更长上下文"、恰 1 次才动手。LLM 的 edit_file 必须这样防呆，否则一次替换可能改错位置。

**46 · startswith 多条件链**
分类：字符串处理。用例：`rrsi/components.py` 的 `text_only`（排除 `+++`/`---` 文件头）、`rrsi/digester.py` 的判分区检测。
diff 行的语义全靠行首前缀：`+`/`-` 是改动、`+++` 是文件头。startswith 链是最直白的分词器，够用且零依赖。

**47 · str.format 占位注入提示词模板**
分类：字符串处理。用例：`rrsi/propose.py` 的 `SYSTEM_TMPL.format(domain_brief=..., components=...)`、`rrsi/analyst.py` / `rrsi/critic.py` 同模式。
巨型系统提示词写三引号模板，`.format()` 注入 domain 段落与词表。注意模板里的 JSON 花括号要双写成 `{{...}}`——RRSI 的 critic 提示词是现成示例。

## 六、迭代与推导式（48-58）

**48 · 集合推导去重断言**
分类：迭代。用例：`rrsi/history.py` 的 `tried`（`{r["component"] for r in self.measured()}`）、`rrsi/components.py` 的 `novelty`（`set(edit_components)`）。
一行得到已尝试组件集合；`novelty` 先 `set()` 再计数，同一编辑包里重复声明的组件不会重复加分。

**49 · 字典推导计数表**
分类：迭代。用例：`rrsi/history.py` 的 `attempted` / `accepted_edits`（`{c: [] for c in K}`）、`rrsi/loop.py` 的 `readjudicate`（`counts = {k: 0 for k in ...}`）。
以词表 K 为骨架预填零值，保证输出的键空间完整——"没试过的组件"显式存在而不是缺失。后续累加只改值不改键。

**50 · any()/all() 生成器短路**
分类：迭代。用例：`rrsi/components.py` 的 `has_evidence` / `text_only`、`rrsi/propose.py` 的保留槽位检查。
存在性/全称断言用生成器表达式 + any/all，短路即返回。`any(re.search(p, diff) for p in pats)` 比手写循环少三行，语义还更清楚。

**51 · enumerate(start=1) 行号标注**
分类：迭代。用例：`rrsi/digester.py` 的 `_grep` / `_read_file`（`f"{i}: {line[:400]}"`）。
grep 结果自带原始行号，LLM 引用证据时能给出精确位置（"step 42"、"line 128"）。`start=1` 让下标直接等于人类行号。

**52 · zip 平行遍历**
分类：迭代。用例：`rrsi/evaluate.py` 的 `aggregate`（`zip(tr.rewards, tr.weights)`）、`rrsi/loop.py` 的 `zip(cands, decisions)`。
奖励与权重、候选与裁决这两类"同序数组"永远 zip 着走，避免下标对位错误。缺失 trial 记 0 而不删槽位，保证 zip 后长度恒一致。

**53 · next() + 生成器取首个匹配**
分类：迭代。用例：`rrsi/propose.py` 等（`act = next((x for x in act if isinstance(x, dict)), None)`）。
LLM 违规输出数组时，取第一个 dict 元素作为降级方案，`None` 默认值统一交给后续分支。取"第一个合格品"的标准写法。

**54 · 负索引切片取尾部**
分类：迭代。用例：`rrsi/loop.py` 的 `build_traces`（`ranked[-self.cfg.n_success_traces:]`）、`rrsi/loop.py` 的 `scoreboard`（`rows[-20:]`）。
升序排完后 `[-n:]` 直接取最高分/最近 n 条。日志回看与"成功透镜"选样都靠这一刀。

**55 · max(key=lambda) 做 argmax**
分类：迭代。用例：`rrsi/selection.py` 的 `select_round`（`max(adm, key=lambda cd: cd[1].S)`）。
在可接纳候选中取 `S'` 最大的，一行完成论文的 argmax。`max` 遇并列保持先见者胜，天然稳定。

**56 · sorted(key=...) 升序任务表**
分类：迭代。用例：`rrsi/loop.py` 的 `build_traces`（`key=lambda kv: kv[1].mean`）、`rrsi/analyst.py` 的 `task_table`、`rrsi/propose.py` 的 `_list_traces`。
按任务均分升序排列，最差任务排最前——LLM 的注意力会先落在最上面，排序即注意力工程。

**57 · 下标切片两两组合**
分类：迭代。用例：`rrsi/calibrate.py` 的 `calibrate`（`[abs(a-b) for i, a in enumerate(scores) for b in scores[i+1:]]`）。
双层推导枚举所有无序对，计算重复评估的最大分数差。i+1 起始是"组合而非排列"的惯用截断。

**58 · setdefault 累积字典**
分类：字典技巧。用例：`rrsi/calibrate.py` 的 `pooled`（`per.setdefault(t, TaskResult(...))`）、`rrsi/propose.py` 的 `_list_traces`（`modes_by_task.setdefault(...)`）。
"取不到就建空容器"一行完成 get-or-create，随后直接追加。多次评估拼接、失败模式聚合都是同一个动作。

## 七、数值与统计（59-68）

**59 · math.cos 余弦退火**
分类：数值计算。用例：`rrsi/schedule.py` 的 `edit_budget`（`0.5 * (1.0 + cos(pi*t/T))`）。
半周期余弦从 1 平滑降到 0，映射出 b_max -> b_min 的退火调度。一条公式替代分段函数，梯度处处连续。

**60 · ceil(round(v, 9)) 浮点守卫**
分类：数值计算。用例：`rrsi/schedule.py` 的 `edit_budget`（原注释：防 1.0000000002 -> 2）。
`ceil` 前先 `round(v, 9)`，吃掉 t=T 处的浮点残渣，否则名义上的 b_min=1 会算成 2。"先归整再取整"是 ceil/floor 的标准护栏。

**61 · math.inf 作哨兵**
分类：数值计算。用例：`rrsi/history.py` 的 `yield_g`（`-math.inf` 表示"近期无测量"）。
max-accumulation 的空集初值取 -inf，任何真实增益都自然胜出；展示层再把 -inf 翻译成 None。论文 Eq. (yield) 的 `max(empty) = -inf` 直接落成代码。

**62 · statistics.pstdev / stdev**
分类：统计。用例：`rrsi/calibrate.py` 的 `bootstrap_se`（pstdev）与 `calibrate`（stdev）。
重采样分布的总体标准差作标准误；多次独立评估的样本标准差作单次评估波动。库函数替代手写方差，免掉自由度纠结。

**63 · random.Random(seed) 独立可复现实例**
分类：统计。用例：`rrsi/calibrate.py` 的 `bootstrap_se`（`rng = random.Random(seed)`，seed=7）。
不用全局 random，而是带种子的局部实例：同一份评估数据永远得到同一个 delta。校准结果可复现，审计才可能。

**64 · randrange 有放回重采样**
分类：统计。用例：`rrsi/calibrate.py` 的 `bootstrap_se`（每个任务内 `for _j in range(n): i = rng.randrange(n)`）。
bootstrap 的本质：与原样本等量、有放回地重抽，重复千次看估计量的离散度。任务内重采样（而非全局）保持任务间的独立结构。

**65 · sqrt(2) 方差合成**
分类：统计。用例：`rrsi/calibrate.py`（`sd_null = stdev * sqrt(2)`、`sd_boot = sqrt(2)*se*sqrt(k_pooled/k_0)`）。
两个独立同方差估计之差的方差是单者的两倍，标准差乘 sqrt(2)。评估噪声带 delta 的推导核心就这一步。

**66 · round(x, 6) 落盘精度收敛**
分类：数值计算。用例：`rrsi/history.py` 的 `append_candidate`（delta_S/delta_C/S 各自 round）、`rrsi/calibrate.py` 的返回值。
写盘前统一舍入，JSONL 行宽可控、两份产物可逐字节 diff。噪声级别以下的尾数不参与比较。

**67 · f"{v:+.4f}" 符号化格式**
分类：格式化。用例：`rrsi/selection.py` 的 `cost_rule`（`f"{delta_S:+.4f}"`）、`rrsi/loop.py` 的日志。
`+` 强制显示正负号，增益/损耗在日志里一眼可辨；`.4f` 对齐小数位。裁决理由是人类要读的，格式就是它的 UX。

**68 · 数字下划线字面量**
分类：可读性。用例：`rrsi/propose.py` 的 `TRACE_READ_CAP = 60_000`、`rrsi/critic.py` 的 `[:120_000]`。
`60_000` 比 `60000` 的千位分组直接可读。上限常量一眼看出量级，改参数时不易多写一个零。

## 八、并发与线程（69-74）

**69 · ThreadPoolExecutor 上下文管理器**
分类：并发。用例：`rrsi/loop.py` 的 `round`（并行评估候选）、`rrsi/analyst.py` 的 `analyze`（并行摘要）。
`with` 块结束时自动 join 并回收线程，异常传播不泄漏。IO 密集（跑基准、调 LLM）选线程池而非进程池，GIL 不是瓶颈。

**70 · ex.map 阻塞收集**
分类：并发。用例：`rrsi/loop.py:344`（`list(ex.map(lambda c: self._evaluate(...), live))`）。
`map` 保持输入顺序返回结果迭代器，`list()` 一句话等全部完成。异常在迭代处浮出，配合 worker 内部的自捕获形成两层防线。

**71 · 闭包作 worker 函数**
分类：并发。用例：`rrsi/analyst.py` 的 `run(r)`（捕获 rendered_dir/digests_dir/domain 后提交给线程池）。
局部闭包免传一长串参数，`ex.map(run, valid)` 提交即并行。闭包捕获的路径都是只读的，线程安全由不可变性保证。

**72 · 并行度常量化**
分类：并发。用例：`rrsi/analyst.py` 的 `DIGEST_PARALLELISM = 6`、`rrsi/config.py` 的 `eval_parallel: int = 1`。
并行度是显式配置而非隐藏魔法：摘要子代理固定 6，候选评估默认 1（跑基准太贵，宁串行）。旋钮露在配置层，压测才有的放矢。

**73 · max(1, n) 并行度下限**
分类：并发。用例：`rrsi/loop.py:343`（`max_workers=max(1, cfg.eval_parallel)`）。
配置可能被填成 0，`ThreadPoolExecutor(max_workers=0)` 会直接抛错。`max(1, ...)` 一处小防御保住"配置错了也能跑"的底线。

**74 · threading.Lock 保护懒初始化**
分类：并发。用例：`rrsi/llm.py` 的 `_clients_lock` + `_clients` 字典。
多线程同时首次调用同一项目的客户端时，锁保证 AnthropicVertex 只构造一次。双重检查（先 get、锁内再 get）避免每次调用都抢锁。

## 九、子进程与 git 驱动（75-85）

**75 · subprocess.run(capture_output=True, text=True)**
分类：子进程。用例：`rrsi/gitops.py` 的 `sh`、`rrsi/digester.py` 的 bash 工具。
一次调用拿到 `returncode`/`stdout`/`stderr`，text=True 直接给 str。gitops 用它包住所有 git 命令，错误信息从 `r.stderr` 尾部截取。

**76 · check 参数的返回码协议**
分类：子进程。用例：`rrsi/gitops.py` 的 `sh(cmd, cwd, check=...)`（`check=True` 时抛 RuntimeError）。
同一个 `sh` 支持两种风格：探测类调用（branch_exists）自己看返回码，命令式调用（worktree add）要求 check。错误信息带 stderr 末 800 字符，git 的报错恰在尾部。

**77 · sys.executable 复用当前解释器**
分类：子进程。用例：`rrsi/driver.py` 的 `base_args = [sys.executable, str(entry), ...]`。
驱动器不假设 `python3` 在 PATH，而是用启动自己的解释器去跑每一轮子进程。venv 内外行为一致。

**78 · stdout/stderr 合流落盘**
分类：子进程。用例：`rrsi/driver.py`（`stdout=lf, stderr=subprocess.STDOUT`）。
每轮子进程的输出全部追加进 `logs/r<t>.log`，时序不乱、事后可查。驱动器主进程只打一行"日志在哪"。

**79 · shell=True + 白名单双闸**
分类：安全。用例：`rrsi/digester.py` 的 bash 动作（`_bash_ok` 两层检查后才 `subprocess.run(cmd, shell=True, cwd=traces_dir)`）。
给 LLM 的 shell 能力必须受限：先正则黑名单（重定向/替换/危险命令），再按段白名单（首词必须只读）。cwd 限制在轨迹目录，timeout 兜底。

**80 · subprocess timeout 防卡死**
分类：子进程。用例：`rrsi/digester.py` 的 `BASH_TIMEOUT = 30`（`timeout=BASH_TIMEOUT`）。
`grep 大文件 | head` 写不好就会永远跑不完，30 秒超时把挂起变成可处理的 TimeoutExpired 异常。给外部世界的每次调用都要有钟。

**81 · git -c 内联身份**
分类：git 集成。用例：`rrsi/gitops.py` 的 `AUTHOR = ["-c", "user.email=rrsi@localhost", ...]`。
不依赖容器/CI 里的 git 全局配置，每条提交命令内联注入作者身份。所有候选提交统一署名 rrsi，来源可辨。

**82 · git status --porcelain 机器可读判定**
分类：git 集成。用例：`rrsi/gitops.py` 的 `is_clean`。
porcelain 输出为空即干净，专为脚本设计、跨版本稳定。人类格式（--long）绝不该被程序解析。

**83 · git rev-parse ref:path 子树哈希**
分类：git 集成。用例：`rrsi/gitops.py` 的 `tree_hash`（frontier 记录 harness 内容指纹）。
`<ref>:<path>` 直接给出某提交下某目录的 tree 对象哈希。harness 变没变看它即可，与提交信息、外部文件统统无关——这是"内容寻址"思维在流程里的落地。

**84 · add -N / merge-base --is-ancestor / update-ref 组合**
分类：git 集成。用例：`rrsi/gitops.py` 的 `diff_with_new_files`（`add -N`）、`fast_forward`（`--is-ancestor` 校验 + `update-ref` 移指针）。
`add -N`（intent-to-add）让新文件进 diff 全文；祖先判定保证只做真快进；`update-ref` 移动分支而不碰任何工作区。三招合成"线性、可回放、零打扰"的演化历史。

**85 · git worktree add -b / --detach / prune**
分类：git 集成。用例：`rrsi/gitops.py` 的 `worktree_add`（候选用 `-b` 新分支）、`rrsi/loop.py` 的 `checkout`（临时用 `--detach`）、`worktree_remove`（收尾 `worktree prune`）。
worktree 是"一个仓库多个工作目录"：候选互相隔离又共享对象库，评估并行不串行。用完 remove + prune 清干净登记与磁盘。

## 十、LLM 调用模式（86-92）

**86 · itertools.count 全局轮转游标**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `_rr = itertools.count()`（每次调用 `next(_rr)` 取起点）。
无锁的自增游标把并发请求的起点打散，配 `% n` 轮转实现对多个 GCP 项目的负载均衡。count 的 `next` 是原子的，够用且零成本。

**87 · 指数退避 min(2**attempt, 30)**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `generate`（`time.sleep(min(2 ** attempt, 30))`）。
1、2、4、8...秒指数增长，封顶 30 秒。对限流与瞬时故障是最经济的重试节奏，封顶防止长等待拖死整轮。

**88 · failover 轮转 (start + attempt) % n**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `generate`（`idx = (start + attempt) % n`）。
每次重试换下一个项目：一个项目配额耗尽或区域故障时自动漂移。轮转与退避叠加，重试不再是对着同一堵墙撞。

**89 · cache_control ephemeral 前缀缓存**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `generate`（把 `cache_prefix` 作为独立 content block 打上 `{"type": "ephemeral"}`）。
宪法 + 模式库 + 稳定上下文作为前缀块缓存，40 轮交互只为读它付一次钱。调用方把"稳定部分"与"易变部分"分开传，是省钱的第一杠杆。

**90 · json_only 后缀硬约束**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `_JSON_SUFFIX`（"Output ONLY a single valid JSON object..."）。
在 system prompt 末尾追加一行硬约束，把"期望 JSON"变成"只许 JSON"。约束放在系统侧而非用户侧，逐轮生效不被遗忘。

**91 · 空响应按失败处理**
分类：LLM 工程。用例：`rrsi/llm.py` 的 `generate`（`if text: ... else last_err = RuntimeError("empty response")`）。
返回 200 但内容为空同样是故障，掉进重试轮转而不是把空串传给上层。协议边界上的每个"不可能"都值得一次显式检查。

**92 · 异常转文本反馈给模型**
分类：agent 协议。用例：`rrsi/propose.py` / `rrsi/digester.py`（`except Exception as e: result = f"ERROR: {e}"`）。
工具抛的异常不杀死 agent 循环，而是格式化成 `[result] ERROR: ...` 回填进交互日志，让模型自己改参数重试。agent 框架的鲁棒性大半来自这条小规则。

## 十一、CLI、日志与工程化（93-100）

**93 · argparse 子命令 + 动态加参循环**
分类：CLI。用例：`rrsi.py` 的 `main`（`sub.add_parser(...)`，`for name in OVERRIDES: ap.add_argument(...)`）。
每个子命令一个 parser，超参覆盖项用循环批量注册并按名单选 int/float。新增一个可调超参只需在 OVERRIDES 里加一个名字。

**94 · RawDescriptionHelpFormatter + __doc__ 帮助**
分类：CLI。用例：`rrsi.py` 的 `argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)`。
模块 docstring 即帮助文本，格式保持原样（不重排）。用法示例紧跟代码，改代码时顺手就能更新。

**95 · sys.path.insert 支持单文件直跑**
分类：打包与运行。用例：`rrsi.py:51-52`（`ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))`）。
把仓库根塞进 sys.path 后，`python3 /any/path/rrsi.py --domain eng status` 无需 pip install 也能 import rrsi 包。随后的 import 标 `# noqa: E402` 说明次序是有意为之。

**96 · sys.exit(str) / raise SystemExit 用户级错误**
分类：工程习惯。用例：`rrsi/loop.py` 的 `frontier` / `delta`、`rrsi/domain.py` 的 `load_domain`、`rrsi.py` 的 heldout 分支。
配置缺失、顺序错误这类"用户能修"的问题用 `SystemExit(带补救建议的消息)`，而非让 traceback 吓人。"no frontier.json; run `baseline` first" 这种报错本身就是文档。

**97 · print(flush=True) 实时日志**
分类：工程习惯。用例：`rrsi/loop.py` 的 `log`、`rrsi/driver.py` / `analyst.py` 的进度输出。
子进程输出被重定向到日志文件时，无 flush 的 print 会攒缓冲，观测者看不到进展。每条带 `[rrsi:domain] HH:MM:SS` 前缀，多 domain 日志混流也能分拣。

**98 · time.strftime 时间戳**
分类：工程习惯。用例：`rrsi/loop.py` 的 `log`、`rrsi/history.py` 的 `append`（`rec.setdefault("ts", ...)`）。
日志与历史记录统一 `"%Y-%m-%d %H:%M:%S"` 本地时间戳。`setdefault` 保证调用方给了时间戳时不被覆盖。

**99 · 哨兵文件 STOP + 熔断计数**
分类：运维模式。用例：`rrsi/driver.py` 的 `stop = run_dir / "STOP"`（每轮开头检查）与 `infra` 连续失败计数（达到 `MAX_CONSECUTIVE_INFRA=3` 即退出）。
协作式停机：`touch STOP` 在当前轮完成后生效，不杀正在跑的评估。连续 3 次未落定则熔断，坏环境烧不掉整个预算。文件即信号，不需要任何进程间协议。

**100 · pyproject 打包：extras / packages.find / pytest 配置**
分类：打包。用例：`pyproject.toml`（`[project.optional-dependencies]` 的 agentic/coding/dev；`[tool.setuptools.packages.find] include=["rrsi*"]`；`[tool.pytest.ini_options]` 的 testpaths/addopts）。
核心依赖与基准运行器依赖分层：搜索核心只装 anthropic，harbor/LiteLLM 各归 extras，环境互不污染。`pip install -e ".[dev]"` 与 `pip install -e ".[agentic]"` 的双环境策略直接写在 README。
