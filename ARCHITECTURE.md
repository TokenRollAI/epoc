# ARCHITECTURE — epoc (Epoch Context) 三层 Cache-First 架构与上下文引擎详解

> 本文档面向希望深入理解、调优或扩展 **`epoc`（前身原型 `CPAT`）** 的工程师，完整拆解核心架构层次（关于外部多客户端网关 `ShadowLedger` 与 `codex` / `claude code` / `agy` / `kimi` / `pi` / `dsh` 六宿主接入设计，请同步参阅 [ROADMAP.md](ROADMAP.md)）：
> 1. **第一性原理：`50:1` 前缀缓存经济学与零后缀偏移定律（Zero-Suffix Shift Law）**
> 2. **生产主形态：三层 Cache-First Pi 插件架构（`extensions/pi-cpat.ts` + `src/sidecar/tool-projector.ts`）**
> 3. **内核引擎：独立 `ContextHarness`（`src/harness/` + `src/sidecar/`）与事务式 `context_update` Patch 引擎（`src/runtime/`）**

---

## 1. 第一性原理：`50:1` 前缀缓存经济学

现代大模型推理服务（如 DeepSeek、Anthropic、OpenAI Prefix Caching）普遍采用极高折扣的 KV Cache 命中定价。以 DeepSeek V4.1 Flash 为例：

| 计费项 | 单价（USD / 1M tokens） | 相对倍率 |
| :--- | :---: | :---: |
| 前缀缓存命中读取（$P_{\text{hit}}$） | **`$0.003`** | **`1×`（基准）** |
| 未命中输入预填充（$P_{\text{miss}}$） | **`$0.150`** | **`50×`** |
| 模型生成输出（$P_{\text{out}}$） | **`$0.600`** | **`200×`** |

### 1.1 为什么“固定步频中段压缩”在 400K 窗口下必亏？

假设上下文累计到 $150\text{K}$ 时，插件将前 $80\text{K}$ 历史压缩为 $5\text{K}$ 摘要（表面省下 $\Delta T_{\text{saved}} = 75\text{K}$），但切点之后保留了 $T_{\text{suffix}} = 70\text{K}$ 的近期消息。由于注意力掩码的位置编码与精确字节前缀匹配规则，**切点后 $70\text{K}$ 消息的 KV Cache 在下一轮全部失效**：

1. **回溯尾税（Suffix Shift Penalty）**：下一轮立刻多交 $70\text{K} \times (\$0.15 - \$0.003)/\text{M} = \$0.0103$；
2. **压缩器自身开销（Compressor Tax）**：若调用 LLM 生成摘要，还需额外支付一次输入与高价输出（`$0.60/M`）；
3. **每轮微薄回本（Per-Turn Cache Saving）**：省掉 $75\text{K}$ 已缓存前缀，每轮仅省 $75\text{K} \times \$0.003/\text{M} = \$0.000225$——**至少要连续无变动再跑 $46$ 轮才能收回一次压缩的尾税！**

由此导出 CPAT 的核心架构铁律——**零后缀偏移定律（Zero-Suffix Shift Law）**：

> **任何上下文缩减操作，要么发生在字节进入 KV Cache 之前（Write-Time Pre-Cache），要么发生在当前轨迹的最末端边界（令受影响后缀 $T_{\text{suffix}} \equiv 0$），要么发生在窗口硬上限的纪元换代点并直接派生复用温热前缀。**

---

## 2. 三层 Cache-First 生产架构（`extensions/pi-cpat.ts`）

CPAT 在宿主生命周期的四个精确钩子（`tool_result`、`before_agent_start`、`before_provider_request`、`session_before_compact`）上构建了互不干扰的三道防线：

### 2.1 Layer 1：入窗前超大 `tool_result` 门控与宿主截断恢复（`src/sidecar/tool-projector.ts`）

当工具调用完成、尚未拼入下一轮 LLM 请求时，`pi.on("tool_result")` 立即执行同步（`<3ms`、`$0` LLM 成本）检查：

1. **宿主静默截断全量恢复（Blind-Spot Recovery）**：
   - Pi 内置 `bash` 默认在 `50KB / 2,000 行` 处执行**尾截断**（丢弃首屏编译报错/首个失败用例），并将完整输出写入临时文件 `event.details.fullOutputPath`；CPAT 自动读取该完整文件（上限 8 MiB 安全阈值）。
   - Pi 内置 `read` 默认在 `50KB / 2,000 行` 处执行**头截断**（`event.details.truncation.truncated === true`）；CPAT 自动从磁盘源文件补齐完整行，并记录精确 `baseLineOffset`（例如 `read(path, offset=100)` 的第 43 行映射为真实文件行 `L142`）。
2. **100% 全量无损建索引**：
   - 恢复后的完整原文写入 `SearchIndex`（内存窗口 BM25 + 正则 + 真实行号切片）与本地快照目录 `raw-index/<blockId>.txt`。
3. **信息熵感知门控（Entropy-Aware Gate）**：
   - 采样最多 400 行掩码归一化（剥离数字、十六进制地址、Hash、时间戳）计算行模板冗余度 `redundancyRatio`；
   - 满足以下任一条件即触发入窗前投影：
     - 宿主已发生截断且全量 `>= 3,500` tokens（`host_truncated_recovery`）；
     - 代码/Shell 工具（`bash`/`read`/`grep`/`find`/`ls`）单次输出 `>= 24,000` tokens（`coding_or_shell_oversized`）；
     - JSON 导出或高重复日志（`redundancyRatio >= 0.36`）单次输出 `>= 28,000` tokens（`high_redundancy_dump`）；
     - 任意工具单次超过绝对内联上限 `96,000` tokens（`hard_cap_exceeded`，确保 `<96K` 的高密度自然语言长文/小说能无损直通 KV Cache）。
4. **高信号确定性投影结构**：
   - 按工具形态输出专用紧凑视图（`~800–1,400` tokens）：
     - `grep/rg`：按文件聚合命中计数 + 每文件前 3 条行号预览；
     - `JSON`：顶层 Schema / 数组长度 + 首尾样本记录；
     - 代码与测试日志：首部 18 行 + **全文 `Lstart-Lend` 函数/类符号跨度表**（`extractCodeSymbolSpans`）+ **跨全文扫描提取的 `ERROR / Traceback / AssertionError` 诊断块** + 尾部 20 行退出摘要。

### 2.2 Layer 2：显式任务边界与历史错误假设折叠（`ScopeTree` + `AC3`）

针对会话尚未撞击 `384K` 上限（如仅 `60K–150K`），但前序探索走入死胡同留下大量错误推理定势（Context Pollution）或完成独立子任务的场景，CPAT 将 `ScopeTree`（`src/runtime/tree.ts`）与 **`AC3` 第一阶段权威源隔离原则**接入三种显式边界触发器：

1. **Agent 主动提交边界（`context_checkpoint` 工具）**：
   - Agent 仅在子任务阶段完结（`mode: "subtask_complete"`）或确认前序假设证伪（`mode: "pollution_purge"`）时调用，声明 `verified_facts` 与 `purged_assumptions`。
   - 当前回合结束（`turn_end`）时，CPAT 将上一个锚点 `lastFoldAnchorEntryId` 与当前回合之间全部中间态 `assistant` + `toolResult` 成对链条发出原生持久 `context_edit(targetId, null)` 折叠，仅保留用户原始提问与 `[cpat:scope_checkpoint #scope_N]` 洁净结论块。
2. **用户显式纠偏 / 切新需求边界（`before_agent_start`）**：
   - 在新用户消息落盘前识别显式边界标签（`[TASK_BOUNDARY]` / `[RESET_CONTEXT]`）或中英文纠偏/切题指令（如 `"Ignore the previous hypothesis..."`、`"前面的假设错了..."`、`"切新需求：..."`）。
   - **严格执行 `AC3` 权威源隔离**：生成的 `[cpat:scope_checkpoint]` **物理排除全部旧 Assistant 文本**（零幻觉残留），仅列出已入库工具原件的客观地标（`#b_001..#b_k` 文件路径与符号行号）；首轮通过 `context` 即刻剥离旧污染链，并在首轮 `turn_end` 持久化为 `context_edit(targetId, null)`，保证新阶段 Turn 1 与 Turn 2 拥有 **100% 字节一致的前缀**！
3. **交互式手动命令（`/cpat-fold`）**：
   - 用户在终端随时输入 `/cpat-fold [备注]` 即可在下一轮提问边界执行零后缀栈帧折叠。

### 2.3 Layer 3：`384K` 同源前缀派生摘要（Cache-Forked Epoch Summary）+ 多纪元无损目录

当超长多代会话（如 `500K → 2M → 10M` tokens）真正跑满窗口高水位（`384K`）触发 `session_before_compact` 时：

1. **只读捕获温热线路请求（`before_provider_request`）**：
   - 每轮请求发出前只读记录 `lastWireRequest = { model, tools, messages }`（返回 `undefined`，零侵入）。
2. **同源前缀派生摘要请求（`runCacheForkEpochSummary`）**：
   - 撞线瞬间，复用刚在服务端温热的 `model + tools + messages` 完整前缀数组，仅在最末尾追加一条 `<250` tokens 的跨纪元因果账本归纳指令（附带 `previousSummary` 跨纪元接力与最近 1 轮增量 `tailDelta`），直连 `/chat/completions`（`max_tokens: 900`, `stream: false`）。
   - 因为前 `~360K–380K` tokens 与几秒前的主会话请求**逐字节 100% 一致**，该摘要调用直接享受 **95%+ KV Cache 命中率**（输入成本仅 $380\text{K} \times \$0.003/\text{M} \approx \$0.0011$，而非冷启动的 $\$0.057$）！
   - 若冷启动恢复会话尚无 `lastWireRequest`，自动降级为 `$0.0000`、`<5ms` 的纯确定性结构骨架目录，绝不误付冷缓存全价。
3. **多纪元无损目录防膨胀压缩（`buildMultiEpochBlockDirectory`）**：
   - 最近 40 个块展示完整描述、文件符号行跨度与章节头；更早纪元的块按每 25 块自动聚合为一行纪元范围索引（如 `Earlier Epoch Range #b_001..#b_025`），使连续换代 10–20 次（`5M–10M+` tokens）后检查点体积仍恒定控制在 `<3.5K` tokens，同时 `#b_001..#b_300` 每一个块都能通过 `context_query` 精准寻址。

---

## 3. 独立 `ContextHarness` 架构（`src/harness/` + `src/sidecar/`）

在不依赖 Pi 宿主的自有 Agent 场景（`src/cli.ts --mode harness` 或编程嵌入 `ContextHarness` 类）中，CPAT 提供完整的多分辨率上下文编译与主从异构套利管线：

1. **四档多分辨率块（Multi-Resolution `ContextBlock`）**：
   - `full`（完整内联）$\leftrightarrow$ `digest`（异步廉价旁路模型 `ShadowDigester` 预提取的结构化 `FactDigest`）$\leftrightarrow$ `skeleton`（`<2ms` 确定性结构骨架）$\leftrightarrow$ `landmark`（单行地标）。切换分辨率仅翻转 `active_resolution` 指针，零拷贝、不改 `kind`、不破坏 `tool_call_id` 配对。
2. **L0–L4 严格前缀对齐视图编译器（`src/runtime/view.ts`）**：
   - 移除传统动态插入头部的易变清单，确保非压缩回合消息字节流 100% 严格只增不改。
3. **白盒两级 `DecisionEngine` + `EconomicCommitGate`（`src/sidecar/decision.ts`, `src/harness/cost.ts`）**：
   - 先经 `<1ms` 水位迟滞门控与缓存经济净收益公式验算（拦截回溯尾税大于预期收益的负优化折叠），再按需执行批量降级或 `AC3` 两阶段隔离污染清洗。

---

## 4. 底层事务式 Patch 引擎与 27 条正确性护栏（`src/runtime/patch.ts`）

无论是 v1 模型前台自调 `context_update` 工具，还是 v2 `ContextHarness` 内部提交状态变更，底层统一经由 `applyContextUpdate` 克隆副本事务引擎执行：**任一操作违反护栏则整批原子回滚**。

### 4.1 单份内容存储（`ContentStore`）与审计日志（`Journal`）
- **`ContentStore`**：每个载荷在摄入时按 `<blockId>@v<version>` 落盘一次；`payload_offload` 与 `restore` 均为零拷贝引用翻转。
- **`Journal`**：append-only 记录每次 `ingest` / `patch` / `scope` / `sidecar` 事件与内容键，支持完整离线重放与成本归因。

### 4.2 核心协议与语义护栏

| 护栏类别 | 规则代号 | 守护的不变量 |
| :--- | :--- | :--- |
| **API 协议完整性** | `chain_atomicity` | `assistant(tool_calls)` 与其全部配对 `tool_result` 必须同见同隐，彻底杜绝孤立 `role: "tool"` 引发的 OpenAI/DeepSeek HTTP 400 |
| **系统与权威状态保护** | `protected_kind` · `protected_state` | `system_prompt`、`budget_report`、`task_state` 及 `api_required` 思维链在开启期间禁止被篡改或隐藏 |
| **防语义漂移护栏** | `protected_current_question` | 禁止 `compact` / `fold` / `merge` 吞并最后一条当前用户提问（消除压缩后“忘记当前在答什么”的退化） |
| **结构与可恢复性校验** | `compact_policy` · `fold_range` · `artifact_missing` | 强制 `compact` 显式声明 `preserve`/`drop`，强制 `fold` 仅作用于连续块区间，校验恢复键存在性 |
