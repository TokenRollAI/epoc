# 0009：Cache-First v3 400K 全量实证、项目更名为 `epoc`（Epoch Context）及六宿主工程化蓝图

- **状态**：v3 架构与 120 题评测已完成；`epoc` 更名已采纳；六宿主接入蓝图已写入 `ROADMAP.md`
- **日期**：2026-11-17
- **关联文档**：`ROADMAP.md`、`research/experiments/08-cache-first-v3-upgrades-and-400k-benchmarks.md`

## 1. 背景与实证结论（Cache-First v3）

在 `50:1` 高折扣前缀缓存定价下（未命中 `$0.15/M` vs 缓存命中 `$0.003/M`），我们将上下文基础设施收敛为遵守**零后缀偏移定律（Zero-Suffix Shift Law，$T_{\text{suffix}} \equiv 0$）**的三层架构（`extensions/pi-cpat.ts` + `src/sidecar/tool-projector.ts`）：

1. **Layer 1 · 入窗前超大 `tool_result` 门控与截断恢复**：代码/Shell `>24K`、高冗余 `>28K`、绝对上限 `>96K` 或宿主截断（`>=3.5K`）在进入 KV Cache 前直接投影为 `~800–1,400` tok 骨架，全量原文落入无损 `SearchIndex`。
2. **Layer 2 · 显式任务/污染边界零后缀折叠（`ScopeTree` + `AC3`）**：仅在子任务完结（`context_checkpoint`）或用户显式纠偏/切新任务边界折叠历史，物理隔离旧 Assistant 幻觉定势。
3. **Layer 3 · `384K` 高水位同源前缀派生摘要（Cache-Forked Epoch Summary）**：日常保持纯追加缓存；撞线瞬间复用温热请求前缀（含 `thinking`、`reasoning_effort`、`x-session-affinity`）派生跨纪元因果摘要，实测达成 **`99.8%` 线上 KV Cache 命中率**（10 次 ~360K 摘要总计仅 `$0.0173`）。

在 `Pi CLI v1.5.2` + `deepseek-flash` 400K 窗口 **120 题 × 四对照臂（480 题次 / 152 会话单元）** 全量评测中：
- **综合得分**：**`67.9%` 宏平均（全场 #1）** / **`63.2%` 加权平均（全场 #1）**（`LongMemEval-S` `82.5%` #1、`MemoryAgentBench` `61.9%` #1、`Oversized+Pollution` `100%` #1）。
- **真实总账单**：**`$1.3250`（存活臂最低 #1）**——比 `billion-context v0.1.216`（`65.0%` / `$2.5285`）**节省 `47.6%`**，比 `pi-native`（`58.5%` / `$1.4760`）还**节省 `10.2%`**。

## 2. 品牌与命名决策：`CPAT` → `epoc`（Epoch Context）

- **弃用 `CPAT`（Context Patch as Tool）原意的原因**：第 1–8 轮实验恰恰证伪了“让模型在对话中途调用工具给历史上下文打补丁（Patch）”的范式——中途打补丁会摧毁切点后的前缀缓存。原名已与最终胜出的“反中途补丁、纪元内单调追加”架构语义相反。
- **定名为 `epoc`（Epoch Context / Epoch-Paged Optimal Context）**：
  - 精准概括 **纪元内前缀单调锁死 + 入窗前门控 + 纪元边界 99.8% 缓存派生换代 + 跨纪元无损检索** 的核心机制；
  - 4 字母自然双音节词，作为日常高频前缀启动器（`epoc codex`、`epoc claude`、`epoc agy`、`epoc kimi`、`epoc pi`、`epoc dsh`）在 macOS/Linux 上与系统命令零冲突，且避免 `.pfx` 证书后缀歧义；
  - 远程 GitHub 仓库同步由 `TokenRollAI/CPAT` 重命名为 `TokenRollAI/epoc`。

## 3. 六宿主工程化包装关键决策

为同时覆盖 **`codex`（主力为 ChatGPT 订阅 OAuth）**、**`claude code`**、**`agy`**、**`kimi`**、**`pi`**、**`dsh`**：
1. **确定性影子账本（`ShadowLedger`，基于内置 `node:sqlite`）**：针对除 `pi` 外每轮全量重放本地原始历史的 5 个客户端，网关在尾部新工具输出首次入窗时建立 `rawHash -> Skeleton` 幂等映射，后续轮次 O(1) 逐字节还原同一骨架并保留各协议签名/缓存断点（`cache_control`、`encrypted_content`、`thoughtSignature`），在无状态 HTTP/WS 层维持 100% 前缀缓存命中。
2. **`codex` ChatGPT 订阅列为 Phase 1 一等公民**：不改写 `model_providers`（避免破坏订阅认证与 Desktop Composer 预测），由 `epoc codex` 注入 `--no-daemon` + 仅针对 `chatgpt.com` / `api.openai.com` 的窄域本地 CA + OpenAI Responses SSE/WebSocket 双栈网关，配合 `[mcp_servers.epoc]` 与 `_meta.threadId` 线程绑定。
3. **免证书优先原则**：凡宿主支持 `BASE_URL` 或进程内扩展的通道（`pi` 进程内扩展、`dsh` Cordis bundle、`claude code` 受管 `ANTHROPIC_BASE_URL`、`kimi` 受管 `config.toml`、`agy` `CLOUD_CODE_URL` + 原生插件），一律走本地明文回环或进程内接缝，不引入多余 TLS 拦截复杂度。
