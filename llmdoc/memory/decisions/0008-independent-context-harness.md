# 0008：从前台单工具 `context_update` 演进为独立 `ContextHarness` 基础设施（廉价模型 Sidecar + 白盒 Decision）

- **状态**：已采纳并经三套专用数据集四对照臂实证（35 项离线单测全过）
- **日期**：2026-11-16
- **缘起**：阶段 5/6 暴露出 CPAT v1 的两大遗留瓶颈——① 小窗口（32K）下主模型每次自调 `context_update` 产生额外前台往返税，总成本反超被动 `threshold`；② 单向 `set_visibility=hidden` 与 6,000 字符截断的 `artifact_get` 在一次性不可重读流（`ephemeral-stream`）上导致细粒度字面证据永久丢失。结合 2026 年前沿工作（Scroll、Spine、ContextPilot、MEMENTO、acp-kernel、Self-GC、Rivet、AC3），将上下文治理独立为可插拔基础设施。

## 核心架构决策

1. **接入形态（极简协同兼容零侵入）**：
   - 将主模型从繁琐的块级 `context_update` 和每轮追加的 `<context_manifest>` 中解放出来（消除每轮末尾伪 `user` 消息对 Prefix Cache 连续性的破坏，使常规回合实现 **100% 纯字节前缀扩展**）。
   - 主模型侧仅按需暴露 `context_query`（精准检索/切片）及可选的 `scope_open` / `scope_submit`（`Spine` 式子任务栈边界）。

2. **Context 定义（`ScopeTree` + 四档多分辨率块 + 认识论分级）**：
   - 每个块按认识论权限分为 `authoritative` / `observation` / `subjective`，挂载 `full` / `skeleton`（`<2ms` 确定性结构与高信息量字面提取）/ `digest`（旁路廉价模型结构化 `FactDigest`）/ `landmark` 四档投影槽。
   - 修复 `cef6582` `layoutGenerational` 新块倒序插队击穿存量尾部缓存的缺陷，改为严格到达顺序追加 + 尾端齐平批次降级 + `ScopeTree` 栈式回退复用父级热缓存。

3. **行为与控制面（异步廉价模型预消化 + 两级白盒 `DecisionEngine`，不设冗余训练接口）**：
   - `ShadowDigester` 在大 `tool_result` 入库瞬间异步调用廉价模型（`deepseek-flash`，`reasoning_effort: "low"`）后台预计算 `FactDigest`，异常/超时无感回退 `skeleton`。
   - `DecisionEngine` 采用 Tier-1 快速算法门控 + `EconomicCommitGate` 缓存净收益核算 + Tier-2 廉价模型战术判别（含 `SelfCompact` 推导中途抑制与 `AC3` 物理隔离 Assistant 文本的污染清洗审计）。

4. **无损恢复通道（`SearchIndex` + `SubReader`）**：
   - 零外部依赖纯 TS 实现行级切片、正则 `grep` 与窗口化 BM25 倒排索引；即使块已降级为 `digest`/`landmark` 或随已闭合 Scope 归档，任意深埋单句/追踪码均可单轮 `<5ms` 精准召回。

5. **真实经济学成本账本（`CostTracker`）**：
   - 统一核算主模型缓存命中（`inputHit`）、缓存未命中（`inputMiss`）、输出 Token 与旁路廉价模型全量调用的真实折算美元成本（`effective_cost_usd`）及同步关键路径耗时（`harness_critical_path_ms`）。

## 实证结果摘要（`deepseek-flash` 四臂全量横评）

在 `bounded-research`（32K 窗口）、`ephemeral-stream`（一次性不可重读流）、`pollution-spec`（诱饵日志 + 中途需求覆写）三套专用数据集共 12 道测试题上：
- **总准确率**：`react` **16.7%**（全部爆窗提前终止） / `threshold` **58.3%** / `cpat` v1 **75.0%** / **`harness` v2 100.0%（12/12 唯一满分）**。
- **32K 小窗口瓶颈攻克（`bounded-research`）**：`harness` 仅用 **13 轮 / 97.5K prompt tokens / 81.6% 缓存命中率 / $0.0243** 完成 100% 准确率，相比 `cpat` v1（35 轮 / 359.3K tokens / 57.7% 命中 / $0.0677）**总有效成本下降 64.1%，主模型轮次减少 62.9%**。
- **不可重读流精确召回（`ephemeral-stream`）**：`threshold`（25%）与 `cpat` v1（25%，写时过早隐藏导致数据丢失）双双折戟于深埋 `TRC-...` 追踪码回访；`harness` 依靠无损倒排与 `context_query` 实现 **100%（4/4）字面精确召回**，耗时仅为 `cpat` v1 的 **1/3.2**（40.9s vs 129.9s）。
