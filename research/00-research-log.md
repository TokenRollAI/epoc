# 研究日志（按时间顺序）

> 这是 CPAT 研究的主叙事线。每个条目记录：**做了什么、为什么、结果、学到什么、下一步**。
> 诚实记录失败与方向修正——失败路径本身是研究产出。

---

## 阶段 0：CPAT 是什么（出发点）

**命题**：上下文管理从"被动阈值压缩"变成 agent 的**主动决策**——runtime 只发预算压力信号，
选择压缩什么由 agent 通过受校验的 `context_update` 工具决定。

**核心机制**（已实现并通过 18 个离线测试）：
- 8 个原子 op：`compact` / `payload_offload` / `set_visibility` / `restore` / `merge` / `fold`（默认启用）+ `replace` / `redact`（gated）。
- 单份存储（ContentStore）+ append-only Journal；`payload_offload` 是零拷贝视图翻转，`restore` 是其逆操作。
- 压力阶梯：70% soft / 80% must_act / 95% critical（runtime 兜底）。

**最初的（错误的）价值主张**：H1 = "agent 主动治理 > 被动阈值压缩，任务做得更好"。

---

## 阶段 1：五组实验，CPAT 未能证明质量优于 ReAct（2026-06-13）

所有实验用 **DeepSeek V4 Pro（100 万真实窗口）**。

| # | 实验 | 关键结果 |
|---|---|---|
| 1 | 通读整个仓库审计 | react prompt 累积 ~6.4 万，cpat 压到 ~3.3 万（机制有效，省 token） |
| 2 | LongBench narrativeqa 单文档 QA（2 万/15 万预算 ×5） | F1 方向随预算翻转、全在噪声内；cpat prompt **反而更高** |
| 3 | 多文档检索（1 gold+4 干扰 ~14.6 万 ×5） | F1 仍噪声；聪明 ReAct 用 grep 选择性读取、反而更省 |
| 4 | longloop 连环多问（25 万语料 5.1× 预算 ×6 问） | **restore=0、offload=1**；cpat 省 35% token 但 F1 没赢（3.44 vs 4.01）；**ReAct append-only cache 反而更高（74% vs 62%）** |

**结论（当时）**：在所有可程序化判分、不需 Docker 的任务上，CPAT 都**未证明任务质量优于 ReAct**，
只稳定省 token（35-57%）。一度据此把 CPAT **重定位为成本优化策略**（llmdoc decision 0006）。

**沉淀**：llmdoc decisions 0004（预算需低于稳态）、0005（价值需超窗口规模）、0006（重定位）、
reflection `2026-06-13-cpat-vs-react-inconclusive`。

---

## 阶段 2：方向修正——诊断出实验范式根本错误（2026-06-14）

**用户的关键纠正**（这是研究的转折点）：

1. CPAT 是为**超长程 agent**设计的，应放在一个**有限窗口**（如 200K）里。当 ReAct 因超阈值
   无法继续工作时，CPAT 让 agent **自主选择**怎么治理。**价值不在百万窗口省 token，而在受限窗口下"ReAct 做不到、CPAT 能做到"。**
2. 应**自建数据集**——让 agent 从头到尾深度研究某个东西，而非用现成 benchmark。
3. 实验**重做**，参考 CAT 论文（arXiv:2512.22087）控制长度的方式。

**为什么之前测不出（根因诊断）**：
> 我一直用 V4 Pro 的**百万真实窗口**，所以 ReAct **永远不会因窗口耗尽而崩**。
> 而 CPAT 的全部价值恰恰是"在受限窗口下还能继续"。把窗口开到百万，等于提前拿走了 ReAct 的死穴，
> CPAT 自然显不出优势。**这是范式错误，不是 CPAT 没用。**

**论文佐证（arXiv:2512.22087 "Context as a Tool"，与 CPAT 同构）**：
- 窗口**硬卡在 32K-65K**；推理最多 **500 轮**。
- ReAct 关键失效："**一旦上下文窗口耗尽，对话提前终止**"（append-only 填满后无法纳入新信息）。
- 三对照臂：**ReAct 49.8 / 阈值压缩 53.8 / CAT 57.6**（SWE-Bench Verified Pass@1）。
- 长程退化：ReAct"**60 轮后饱和退化**"，而 CAT 持续上升；500 步时 ReAct 48.8 vs CAT 57.8。
- 用 **step 数 × 窗口上限** 控制长度（Table 3），而非语料大小。
- 平均每条轨迹 **4.22 次 context-mgmt 动作**，压缩比约 **30%**（15585→4676 token）。

详见 [`02-related-work.md`](02-related-work.md)。

---

## 阶段 3：重做实验（进行中，2026-06-14 起）

**新假设**（可证伪，详见 [`01-problem-and-hypotheses.md`](01-problem-and-hypotheses.md)）：
> **H-bounded**：在受限窗口（32K / 200K）下的长程研究任务中，ReAct 会因上下文耗尽而提前终止或
> 准确率退化；阈值压缩居中；CPAT 主动可逆治理维持最高任务完成度。

**设计**（详见 `design/`）：
- **硬窗口机制**：渲染 context 超过固定预算时，ReAct 臂模拟"窗口耗尽→被迫提前作答/终止"。
- **三对照臂**：ReAct（会崩）/ threshold-compression（被动）/ CPAT（主动）——对齐论文。
- **双扫窗口**：32K（对齐论文）+ 200K（贴近 V4 Pro 部署）。
- **自建深度研究任务**：多轮、强制累积超窗口、含回访早期内容的问题、可程序化判分。
- **指标**：任务能否完成 / 是否提前终止 / 准确率 / token 吞吐 / 治理动作数 / restore 命中。

**当前进度**：实现硬窗口 + 阈值压缩臂中（task #7）。

> 下一条目在实现与首跑后追加。

---

## 阶段 4：实现三臂 + 自建数据集 + 关键突破（2026-06-25）

### 实现
- **硬窗口机制**（`loop.ts`）：`hardWindowTokens` + `terminated_early`/`peak_view_tokens` 指标。
  react 臂超窗口 → 强制作答（复现论文"提前终止"）；threshold/cpat 不终止。
- **第三臂 threshold**：有 runtime 安全网（checkBudget 自动 offload）但**无 context_update 工具**——被动压缩。
- **`taskToolNames` 过滤**：可只给 list_dir+read_file（去 grep，强制整篇读）。
- **自建数据集** `bench/deepresearch.ts`：合成 dossier 语料、精确匹配答案、聚合/跨文档/回访问题。
- 测试 25→26 全过（含硬窗口终止、threshold 无 context_update、语义漂移护栏）。

### smoke 序列（每次都在拦截设计缺陷，这是先跑 smoke 的价值）
| smoke | 配置 | 结果 | 学到 |
|---|---|---|---|
| v1 (32K, 可grep) | 8 docs, 有 grep | 三臂全 100%, peak<10K | **任务可 grep 跳过 → 不累积 → 治理无价值**。第 6 次踩坑。 |
| v2 (32K, 无grep+聚合) | 20 docs ~81K, 聚合问题 | **react 0%(提前终止) / threshold 100% / cpat 100%** | **突破**：受限窗口第一次咬住 ReAct。"治理>不治理"成立。 |
| v3 (32K, 埋深事实回访) | 20 docs ~98K | react 10% / **threshold 100% / cpat 90%(贵3倍)** | **cpat 输 threshold**：主动 compact 引发**语义漂移**（把当前问题压没了，agent"忘了在答什么"）。 |
| v3b (32K, +护栏) | 同 v3 + 语义漂移护栏 | （进行中） | 测护栏能否让 cpat 追平/超过 threshold。 |

### 已确立的核心结论
1. **H-bounded 成立**：受限窗口下，**任何治理（threshold 或 cpat）都让 ReAct 从 0-10% → 90-100%**。
   这是"CPAT 有价值"的第一个无歧义证据——ReAct 在受限窗口下会因上下文耗尽提前终止而彻底失败。
2. **prompt-only 主动 CPAT 暂不优于被动 threshold**（v3）：主动 compact 若不精准会引发语义漂移、更贵、更易错。
   **这恰好复现 CAT 论文动机**：CAT 赢是因为**经过 SFT 训练**学会压什么；prompt-only 不够。
3. **restore 始终为 0**：只要信息源（文件）可重读，agent 永远选 re-read 而非 restore。
   restore 的价值场景是"信息源不可重得"，当前静态文件任务天然让它无用武之地。

### 修复（task 进行中）
针对 v3 诊断出的语义漂移，给 patch 引擎加**硬护栏**：compact/fold/merge 不能吞掉
`task_state` 和**当前问题（最近 user_message）**（新规则 `protected_state` / `protected_current_question`）。
v3b 重跑 cpat 检验护栏效果。

详见 `experiments/03-smoke-v1/v2/v3*.md` 与 `design/03`、`design/04`。


---

## 阶段 5：提示工程重写 + 架构瓶颈发现（2026-06-25）

**用户洞察**：CPAT 的三层提示（tool description / system prompt / harness）没做好，
尤其"下一轮压缩后老的那部分应删掉"的策略缺失。

**做了什么**：全面重写三层提示（详见 `design/05`）——里程碑式"读完即 offload"、
处置陈旧块（摘要已含事实就 hidden 掉冗余源）、preserve 精确事实启发、护栏入提示。

**v3c 结果**（新提示，32K，同 v3 配置）：
- ✅ compact 9→0（成功导向安全 offload）、accuracy 稳 100%、peak 仅 20K（里程碑策略生效）。
- ❌ **total token 445K，比 v3b(319K) 更高，远超 threshold(142K)**：新提示让 agent 更勤治理，
  但每次 context_update 是一次额外 LLM 往返，开销 > 节省。

**决定性发现（比"提示没调好"更深）**：
> **prompt-only 主动 CPAT 的瓶颈不在提示，在架构**——"治理作为独立 tool call"每次都有
> LLM 往返开销，prompt 再好也省不掉。这印证 CAT 论文为何用 SFT 训练（让治理零往返融入推理）。

**三维最终判定**：
| 维度 | 判定 |
|---|---|
| 治理 > 不治理（受限窗口） | ✅ 成立（react 10% → 治理 100%） |
| 主动 > 被动（质量） | ⚠️ 打平（加护栏后都 100%） |
| 主动 > 被动（效率） | ❌ 不成立（cpat 445K vs threshold 142K） |
| restore 独立价值 | ❓ 未触发（文件可重读） |

**仍开放**：restore 的价值需"信息源不可重得"任务才能验证（唯一还没试、理论上 CPAT 该赢的场景）。

---

## 阶段 6：全量双扫（进行中，2026-06-25）

把已成立的"治理 > 不治理"做成统计稳健的完整证据：32K + 200K 两窗口 × 三臂
（react/threshold/cpat）× 大语料（32K 用 ~100K 语料，200K 用 ~480K 语料）。
结果见 `experiments/04-full-double-sweep.*`。

### 阶段 6 结果：双扫完成，CPAT 价值被强证明（2026-06-25）

**完整对照见 `experiments/04-full-double-sweep.md`。两个决定性结论：**

1. **治理 > 不治理（铁证，两窗口）**：react 在 32K/200K 都提前终止、0%/8.3%；
   治理臂都 100%。受限窗口下 ReAct 因上下文耗尽彻底失败。

2. **突破：200K 窗口下主动 CPAT 在效率上碾压被动 threshold**：
   同 100% 准确率，**cpat 488K token vs threshold 2,819K token —— 省 83%（5.8×）**。
   机制：threshold 被动等 context 涨满才压（每轮背近满窗口），CPAT 主动读完即 offload
   （context 始终压在 25K）。**窗口越大，CPAT 优势越明显**——正是它为"有限但较大窗口"设计的原因。

**这修正了 decision 0006**："CPAT 仅成本优化、非质量提升"在受限窗口下是错的：
- 相对 ReAct：质量决定性提升（0%→100%，能否完成）。
- 相对 threshold：大窗口下巨大成本优化（省 83%），质量打平。

**研究目标达成**：跑出了对 CPAT 有实际价值的 case（强价值，非边际）。

---

## 阶段 7：演进为独立 `ContextHarness` 基础设施（廉价模型 Sidecar + 白盒 Decision，2026-11-16）

### 动机：攻克阶段 5/6 遗留的两大公开局限
1. **小窗口往返税倒挂**：阶段 6 中 32K 小窗口下 `cpat` v1 因主模型频繁自调 `context_update`，总耗费反比被动 `threshold` 高 52%。
2. **写时主观丢弃与不可重读信息源盲区**：静态文件可重读时 `restore=0`；一旦换成一次性不可重读工具流，写时主动 `hidden` 处置陈旧块会误删未来追问所需的细粒度字面事实。

### 架构升级（`src/harness/` + `src/sidecar/` + `src/runtime/tree.ts,index.ts`）
结合 2026 年前沿范式（Scroll、Spine、ContextPilot、MEMENTO、acp-kernel、Self-GC、Rivet、AC3），将上下文管理从“寄生于主模型推理回合的单工具”解耦为独立 **`ContextHarness`**：
- **定义 Context**：L0–L4 严格前缀对齐编译视图 + `ScopeTree` 任务栈帧 + 四档多分辨率块（`full` / `skeleton` / `digest` / `landmark`）+ 认识论权限分级（`authoritative` / `observation` / `subjective`），移除每轮 `<context_manifest>` 伪用户消息以保持 100% 字节前缀连续性。
- **定义 行为**：写入时同步建立零依赖 `SearchIndex`（行级 + Regex + 窗口 BM25）并异步触发廉价模型 `ShadowDigester`（`deepseek-flash`, `reasoning_effort="low"`）；回合前由两级白盒 `DecisionEngine`（Tier-1 快速门控 + `EconomicCommitGate` 缓存净收益核算 + Tier-2 战术判别 + `AC3` 物理隔离污染清洗）零往返税完成批量降级；读取时由 `SubReader` 响应 `context_query` 做精准行级投影。
- **定义 效果与成本**：五维效果指标 + `CostTracker` 真实缓存折后美元账本（区分主模型 `hit`/`miss`/`out` 与旁路 `sidecar` 全量开销）。

### 决定性实证结果（详见 [`experiments/05-context-harness-eval.md`](experiments/05-context-harness-eval.md)）
1. **三套专用数据集（`bounded-research` 32K / `ephemeral-stream` 不可重读流 / `pollution-spec` 污染覆写，共 12 题）同构 `deepseek-flash` 四臂全量横评**：
   - 总准确率：`react` **16.7%**（全爆窗） / `threshold` **58.3%** / `cpat` v1 **75.0%** / **`harness` v2 100.0%（12/12 唯一满分）**。
   - 在一次性不可重读流 `ephemeral-stream` 上，`threshold`（无恢复通道）与 `cpat` v1（写时认为追踪表是噪声将其 `hidden`）在深埋 `TRC-...` 针式召回上双双 **0%**（总分 25%）；`harness` 达成 **100%（4/4）** 且墙钟速度比 `cpat` v1 快 **3.2×**。
   - 在 32K `bounded-research` 上，`harness` 缓存命中率达 **81.6%**（四臂最高），总成本比 `cpat` v1 **直降 64.1%**。
2. **主从异构模型套利验证（Main `deepseek-v4-pro` + Sidecar `deepseek-flash`）**：
   - 在 15K 极小窗口 `bounded-research` 上，三臂均 100% 准确率，但 **`harness` 仅用理论极限 4 轮、真实总成本 $0.0195**，不仅比 `cpat` v1（$0.1332）**暴省 85.4%（便宜 6.8×）**，更首次在小窗口正面击败被动 `threshold`（$0.0287，**省 32.1%**），彻底闭合阶段 6 遗留缺口。

---

## 阶段 8：v2.1 迟滞批量消化打磨 + 三大公开 Benchmark 验证（2026-11-16）

针对阶段 7 暴露的摩擦点落地三项升级（按需多块打包 `digestBatch`、默认 `query_only` 极简工具面、**GC 高低水位迟滞门控**与停用词感知检索），并接入三大第三方公开基准（[`bench/public-bench.ts`](../bench/public-bench.ts)，详见 [`experiments/06-public-benchmarks-v2-1.md`](experiments/06-public-benchmarks-v2-1.md)）：

- **`bounded-research` (32K) 复测**：即使面对主模型单轮并发读取全部 14 篇档案（~70K tokens）的极端冲击，v2.1 零阻塞完成降级，主模型轮次 `13 → 11`、Prompt Tokens **`97.5K → 69.5K`（再降 28.8%）**、缓存命中率 **`82.4%`**、准确率 **100%**。
- **`LongMemEval-S`（真实 50+ 轮 ~110K 会话流 / 32K 窗口）**：`react` 100% 爆窗（50%），`cpat` v1 因治理挤占步数未读完会话（50%，$0.1534）；`harness` v2.1 拿到 **100%**，GC 迟滞门控将缓存命中率从 `30.6%` 推升至 **`58.6%`**，总成本降至 **$0.1262（比 v1 省 18%、快 1.6×）**。
- **`MemoryAgentBench`（单向增量块摄取 / 26K 窗口）**：`react` 爆窗仅 **12.5%**；`harness` v2.1 以最低峰值水位（`14.8K`）和 `62.2%` 缓存命中率，比 `cpat` v1 **省 62% 主模型 Prompt Tokens、省 39% 总成本、快 2.2×**。
- **`ContextBench Verified`（真实检出 `astropy` / `django` 历史 Commit / 24K 窗口）**：`react` 在 `astropy` 上爆窗；被动 `threshold` 虽找对文件但丢掉行号导致 **`Utilized Line F1` 崩塌至 `12.9%`**；`harness` v2.1 全程零爆窗、保持 **`78.8%` 最高缓存命中率**，行级定位 `Line F1` 达 **`41.2%`（`threshold` 的 3.2 倍、`cpat` v1 的 1.4 倍）**。

---

## 阶段 9：接入真实 `Pi CLI` 宿主、对标 `billion-context` 与 400K `Cache-First Lazy` 架构跃迁（2026-11-17）

将 CPAT 封装为原生 Pi 插件（[`extensions/pi-cpat.ts`](../extensions/pi-cpat.ts)，`pi -e ./extensions/pi-cpat.ts`），在真实 `Pi CLI v1.5.2` + `deepseek-flash` **400K 窗口（50:1 缓存价差：`$0.15/M` vs `$0.003/M`）** 下正面对标官方 `billion-context v0.1.216` 与 `pi-native`（[`bench/pi-billion-bench.ts`](../bench/pi-billion-bench.ts)，详见 [`experiments/07-pi-400k-cache-first-vs-billion-context.md`](experiments/07-pi-400k-cache-first-vs-billion-context.md)）：

- **第一性原理反转发现**：在 `50:1` 缓存折扣下，高频中段折叠（如 `billion-context` 50K 步频与 CPAT 48K 滑动步频）因反复使切点后尾巴缓存失效，总成本反比不改前缀的 `pi-native` 贵 30%–80%；而 `pi-native` 的唯一短板是跨过 `384K` 撞线时将工具输出硬截断为 2,000 字符（`LongMemEval-S` 准确率跌至 `50%`）。
- **`Cache-First Lazy` 合体（默认策略）**：`0 → 384K` 零字节改动纯追加锁死前缀缓存（`$0` 旁路费，仅在内存 `<1ms` 同步建 `SearchIndex` 与代码 `Lstart-Lend` 符号表），仅在撞线瞬间以 `$0` 确定性结构目录 + 无损 `context_query` 接管 `session_before_compact`：
  - **`LongMemEval-S` (~510K)**：**`100%` Judge Acc**（`pi-native` 50%），Cache 命中率 **`77.9%`（全场最高，较 48K 步频 +60.8pp）**，成本 **`$0.0719`（紧贴 `pi-native` 地板价 `$0.0699`，较 `billion-context` `$0.1276` 省 43.7%、快 1.76×）**。
  - **`MemoryAgentBench AR` (~502K)**：**`100%` EM**（`billion-context` 83.3%），Cache 命中率 **`82.6%`（全场最高）**，成本 **`$0.0621`（较 `billion-context` `$0.1084` 省 42.7%、快 2.19×）**。
  - **`ContextBench`（连续三仓库单会话）**：**`33.4%` Utilized Line F1（全场第 1，`billion-context` 21.8%、`pi-native` 15.8%）**，Cache 命中率 **`95.4%`**。

---

## 阶段 10：三层 `Cache-First v3` 完整形态落地——入窗前门控 + 同源派生摘要 + 显式污染边界折叠（2026-11-17）

针对阶段 9 纯懒压缩剩余的三个真实工程痛点，在 `extensions/pi-cpat.ts` 与 `src/sidecar/tool-projector.ts` 完成三层零回溯偏移闭环（42/42 离线测试全绿，详见 [`experiments/08-cache-first-v3-upgrades-and-400k-benchmarks.md`](experiments/08-cache-first-v3-upgrades-and-400k-benchmarks.md)）：

1. **Layer 1 — `tool_result` Pre-Cache 入窗前门控（击穿首轮未命中税地板）**：
   - 在超大/低熵工具输出进入主模型与 KV Cache **之前**即转存 `SearchIndex` 并返回高信号骨架，同时从 `details.fullOutputPath` 与源文件自动恢复宿主 50KB / 2,000 行截断盲区。
   - 实测将 `LongMemEval-S` (~600K) 缓存命中率推至 **`90.9%`（全场第 1）**、总账单压至 **`$0.0087`（较 `pi-native` `$0.0695` 便宜 8.0×，较 `billion-context` `$0.1506` 便宜 17.3×）**；`MemoryAgentBench AR` (~502K) 账单压至 **`$0.0067`（便宜 9.0×–16.8×）**；`oversized-tools` 两轮 100% 满分且成本全场最低（`$0.0021` / `$0.0099`，而 `billion-context` 在第二轮跌至 `0%`）。
2. **Layer 2 — 显式任务/污染边界 `ScopeTree` + `AC3` 零后缀栈帧折叠**：
   - 支持 `context_checkpoint` 工具、`/cpat-fold` 命令与用户显式纠偏/切新任务识别，在 `suffixCachedTokens == 0` 处清洗错误推理定势。
   - 在 `pollution-boundary`（Phase 1 注入 ~54K 废弃 v1 规范 → Phase 2 切 v3 合约）中，Phase 2 活跃窗口直降 **`82.3%–88.3%`**（`53.8K → 3.1K` 起步、峰值仅 `9.5K`），总 Prompt 减半至三分之一，无污染准确率 **100%**。
3. **Layer 3 — `384K` 同源前缀派生摘要（Cache-Forked Epoch Summary）+ 120 题全子任务（480 题次）终局验证**：
   - 通过 `before_provider_headers` + `before_provider_request` 完整透传 `rawPayload`（含 `thinking` 模板参数）与 `x-session-affinity` 亲和头，在 `LongMemEval-S` 全部 10 组跨 `384K` 马拉松上实测拿到 **`99.8%` 派生摘要 KV Cache 命中率（10 次 ~360K 摘要合计仅花 `$0.0173`）**。
   - **120 题四臂终局成绩单**（`LongMemEval-S` 40 题 + `MemoryAgentBench` 40 题含 `1.48M TTL` + `ContextBench` 9 仓 30 题 + 专项 10 题）：**`cpat` 以 `67.9%` 宏平均 / `63.2%` 加权总分夺得效果全场第 1**（`billion-context` `65.0%` / `59.1%`，`pi-native` `58.5%` / `50.4%`，`full` `40.1%` / `26.5%`），同时 **120 题全链路总账单仅 `$1.3250`（在全部存活臂中成本最低：比 `billion-context` `$2.5285` 省 `47.6%`，甚至比不做任何治理的 `pi-native` `$1.4760` 还省 `10.2%`！）**。

---

## 阶段 11：正式定名 `epoc`（Epoch Context）与六宿主统一工程化蓝图（2026-11-17）

随着第 1–10 阶段实证彻底否定“中途给上下文打补丁（Context Patch as Tool）”路线，项目正式由研究代号 `CPAT` 更名为 **`epoc`（Epoch Context / Epoch-Paged Optimal Context）**，远程 GitHub 仓库迁移为 `TokenRollAI/epoc`，并在 [`../ROADMAP.md`](../ROADMAP.md) 完成面向 **`codex`（含 ChatGPT 订阅 OAuth）· `claude code` · `agy` · `kimi` · `pi` · `dsh`** 六大宿主的统一 CLI、确定性影子账本（`ShadowLedger`）本地多协议网关与轻量 `epoc-mcp` 工程化蓝图。
