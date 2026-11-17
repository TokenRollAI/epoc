import { mkdirSync } from "node:fs";
import type { ChatClient } from "../deepseek/client.ts";
import { BlockStore, describeText } from "../runtime/blocks.ts";
import { SearchIndex } from "../runtime/index.ts";
import { applyContextUpdate } from "../runtime/patch.ts";
import { ContentStore, Journal } from "../runtime/stores.ts";
import { ScopeTree } from "../runtime/tree.ts";
import { buildHarnessMessages } from "../runtime/view.ts";
import { DecisionEngine } from "../sidecar/decision.ts";
import { ShadowDigester } from "../sidecar/digester.ts";
import { SubReader, type ContextQueryArgs } from "../sidecar/reader.ts";
import { projectOversizedToolResult } from "../sidecar/tool-projector.ts";
import type {
  ChatMessage,
  ChatUsage,
  ContextBlock,
  ContextOperation,
  ContextView,
  CostBreakdown,
  HarnessConfig,
  HarnessMode,
  TokenBudget,
  ToolCall,
  ToolDefinition,
} from "../types.ts";
import { CostTracker, pricingForModel } from "./cost.ts";
import {
  augmentSystemPromptForHarness,
  harnessToolDefinitions,
} from "./tools.ts";

interface PendingScopeClosure {
  scopeId: string;
  summary: string;
  keyFacts: string[];
}

/**
 * Independent, pluggable Context Harness Infrastructure (CPAT v2).
 *
 * Separates context governance from the main agent's foreground reasoning turns:
 *   1. Data Plane: Single-copy ContentStore + Append-only Journal + SearchIndex
 *      + ScopeTree + Multi-Resolution ContextBlocks + L0..L4 strict prefix view.
 *   2. Control Plane: Async ShadowDigester + 2-Tier White-Box DecisionEngine
 *      (with EconomicCommitGate & AC3 isolated pollution audit) + SubReader.
 *   3. Cost Plane: Exact cached/uncached/output/sidecar USD & latency accounting.
 */
export class ContextHarness {
  readonly config: Required<
    Pick<
      HarnessConfig,
      | "maxContextTokens"
      | "workingTailTokens"
      | "softLimitRatio"
      | "mustActRatio"
      | "criticalRatio"
      | "minDigestTokens"
      | "keepRecentFullChains"
      | "expectedHorizonTurns"
      | "mode"
      | "sidecarModel"
      | "runDir"
      | "verbose"
    >
  >;

  readonly journal: Journal;
  readonly content: ContentStore;
  readonly blocks: BlockStore;
  readonly index: SearchIndex;
  readonly scopes: ScopeTree;
  readonly cost: CostTracker;
  readonly digester: ShadowDigester;
  readonly decision: DecisionEngine;
  readonly reader: SubReader;

  private lastEstimate = 0;
  private lastActualPrompt = 0;
  private lastSentBlockIds = new Set<string>();
  private lastAssistantBlockId?: string;
  private pendingScopeClosures: PendingScopeClosure[] = [];
  private questionBoundaryPending = false;
  private newUserSinceLastCompile = false;
  private userMessageCount = 0;

  contextQueryCalls = 0;
  scopesOpened = 0;
  scopesSubmitted = 0;
  harnessCommits = 0;

  constructor(
    rawConfig: HarnessConfig,
    opts: { mainModel?: string; sidecarClient?: ChatClient } = {},
  ) {
    mkdirSync(rawConfig.runDir, { recursive: true });
    const maxTok = rawConfig.maxContextTokens;
    const sidecarModel = rawConfig.sidecarModel ?? "deepseek-flash";
    const mainModel = opts.mainModel ?? sidecarModel;

    this.config = {
      maxContextTokens: maxTok,
      workingTailTokens:
        rawConfig.workingTailTokens ?? Math.min(22_000, Math.floor(maxTok * 0.72)),
      softLimitRatio: rawConfig.softLimitRatio ?? 0.72,
      mustActRatio: rawConfig.mustActRatio ?? 0.84,
      criticalRatio: rawConfig.criticalRatio ?? 0.93,
      minDigestTokens: rawConfig.minDigestTokens ?? 450,
      keepRecentFullChains: rawConfig.keepRecentFullChains ?? 1,
      expectedHorizonTurns: rawConfig.expectedHorizonTurns ?? 8,
      mode: rawConfig.mode ?? "query_only",
      sidecarModel,
      runDir: rawConfig.runDir,
      verbose: rawConfig.verbose ?? false,
    };

    this.journal = new Journal(this.config.runDir);
    this.content = new ContentStore(this.config.runDir);
    this.blocks = new BlockStore(this.content, this.journal);
    this.index = new SearchIndex();
    this.scopes = new ScopeTree(this.blocks, this.journal);
    this.cost = new CostTracker({
      mainPricing: rawConfig.mainPricing ?? pricingForModel(mainModel),
      sidecarPricing: rawConfig.sidecarPricing ?? pricingForModel(sidecarModel),
    });
    this.digester = new ShadowDigester({
      client: opts.sidecarClient,
      model: sidecarModel,
      blocks: this.blocks,
      journal: this.journal,
      cost: this.cost,
      minTokens: this.config.minDigestTokens,
    });
    this.decision = new DecisionEngine({
      client: opts.sidecarClient,
      model: sidecarModel,
      blocks: this.blocks,
      content: this.content,
      journal: this.journal,
      cost: this.cost,
      digester: this.digester,
    });
    this.reader = new SubReader({
      index: this.index,
      client: opts.sidecarClient,
      model: sidecarModel,
      journal: this.journal,
      cost: this.cost,
    });
  }

  /** Merge task tools with the active HarnessMode's lightweight context tools. */
  tools(taskTools: ToolDefinition[]): ToolDefinition[] {
    return [...taskTools, ...harnessToolDefinitions(this.config.mode)];
  }

  get mode(): HarnessMode {
    return this.config.mode;
  }

  // -- Ingestion -------------------------------------------------------------

  ingestSystem(baseSystemPrompt: string): ContextBlock {
    const augmented = augmentSystemPromptForHarness(
      baseSystemPrompt,
      this.config.mode,
    );
    return this.blocks.createSystem(augmented);
  }

  ingestUser(content: string, opts: { isFollowupBoundary?: boolean } = {}): ContextBlock {
    this.userMessageCount += 1;
    const block = this.blocks.createUserMessage(content);
    this.index.upsert(block.id, block.kind, block.description, content);
    if (this.userMessageCount > 1) {
      this.newUserSinceLastCompile = true;
    }
    if (opts.isFollowupBoundary) {
      this.questionBoundaryPending = true;
    }
    return block;
  }

  ingestAssistant(msg: ChatMessage): ContextBlock {
    this.releaseReasoning();
    const { block } = this.blocks.createAssistantMessage(
      msg,
      this.scopes.activeScopeId(),
    );
    this.lastAssistantBlockId = block.id;
    this.scopes.registerBlock(block.id);
    if (msg.content?.trim()) {
      this.index.upsert(block.id, block.kind, block.description, msg.content);
    }
    return block;
  }

  ingestToolResult(toolCall: ToolCall, output: string): ContextBlock {
    let args = "";
    try {
      const parsed = JSON.parse(toolCall.function.arguments || "{}");
      args = Object.entries(parsed)
        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
        .join(", ")
        .slice(0, 80);
    } catch {
      args = toolCall.function.arguments.slice(0, 80);
    }
    const desc = describeText(`Result of ${toolCall.function.name}(${args})`, output);
    const block = this.blocks.createToolResult(
      toolCall,
      output,
      desc,
      this.scopes.activeScopeId(),
    );
    this.scopes.registerBlock(block.id);
    this.index.upsert(block.id, block.kind, desc, output);

    // Pre-Cache Oversized Tool-Result Gate: compact monster coding/log dumps
    // BEFORE Turn t+1 ever prefills or caches them (zero retroactive cache shift!).
    const softCap = Math.min(24_000, Math.max(8_500, Math.floor(this.config.maxContextTokens * 0.38)));
    const proj = projectOversizedToolResult({
      blockId: block.id,
      toolName: toolCall.function.name,
      description: desc,
      rawText: output,
      codingToolSoftCapTokens: softCap,
      genericRedundantCapTokens: Math.min(28_000, Math.max(12_000, Math.floor(this.config.maxContextTokens * 0.45))),
      hardInlineCapTokens: Math.min(48_000, Math.max(20_000, Math.floor(this.config.maxContextTokens * 0.65))),
    });
    if (proj.shouldProject && proj.projectedText) {
      block.resolutions = { ...block.resolutions, skeleton: proj.projectedText };
      block.active_resolution = "skeleton";
      block.token_count = proj.projectedTokens;
    }

    return block;
  }

  /**
   * Intercepts Harness-owned tool calls (`context_query`, `scope_open`,
   * `scope_submit`). Returns `undefined` if `tc` is an ordinary task tool.
   */
  async tryHandleToolCall(tc: ToolCall): Promise<string | undefined> {
    const name = tc.function.name;
    if (name === "context_query") {
      this.contextQueryCalls += 1;
      const t0 = Date.now();
      const raw = JSON.parse(tc.function.arguments || "{}") as ContextQueryArgs;
      const out = await this.reader.query(raw);
      // SubReader `ask` mode already logs its LLM wait; for pure index hits add sync time
      if ((raw.mode ?? "snippets") !== "ask") {
        this.cost.addCriticalPathMs(Date.now() - t0);
      }
      return out;
    }

    if (name === "scope_open") {
      const raw = JSON.parse(tc.function.arguments || "{}") as {
        label?: string;
        goal?: string;
      };
      const node = this.scopes.openScope({
        label: raw.label ?? "subtask",
        goal: raw.goal ?? "",
        openingBlockId: this.lastAssistantBlockId,
      });
      this.scopesOpened += 1;
      return `Opened scope "${node.id}" (${node.label}). Proceed with focused exploration; call scope_submit when complete.`;
    }

    if (name === "scope_submit") {
      const raw = JSON.parse(tc.function.arguments || "{}") as {
        summary?: string;
        key_facts?: string[];
      };
      const active = this.scopes.activeScope();
      if (!active) {
        return `No active scope to submit (already at root scope). Your summary has been noted inline.`;
      }
      this.pendingScopeClosures.push({
        scopeId: active.id,
        summary: (raw.summary ?? "").trim() || `Completed ${active.label}`,
        keyFacts: Array.isArray(raw.key_facts)
          ? raw.key_facts.map((x) => String(x).trim()).filter(Boolean)
          : [],
      });
      this.scopesSubmitted += 1;
      return `Scope "${active.id}" (${active.label}) marked complete; intermediate traces will collapse before the next turn.`;
    }

    return undefined;
  }

  /**
   * Executes any queued scope closures (now that all tool results of the turn
   * have been ingested), runs the DecisionEngine + EconomicCommitGate, and
   * compiles the strict prefix-preserving message sequence for the Main Model.
   */
  async compileTurn(ephemeralTailMessages?: ChatMessage[]): Promise<{
    messages: ChatMessage[];
    view: ContextView;
    decisionReason: string;
    freedThisTurn: number;
  }> {
    const t0 = Date.now();
    let freedThisTurn = this.flushPendingScopeClosures();

    const estBefore = this.estimatedUsed();
    const dec = await this.decision.evaluateBeforeTurn({
      estUsedTokens: estBefore,
      maxContextTokens: this.config.maxContextTokens,
      workingTailTokens: this.config.workingTailTokens,
      softLimitRatio: this.config.softLimitRatio,
      mustActRatio: this.config.mustActRatio,
      criticalRatio: this.config.criticalRatio,
      minDigestTokens: this.config.minDigestTokens,
      keepRecentFullChains: this.config.keepRecentFullChains,
      expectedHorizonTurns: this.config.expectedHorizonTurns,
      lastSentBlockIds: this.lastSentBlockIds,
      questionBoundary: this.questionBoundaryPending,
      newUserMessageSinceLastTurn: this.newUserSinceLastCompile,
    });

    this.questionBoundaryPending = false;
    this.newUserSinceLastCompile = false;
    if (dec.committed) {
      this.harnessCommits += 1;
      freedThisTurn += dec.freedTokens;
    }

    const { messages, view } = buildHarnessMessages(
      this.blocks,
      ephemeralTailMessages,
    );
    this.lastEstimate = view.token_budget.used;
    view.token_budget = this.budget(view.token_budget.used);
    // Count only non-sidecar synchronous compile overhead here (sidecar calls
    // inside evaluateBeforeTurn already added their waited ms).
    this.cost.addCriticalPathMs(Math.min(25, Date.now() - t0));

    return {
      messages,
      view,
      decisionReason: dec.reason,
      freedThisTurn,
    };
  }

  /** Flush pending `scope_submit` operations safely after tool-result ingestion. */
  flushPendingScopeClosures(): number {
    let freed = 0;
    while (this.pendingScopeClosures.length > 0) {
      const req = this.pendingScopeClosures.shift()!;
      const collected = this.scopes.collectCollapsibleIds(req.scopeId);
      if (!collected || collected.ids.length === 0) {
        this.scopes.markCompleted(req.scopeId, req.summary, req.keyFacts);
        continue;
      }
      const { node, ids } = collected;
      const bodyLines = [
        `[completed scope #${node.id}: ${node.label}]`,
        `Goal: ${node.goal}`,
        `Conclusion: ${req.summary}`,
        ...(req.keyFacts.length > 0
          ? [`Verified Facts: ${req.keyFacts.join(" | ")}`]
          : []),
        `(Archived ${ids.length} intermediate blocks: ${ids[0]}..${ids[ids.length - 1]} — searchable via context_query)`,
      ];
      const op: ContextOperation = {
        op: "compact",
        ids,
        output: {
          description: `Completed scope ${node.label} (${node.id})`,
          content: bodyLines.join("\n"),
        },
        preserve: req.keyFacts.length > 0 ? req.keyFacts : [req.summary.slice(0, 120)],
        drop: ["raw intermediate scope tool calls"],
      };
      const outcome = applyContextUpdate(this.blocks, this.content, [op], {
        allowReplace: false,
        allowRedact: false,
      });
      if (outcome.ok) {
        const createdId = outcome.created_block_ids[0];
        if (createdId) {
          this.index.upsert(
            createdId,
            "summary",
            `Completed scope ${node.label}`,
            bodyLines.join("\n"),
          );
        }
        this.scopes.markCompleted(
          req.scopeId,
          req.summary,
          req.keyFacts,
          createdId,
          outcome.freed_tokens,
        );
        this.journal.append("patch", {
          actor: "harness",
          reason: `scope_submit (${node.label})`,
          operations: [op],
          result: {
            ...outcome,
            budget: this.budget(this.estimatedUsed()),
          },
        });
        freed += outcome.freed_tokens;
        this.harnessCommits += 1;
      } else {
        this.scopes.markCompleted(req.scopeId, req.summary, req.keyFacts);
      }
    }
    return freed;
  }

  recordMainCall(model: string, usage: ChatUsage, view: ContextView): void {
    this.lastActualPrompt = usage.prompt_tokens;
    this.lastSentBlockIds = new Set(view.blocks);
    this.cost.recordMain(usage);
    this.journal.append("llm_call", {
      model,
      usage,
      visible_blocks: view.blocks.length,
      est_view_tokens: view.token_budget.used,
    });
  }

  estimatedUsed(): number {
    const { view } = buildHarnessMessages(this.blocks);
    const est = view.token_budget.used;
    if (this.lastActualPrompt > 0 && this.lastEstimate > 0) {
      const ratio = Math.min(2, Math.max(0.5, this.lastActualPrompt / this.lastEstimate));
      return Math.round(est * ratio);
    }
    return est;
  }

  costBreakdown(): CostBreakdown {
    return this.cost.snapshot();
  }

  private budget(used: number): TokenBudget {
    return {
      max: this.config.maxContextTokens,
      soft_limit: Math.floor(this.config.maxContextTokens * this.config.softLimitRatio),
      used,
    };
  }

  private latestUserTaskHint(): string {
    const users = this.blocks
      .all()
      .filter((b) => b.kind === "user_message" && typeof b.content === "string");
    if (users.length === 0) return "";
    const first = String(users[0].content).slice(0, 260);
    const last =
      users.length > 1 ? String(users[users.length - 1].content).slice(0, 260) : "";
    return last ? `${first}\nCurrent question: ${last}` : first;
  }

  private releaseReasoning(): void {
    for (const b of this.blocks.all()) {
      if (b.kind === "reasoning_trace" && b.visibility === "api_required") {
        b.visibility = "hidden";
        const parent = b.source_ids[0] ? this.blocks.get(b.source_ids[0]) : undefined;
        if (parent?.api) delete parent.api.reasoning_content;
      }
    }
  }
}
