/**
 * CPAT ContextHarness Extension for Pi Coding Agent (`@earendil-works/pi-coding-agent`).
 *
 * Turns CPAT v2 into a drop-in, near-zero-intrusion context plugin for Pi:
 *   - Usage: `pi -e ./extensions/pi-cpat.ts` (or `pi install <path-to-cpat>`)
 *   - Foreground surface: registers ONLY `context_query` (~90-token prompt hint +
 *     1 lightweight tool schema, vs 9 tools & ~6.6K prompt tokens in model-driven
 *     compression plugins) plus `/cpat` status command for interactive TUI users.
 *   - Write-time lossless indexing: every ingested user message and tool result
 *     is indexed in CPAT's zero-dependency `SearchIndex` (BM25 + Regex + Line-slice),
 *     preserving true source-file line offsets (`read(path, offset, limit)` -> `L<offset>..`).
 *   - Read-time multi-resolution projection:
 *     1. `turn_end` applies GC High/Low Watermark Hysteresis + `ShadowDigester.digestBatch`
 *        (`deepseek-flash` sidecar with `<2ms` `extractSkeleton` + `[Lstart-Lend]` code
 *        symbol spans fallback) via Pi's native append-only `context_edit` session entries,
 *        keeping Prefix Cache 100% byte-identical between batch drains.
 *     2. `context` stays zero-touch during normal turns (preserving Prefix Cache) and
 *        acts as an instant pre-call overflow shield when massive pre-seeded transcripts
 *        (e.g. 110K-200K tokens) arrive before Turn 1.
 *     3. `session_before_compact` cancels redundant threshold compactions when CPAT has
 *        already drained context within budget, and supplies lossless-anchored CPAT
 *        epoch summaries (recording sidecar token usage) on deep-history overflow.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DeepSeekClient, type ChatClient } from "../src/deepseek/client.ts";
import { CostTracker, pricingForModel } from "../src/harness/cost.ts";
import { BlockStore, formatDigestProjection } from "../src/runtime/blocks.ts";
import { SearchIndex } from "../src/runtime/index.ts";
import { ContentStore, Journal } from "../src/runtime/stores.ts";
import { ScopeTree } from "../src/runtime/tree.ts";
import { ShadowDigester } from "../src/sidecar/digester.ts";
import { SubReader } from "../src/sidecar/reader.ts";
import { buildLandmark, extractCodeSymbolSpans, extractSkeleton } from "../src/sidecar/skeleton.ts";
import { projectOversizedToolResult } from "../src/sidecar/tool-projector.ts";
import type { ChatUsage, ContextBlock, ModelPricing } from "../src/types.ts";
import { loadSidecarEnv } from "../src/util/env.ts";
import { bumpIdCounterAtLeast, estimateTokens } from "../src/util/misc.ts";

// Minimal structural host types so this file runs both inside Pi's jiti loader
// AND directly under `node --test` with zero external runtime imports.
export interface PiTextPart {
  type: "text";
  text: string;
}

export interface PiToolCallPart {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface PiUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
}

export interface PiAgentMessage {
  role: string;
  content?: unknown;
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  summary?: string;
  timestamp?: number;
  [key: string]: unknown;
}

export interface PiProjectedEntry {
  sourceEntry: {
    id: string;
    type: string;
    message?: PiAgentMessage;
    [key: string]: unknown;
  };
  messages: PiAgentMessage[];
}

export interface PiExtensionContext {
  cwd: string;
  hasUI?: boolean;
  mode?: string;
  signal?: AbortSignal;
  ui?: {
    notify?: (msg: string, level?: "info" | "warning" | "error") => void;
    setStatus?: (key: string, text: string | undefined) => void;
  };
  model?: {
    id?: string;
    provider?: string;
    contextWindow?: number;
    maxTokens?: number;
  };
  sessionManager: {
    getSessionId(): string;
    buildSessionProjection?(): { entries: PiProjectedEntry[]; messages: PiAgentMessage[] };
    getBranch?(): Array<{ id: string; type: string; [key: string]: unknown }>;
  };
  getContextUsage?(): { tokens: number | null; contextWindow: number; percent: number | null } | undefined;
}

export interface PiContextEditDraft {
  type: "context_edit";
  targetId: string;
  replacement: { content: PiTextPart[] } | null;
}

export interface PiCustomEntryDraft {
  type: "custom";
  customType: string;
  data?: unknown;
}

export interface PiTurnEndEvent {
  type: "turn_end";
  turnIndex: number;
  message: PiAgentMessage;
  toolResults: PiAgentMessage[];
  messageEntryId: string;
  toolResultEntryIds: string[];
  entries: Array<PiContextEditDraft | PiCustomEntryDraft | Record<string, unknown>>;
  continue: boolean;
  outcome: "completed" | "aborted" | "error";
  context: {
    contextEntries?: PiProjectedEntry[];
    contextMessages: PiAgentMessage[];
  };
}

export interface PiContextEvent {
  type: "context";
  messages: PiAgentMessage[];
}

export interface PiBeforeAgentStartEvent {
  type: "before_agent_start";
  systemPrompt: string;
  systemPromptOptions: {
    appendSystemPrompt?: string | string[];
    promptGuidelines?: string[];
    toolSnippets?: Record<string, string>;
    [key: string]: unknown;
  };
  prompt?: string;
}

export interface PiSessionBeforeCompactEvent {
  type: "session_before_compact";
  reason: "manual" | "threshold" | "overflow";
  willRetry: boolean;
  signal: AbortSignal;
  preparation: {
    messagesToSummarize: PiAgentMessage[];
    turnPrefixMessages: PiAgentMessage[];
    previousSummary?: string;
    tokensBefore: number;
    firstKeptEntryId: string;
  };
}

export interface PiExtensionAPI {
  on(event: string, handler: (event: any, ctx: PiExtensionContext) => any): void;
  registerTool(tool: {
    name: string;
    label?: string;
    description: string;
    parameters: Record<string, unknown>;
    execute: (
      toolCallId: string,
      params: Record<string, unknown>,
      signal: AbortSignal | undefined,
      onUpdate: unknown,
      ctx: PiExtensionContext,
    ) => Promise<{ content: PiTextPart[]; details?: unknown; isError?: boolean; usage?: PiUsage }>;
  }): void;
  registerCommand?(
    name: string,
    opts: { description: string; handler: (args: string, ctx: PiExtensionContext) => void | Promise<void> },
  ): void;
}

export interface PiBeforeProviderRequestEvent {
  type: "before_provider_request";
  payload: unknown;
}

export interface PiToolResultEvent {
  type?: "tool_result";
  toolName: string;
  toolCallId: string;
  input?: Record<string, unknown>;
  content: unknown;
  details?: Record<string, any>;
  isError?: boolean;
}

interface PendingAgentCheckpoint {
  scopeId: string;
  mode: "subtask_complete" | "pollution_purge";
  label: string;
  summary: string;
  verifiedFacts: string[];
  purgedAssumptions: string[];
}

interface PendingUserBoundaryFold {
  scopeId: string;
  mode: "pollution_reset" | "task_pivot";
  targetEntryIds: string[];
  startNonSysIdx: number;
  endNonSysIdx: number;
  savedTokens: number;
}

interface WireRequestSnapshot {
  model: string;
  messages: unknown[];
  tools?: unknown;
  rawPayload: Record<string, unknown>;
  affinityHeaders: Record<string, string>;
}

interface IndexedEntryMeta {
  blockId: string;
  entryId?: string;
  toolCallId?: string;
  msgFingerprint: string;
  role: string;
  description: string;
  rawTokens: number;
  activeTokens: number;
  baseLineOffset: number;
  turnIndex: number;
  compacted: boolean;
  compactText?: string;
  symbolLineSummary?: string;
}

export interface CpatPiSessionMetrics {
  sessionId: string;
  effectiveWindowTokens: number;
  indexedBlocks: number;
  compactedBlocks: number;
  rawIndexedTokens: number;
  estimatedSavedTokens: number;
  preCacheToolProjections: number;
  preCacheSavedTokens: number;
  boundaryFolds: number;
  boundaryFoldSavedTokens: number;
  cacheForkSummaries: number;
  cacheForkPromptTokens: number;
  cacheForkCachedTokens: number;
  contextEditsEmitted: number;
  preCallProjections: number;
  compactionsCancelled: number;
  epochCompactions: number;
  contextQueries: number;
  sidecarCalls: number;
  sidecarInputTokens: number;
  sidecarCachedTokens: number;
  sidecarOutputTokens: number;
  sidecarCostUsd: number;
  sidecarBcUnifiedCostUsd: number;
  sidecarLatencyMs: number;
}

export interface CpatPiOptions {
  maxContextTokens?: number;
  strategy?: "cache-first" | "sliding";
  codingToolSoftCapTokens?: number;
  genericRedundantCapTokens?: number;
  hardInlineCapTokens?: number;
  boundaryFoldMinTokens?: number;
  enableCacheForkSummary?: boolean;
  cacheForkFetch?: typeof fetch;
  cacheForkApiKey?: string;
  cacheForkBaseUrl?: string;
  sidecarClient?: ChatClient | null;
  sidecarModel?: string;
  metricsFile?: string;
  storeRootDir?: string;
}

// Billion-Context third-party-1500 unified pricing ($0.15/M input, $0.60/M output, $0.003/M cacheRead)
const BC_UNIFIED_PRICING: ModelPricing = {
  inputMissPerM: 0.15,
  inputHitPerM: 0.003,
  outputPerM: 0.6,
};

const CPAT_SYSTEM_HINT =
  "[CPAT Context Harness] Older bulky tool outputs or transcript chunks may be compacted into `[context:digest #bN]` / `[context:skeleton #bN]` / `[cpat:precache_digest #bN]` / `[cpat:scope_checkpoint #scope_N]` blocks preserving file paths, `Lstart-Lend` code symbol spans, and key facts. Their 100% lossless raw lines remain indexed in memory — call `context_query` (by `block_ids` + `start_line`/`end_line`, `regex`, or `query`) whenever you need exact verbatim code, line numbers, or facts. Call `context_checkpoint` ONLY at explicit subtask completion boundaries or when purging a falsified dead-end hypothesis (`mode: \"pollution_purge\"`), never during normal step-by-step work.";

const CONTEXT_QUERY_SCHEMA = {
  type: "object",
  properties: {
    query: {
      type: "string",
      description:
        "BM25 keyword, identifier, symbol, number, or phrase query across all raw historical tool outputs and messages.",
    },
    regex: {
      type: "string",
      description: "Case-insensitive regular expression to grep exact lines across raw blocks.",
    },
    block_ids: {
      type: "array",
      items: { type: "string" },
      description: 'Optional `#bN` block IDs (e.g. ["b3"]) to restrict search or line slicing.',
    },
    start_line: {
      type: "number",
      description: "1-indexed start line (file line or block line) when slicing `block_ids`.",
    },
    end_line: {
      type: "number",
      description: "1-indexed inclusive end line when slicing `block_ids`.",
    },
  },
  additionalProperties: false,
};

const CONTEXT_CHECKPOINT_SCHEMA = {
  type: "object",
  properties: {
    mode: {
      type: "string",
      enum: ["subtask_complete", "pollution_purge"],
      description:
        "`subtask_complete` when a self-contained investigation/task phase is finished; `pollution_purge` when prior reasoning pursued a wrong hypothesis/dead-end that should be folded out of active context.",
    },
    label: {
      type: "string",
      description: "Short identifier for the completed/purged scope (e.g. 'legacy-v1-hypothesis').",
    },
    summary: {
      type: "string",
      description: "Concise 1-3 sentence verified conclusion or pivot rationale.",
    },
    verified_facts: {
      type: "array",
      items: { type: "string" },
      description:
        "Ground-truth facts, exact file paths, `Lstart-Lend` spans, identifiers, or numbers to preserve inline.",
    },
    purged_assumptions: {
      type: "array",
      items: { type: "string" },
      description: "Optional falsified hypotheses or wrong assumptions being explicitly discarded.",
    },
  },
  required: ["mode", "label", "summary", "verified_facts"],
  additionalProperties: false,
};

class SessionHarnessState {
  readonly sessionId: string;
  readonly dir: string;
  readonly content: ContentStore;
  readonly journal: Journal;
  readonly blocks: BlockStore;
  readonly scopes: ScopeTree;
  readonly index: SearchIndex;
  readonly reader: SubReader;
  readonly cost: CostTracker;
  readonly bcUnifiedCost: CostTracker;
  readonly digester: ShadowDigester;
  readonly sidecarClient?: ChatClient;
  readonly sidecarModel: string;

  readonly byEntryId = new Map<string, IndexedEntryMeta>();
  readonly byToolCallId = new Map<string, IndexedEntryMeta>();
  readonly byFingerprint = new Map<string, IndexedEntryMeta>();
  readonly orderedBlocks: IndexedEntryMeta[] = [];
  readonly foldedEntryIds = new Set<string>();

  latestTaskHint = "";
  nextSeq = 1;
  preCacheToolProjections = 0;
  preCacheSavedTokens = 0;
  boundaryFolds = 0;
  boundaryFoldSavedTokens = 0;
  cacheForkSummaries = 0;
  cacheForkPromptTokens = 0;
  cacheForkCachedTokens = 0;
  contextEditsEmitted = 0;
  preCallProjections = 0;
  compactionsCancelled = 0;
  epochCompactions = 0;
  contextQueries = 0;
  sidecarLatencyMs = 0;
  sidecarCachedTokens = 0;
  priorSidecarCalls = 0;
  priorSidecarInputTokens = 0;
  priorSidecarOutputTokens = 0;
  priorSidecarCostUsd = 0;
  priorSidecarBcCostUsd = 0;
  lastEffectiveWindow = 400_000;

  lastWireRequest?: WireRequestSnapshot;
  lastAffinityHeaders: Record<string, string> = {};
  lastFoldAnchorEntryId?: string;
  manualFoldNote?: string;
  pendingAgentCheckpoint?: PendingAgentCheckpoint;
  pendingUserBoundaryFold?: PendingUserBoundaryFold;

  constructor(sessionId: string, opts: CpatPiOptions) {
    this.sessionId = sessionId;
    const baseDir = opts.storeRootDir ?? process.env.CPAT_STORE_DIR ?? path.join(os.tmpdir(), "cpat-pi-sessions");
    this.dir = path.join(baseDir, sessionId.replace(/[^A-Za-z0-9_-]/g, "_"));
    fs.mkdirSync(path.join(this.dir, "raw-index"), { recursive: true });

    this.content = new ContentStore(this.dir);
    this.journal = new Journal(this.dir);
    this.blocks = new BlockStore(this.content, this.journal);
    this.scopes = new ScopeTree(this.blocks, this.journal);
    this.index = new SearchIndex();

    if (opts.sidecarClient !== undefined) {
      const rawClient = opts.sidecarClient ?? undefined;
      this.sidecarModel = opts.sidecarModel ?? process.env.CPAT_SIDECAR_MODEL ?? "deepseek-flash";
      this.sidecarClient = rawClient ? this.wrapSidecarAccounting(rawClient) : undefined;
    } else {
      const envCfg = resolveSidecarConfig();
      this.sidecarModel = opts.sidecarModel ?? envCfg?.model ?? "deepseek-flash";
      this.sidecarClient = envCfg?.client ? this.wrapSidecarAccounting(envCfg.client) : undefined;
    }

    const p = pricingForModel(this.sidecarModel);
    this.cost = new CostTracker({ mainPricing: p, sidecarPricing: p });
    this.bcUnifiedCost = new CostTracker({
      mainPricing: BC_UNIFIED_PRICING,
      sidecarPricing: BC_UNIFIED_PRICING,
    });

    this.digester = new ShadowDigester({
      client: this.sidecarClient,
      model: this.sidecarModel,
      blocks: this.blocks,
      journal: this.journal,
      cost: this.cost,
      minTokens: 320,
      maxConcurrent: 3,
    });
    this.reader = new SubReader({
      index: this.index,
      client: this.sidecarClient,
      model: this.sidecarModel,
      journal: this.journal,
      cost: this.cost,
    });

    this.restorePersistedIndex();
  }

  private restorePersistedIndex(): void {
    const snapPath = path.join(this.dir, "state-snapshot.json");
    if (!fs.existsSync(snapPath)) return;
    try {
      const saved = JSON.parse(fs.readFileSync(snapPath, "utf8")) as {
        orderedBlocks?: IndexedEntryMeta[];
        metrics?: Partial<CpatPiSessionMetrics>;
      };
      let maxNum = 0;
      for (const meta of saved.orderedBlocks ?? []) {
        const rawFile = path.join(this.dir, "raw-index", `${meta.blockId}.txt`);
        if (!fs.existsSync(rawFile)) continue;
        const raw = fs.readFileSync(rawFile, "utf8");
        this.index.upsert(meta.blockId, meta.role, meta.description, raw, meta.baseLineOffset);
        this.orderedBlocks.push(meta);
        if (meta.entryId) this.byEntryId.set(meta.entryId, meta);
        if (meta.toolCallId) this.byToolCallId.set(meta.toolCallId, meta);
        this.byFingerprint.set(meta.msgFingerprint, meta);
        const m = meta.blockId.match(/_(\d+)$/);
        if (m) maxNum = Math.max(maxNum, Number(m[1]));
      }
      if (maxNum > 0) bumpIdCounterAtLeast(maxNum);
      if (saved.metrics) {
        this.preCacheToolProjections = saved.metrics.preCacheToolProjections ?? 0;
        this.preCacheSavedTokens = saved.metrics.preCacheSavedTokens ?? 0;
        this.boundaryFolds = saved.metrics.boundaryFolds ?? 0;
        this.boundaryFoldSavedTokens = saved.metrics.boundaryFoldSavedTokens ?? 0;
        this.cacheForkSummaries = saved.metrics.cacheForkSummaries ?? 0;
        this.cacheForkPromptTokens = saved.metrics.cacheForkPromptTokens ?? 0;
        this.cacheForkCachedTokens = saved.metrics.cacheForkCachedTokens ?? 0;
        this.contextEditsEmitted = saved.metrics.contextEditsEmitted ?? 0;
        this.preCallProjections = saved.metrics.preCallProjections ?? 0;
        this.compactionsCancelled = saved.metrics.compactionsCancelled ?? 0;
        this.epochCompactions = saved.metrics.epochCompactions ?? 0;
        this.contextQueries = saved.metrics.contextQueries ?? 0;
        this.sidecarLatencyMs = saved.metrics.sidecarLatencyMs ?? 0;
        this.sidecarCachedTokens = saved.metrics.sidecarCachedTokens ?? 0;
        this.priorSidecarCalls = saved.metrics.sidecarCalls ?? 0;
        this.priorSidecarInputTokens = saved.metrics.sidecarInputTokens ?? 0;
        this.priorSidecarOutputTokens = saved.metrics.sidecarOutputTokens ?? 0;
        this.priorSidecarCostUsd = saved.metrics.sidecarCostUsd ?? 0;
        this.priorSidecarBcCostUsd = saved.metrics.sidecarBcUnifiedCostUsd ?? 0;
      }
    } catch {
      // ignore corrupted snapshot
    }
  }

  private wrapSidecarAccounting(inner: ChatClient): ChatClient {
    return {
      chat: async (req) => {
        const resp = await inner.chat(req);
        this.sidecarCachedTokens += resp.usage.prompt_cache_hit_tokens ?? 0;
        this.bcUnifiedCost.recordSidecar(resp.usage);
        return resp;
      },
    };
  }

  ingestRaw(params: {
    entryId?: string;
    toolCallId?: string;
    role: string;
    description: string;
    text: string;
    baseLineOffset?: number;
    turnIndex: number;
  }): IndexedEntryMeta | undefined {
    const raw = params.text;
    if (!raw || raw.trim().length < 24) return undefined;
    if (
      raw.startsWith("[context:digest #") ||
      raw.startsWith("[context:skeleton #") ||
      raw.startsWith("[context:landmark #") ||
      raw.startsWith("[cpat:precache_digest #") ||
      raw.startsWith("[cpat:scope_checkpoint #")
    ) {
      return undefined;
    }

    if (params.toolCallId && this.byToolCallId.has(params.toolCallId)) {
      const existing = this.byToolCallId.get(params.toolCallId)!;
      if (params.entryId && !existing.entryId) {
        existing.entryId = params.entryId;
        this.byEntryId.set(params.entryId, existing);
      }
      return existing;
    }
    if (params.entryId && this.byEntryId.has(params.entryId)) {
      return this.byEntryId.get(params.entryId);
    }
    const fp = fingerprintOf(params.role, raw);
    if (!params.entryId && this.byFingerprint.has(fp)) {
      return this.byFingerprint.get(fp);
    }

    const baseLine = Math.max(1, Math.floor(params.baseLineOffset ?? 1));
    const rawTokens = estimateTokens(raw);
    const lines = raw.split("\n");
    const spans = extractCodeSymbolSpans(lines, baseLine, 18);
    const symbolLineSummary =
      spans.length > 0
        ? `Symbols: ${spans.map((s) => `L${s.startLine}-L${s.endLine} ${s.kind} ${s.name}`).join("; ")}`
        : undefined;

    // Numbered view for ShadowDigester when baseLine > 1 or code spans exist so
    // LLM-extracted line anchors match exact file lines!
    const numberedForDigest =
      baseLine > 1 || spans.length > 0
        ? lines.map((l, idx) => `L${baseLine + idx}: ${l}`).join("\n")
        : raw;

    const created = this.blocks.create({
      idPrefix: "b",
      role: params.role === "user" ? "user" : "tool",
      kind: params.role === "user" ? "user_message" : "tool_result",
      description: params.description,
      content: numberedForDigest,
      protected: false,
    });
    const blockId = created.id;
    const skel = extractSkeleton(blockId, params.description, raw, 1450, baseLine);
    const land = buildLandmark(blockId, params.description, rawTokens, lines.length);
    created.resolutions = {
      ...created.resolutions,
      skeleton: skel,
      landmark: land,
    };

    this.scopes.registerBlock(blockId);
    this.index.upsert(blockId, params.role, params.description, raw, baseLine);
    try {
      fs.writeFileSync(path.join(this.dir, "raw-index", `${blockId}.txt`), raw, "utf8");
    } catch {
      // non-fatal
    }

    const meta: IndexedEntryMeta = {
      blockId,
      entryId: params.entryId,
      toolCallId: params.toolCallId,
      msgFingerprint: fp,
      role: params.role,
      description: params.description,
      rawTokens,
      activeTokens: rawTokens,
      baseLineOffset: baseLine,
      turnIndex: params.turnIndex,
      compacted: false,
      symbolLineSummary,
    };

    if (params.entryId) this.byEntryId.set(params.entryId, meta);
    if (params.toolCallId) this.byToolCallId.set(params.toolCallId, meta);
    this.byFingerprint.set(fp, meta);
    this.orderedBlocks.push(meta);
    return meta;
  }

  async compactTargets(
    targets: IndexedEntryMeta[],
    mode: "digest" | "skeleton" | "landmark",
  ): Promise<void> {
    if (targets.length === 0) return;

    if (mode === "digest" && this.sidecarClient) {
      const cpatBlocks: ContextBlock[] = [];
      for (const t of targets) {
        const b = this.blocks.get(t.blockId);
        if (b && t.rawTokens >= 340) {
          if (b.kind !== "tool_result") {
            (b as { kind: ContextBlock["kind"] }).kind = "tool_result";
          }
          cpatBlocks.push(b);
        }
      }
      if (cpatBlocks.length > 0) {
        const ms = await this.digester.digestBatch(cpatBlocks, this.latestTaskHint, 8500);
        this.sidecarLatencyMs += ms;
      }
    }

    for (const t of targets) {
      const b = this.blocks.get(t.blockId);
      if (!b) continue;
      let rendered: string;
      if (mode === "landmark") {
        rendered = b.resolutions?.landmark ?? buildLandmark(t.blockId, t.description, t.rawTokens, 1);
      } else if (mode === "digest" && b.resolutions?.digest) {
        const baseDigest = b.resolutions.digest;
        rendered = t.symbolLineSummary ? `${baseDigest}\n${t.symbolLineSummary}` : baseDigest;
      } else if (mode === "digest" && b.digest) {
        const baseDigest = formatDigestProjection(t.blockId, t.rawTokens, b.digest);
        rendered = t.symbolLineSummary ? `${baseDigest}\n${t.symbolLineSummary}` : baseDigest;
      } else {
        rendered =
          b.resolutions?.skeleton ??
          extractSkeleton(t.blockId, t.description, String(b.content), 1450, t.baseLineOffset);
      }

      const newTok = estimateTokens(rendered);
      if (newTok < t.rawTokens * 0.85) {
        t.compacted = true;
        t.compactText = rendered;
        t.activeTokens = newTok;
      }
    }
  }

  snapshotMetrics(): CpatPiSessionMetrics {
    const snap = this.cost.snapshot();
    const bcSnap = this.bcUnifiedCost.snapshot();
    const rawIndexedTokens = this.orderedBlocks.reduce((s, b) => s + b.rawTokens, 0);
    const compactedBlocks = this.orderedBlocks.filter((b) => b.compacted).length;
    const estimatedSavedTokens = this.orderedBlocks.reduce(
      (s, b) => s + (b.compacted ? Math.max(0, b.rawTokens - b.activeTokens) : 0),
      0,
    );
    return {
      sessionId: this.sessionId,
      effectiveWindowTokens: this.lastEffectiveWindow,
      indexedBlocks: this.orderedBlocks.length,
      compactedBlocks,
      rawIndexedTokens,
      estimatedSavedTokens,
      preCacheToolProjections: this.preCacheToolProjections,
      preCacheSavedTokens: this.preCacheSavedTokens,
      boundaryFolds: this.boundaryFolds,
      boundaryFoldSavedTokens: this.boundaryFoldSavedTokens,
      cacheForkSummaries: this.cacheForkSummaries,
      cacheForkPromptTokens: this.cacheForkPromptTokens,
      cacheForkCachedTokens: this.cacheForkCachedTokens,
      contextEditsEmitted: this.contextEditsEmitted,
      preCallProjections: this.preCallProjections,
      compactionsCancelled: this.compactionsCancelled,
      epochCompactions: this.epochCompactions,
      contextQueries: this.contextQueries,
      sidecarCalls: this.priorSidecarCalls + snap.sidecar_calls,
      sidecarInputTokens: this.priorSidecarInputTokens + snap.sidecar_prompt_tokens,
      sidecarCachedTokens: this.sidecarCachedTokens,
      sidecarOutputTokens: this.priorSidecarOutputTokens + snap.sidecar_completion_tokens,
      sidecarCostUsd: Number((this.priorSidecarCostUsd + snap.sidecar_total_usd).toFixed(6)),
      sidecarBcUnifiedCostUsd: Number((this.priorSidecarBcCostUsd + bcSnap.sidecar_total_usd).toFixed(6)),
      sidecarLatencyMs: this.sidecarLatencyMs,
    };
  }

  flushMetricsFile(explicitPath?: string): void {
    const metrics = this.snapshotMetrics();
    try {
      fs.writeFileSync(
        path.join(this.dir, "state-snapshot.json"),
        JSON.stringify({ orderedBlocks: this.orderedBlocks, metrics }, null, 2),
        "utf8",
      );
    } catch {
      // non-fatal
    }
    const outPath = explicitPath ?? process.env.CPAT_METRICS_FILE;
    if (!outPath) return;
    try {
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, JSON.stringify(metrics, null, 2), "utf8");
    } catch {
      // non-fatal telemetry write
    }
  }
}

function resolveSidecarConfig(): { client: ChatClient; model: string } | undefined {
  try {
    if (process.env.CPAT_DISABLE_SIDECAR === "1") return undefined;
    const apiKey =
      process.env.SIDECAR_API_KEY ||
      process.env.DEEPSEEK_API_KEY ||
      process.env.API_KEY ||
      process.env.OPENAI_API_KEY;
    if (apiKey) {
      const rawUrl =
        process.env.SIDECAR_BASE_URL ||
        process.env.OPENAI_BASE_URL ||
        process.env.DEEPSEEK_BASE_URL ||
        "https://api.deepseek.com";
      const baseUrl = rawUrl.replace(/\/+$/, "");
      const model = process.env.CPAT_SIDECAR_MODEL || process.env.SIDECAR_MODEL || "deepseek-flash";
      return { client: new DeepSeekClient({ apiKey, baseUrl }), model };
    }
    const cfg = loadSidecarEnv();
    return { client: new DeepSeekClient({ apiKey: cfg.apiKey, baseUrl: cfg.baseUrl }), model: cfg.model };
  } catch {
    return undefined;
  }
}

function extractMessageText(msg: PiAgentMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((p): p is { type: string; text?: string } => typeof p === "object" && p !== null)
      .filter((p) => p.type === "text" && typeof p.text === "string")
      .map((p) => p.text!)
      .join("\n");
  }
  if (typeof msg.summary === "string") return msg.summary;
  return "";
}

function estimateMessagesTokens(messages: PiAgentMessage[]): number {
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(extractMessageText(m)) + 8;
    if (Array.isArray(m.content)) {
      for (const p of m.content) {
        if (typeof p === "object" && p !== null && (p as { type?: string }).type === "toolCall") {
          total += estimateTokens(JSON.stringify((p as { arguments?: unknown }).arguments ?? {})) + 12;
        }
      }
    }
  }
  return total;
}

function fingerprintOf(role: string, text: string): string {
  const head = text.slice(0, 120);
  const tail = text.slice(-120);
  return `${role}:${text.length}:${head}:${tail}`;
}

function resolveEffectiveWindow(ctx: PiExtensionContext, opts: CpatPiOptions): number {
  const envCap = Number(process.env.CPAT_MAX_CONTEXT);
  if (opts.maxContextTokens && opts.maxContextTokens > 0) return opts.maxContextTokens;
  if (Number.isFinite(envCap) && envCap >= 4_000) return Math.floor(envCap);
  const modelWin = ctx.model?.contextWindow;
  if (typeof modelWin === "number" && modelWin > 0) {
    return modelWin;
  }
  return 400_000;
}

function describeToolCall(
  toolName: string,
  args: Record<string, unknown> | undefined,
): { description: string; baseLineOffset: number } {
  if (!args) return { description: `tool:${toolName}`, baseLineOffset: 1 };
  if (toolName === "read") {
    const p = String(args.path ?? args.file_path ?? "");
    const offset = Number(args.offset) || 1;
    const limit = args.limit ? `, limit=${args.limit}` : "";
    return {
      description: `read(${p}${offset > 1 ? `, offset=${offset}` : ""}${limit})`,
      baseLineOffset: Math.max(1, offset),
    };
  }
  if (toolName === "grep") {
    return {
      description: `grep(pattern=${JSON.stringify(String(args.pattern ?? "").slice(0, 50))}, path=${String(args.path ?? ".")})`,
      baseLineOffset: 1,
    };
  }
  if (toolName === "bash") {
    return {
      description: `bash(${String(args.command ?? "").replace(/\s+/g, " ").trim().slice(0, 70)})`,
      baseLineOffset: 1,
    };
  }
  const summary = JSON.stringify(args).slice(0, 70);
  return { description: `${toolName}(${summary})`, baseLineOffset: 1 };
}

export function createCpatPiExtension(opts: CpatPiOptions = {}) {
  return function cpatPiExtension(pi: PiExtensionAPI): void {
    const sessions = new Map<string, SessionHarnessState>();
    const resolveStrategy = (): "cache-first" | "sliding" => {
      if (opts.strategy) return opts.strategy;
      return process.env.CPAT_STRATEGY === "sliding" ? "sliding" : "cache-first";
    };

    function getState(ctx: PiExtensionContext): SessionHarnessState {
      const sid = ctx.sessionManager?.getSessionId?.() || "default";
      let st = sessions.get(sid);
      if (!st) {
        st = new SessionHarnessState(sid, opts);
        sessions.set(sid, st);
      }
      st.lastEffectiveWindow = resolveEffectiveWindow(ctx, opts);
      return st;
    }

    // 1. Register single lightweight recovery tool: `context_query`
    pi.registerTool({
      name: "context_query",
      label: "CPAT Context Query",
      description:
        "Retrieve exact verbatim lines, numbers, identifiers, or code spans from CPAT's lossless memory index across all earlier tool outputs and conversation turns (including compacted `#bN` blocks).",
      parameters: CONTEXT_QUERY_SCHEMA,
      async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
        const st = getState(ctx);
        st.contextQueries += 1;
        const blockIds = Array.isArray(params.block_ids)
          ? params.block_ids.filter((x): x is string => typeof x === "string")
          : undefined;
        const hasLineRange =
          typeof params.start_line === "number" || typeof params.end_line === "number";
        const regexStr = typeof params.regex === "string" ? params.regex.trim() : "";
        const queryStr = typeof params.query === "string" ? params.query.trim() : "";

        let mode: "snippets" | "grep" | "raw_range" = "snippets";
        let q = queryStr;
        if (blockIds && blockIds.length > 0 && hasLineRange && !queryStr && !regexStr) {
          mode = "raw_range";
        } else if (regexStr) {
          mode = "grep";
          q = regexStr;
        } else if (!q && blockIds && blockIds.length > 0) {
          mode = "raw_range";
        }

        const text = await st.reader.query({
          query: q,
          mode,
          block_ids: blockIds,
          start_line: typeof params.start_line === "number" ? params.start_line : undefined,
          end_line: typeof params.end_line === "number" ? params.end_line : undefined,
        });
        st.flushMetricsFile(opts.metricsFile);
        return {
          content: [{ type: "text", text }],
          details: { mode, blockIds },
        };
      },
    });

    // 1b. Register explicit subtask / pollution-purge frame-folding tool: `context_checkpoint`
    pi.registerTool({
      name: "context_checkpoint",
      label: "CPAT Scope Checkpoint",
      description:
        "Fold completed subtask exploration or purge falsified dead-end reasoning (`Context Pollution`) at an explicit task boundary while keeping 100% of raw tool outputs searchable via `context_query`. Call ONLY when finishing a distinct subtask phase or abandoning a wrong hypothesis — never during routine progress.",
      parameters: CONTEXT_CHECKPOINT_SCHEMA,
      async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
        const st = getState(ctx);
        const mode =
          params.mode === "pollution_purge" ? "pollution_purge" : "subtask_complete";
        const label = String(params.label ?? "subtask").trim() || "subtask";
        const summary = String(params.summary ?? "").trim();
        const verifiedFacts = Array.isArray(params.verified_facts)
          ? params.verified_facts.filter((x): x is string => typeof x === "string" && x.trim().length > 0)
          : [];
        const purgedAssumptions = Array.isArray(params.purged_assumptions)
          ? params.purged_assumptions.filter((x): x is string => typeof x === "string" && x.trim().length > 0)
          : [];

        const scopeNode =
          st.scopes.activeScope() ??
          st.scopes.openScope({
            label,
            goal: st.latestTaskHint || label,
          });
        st.scopes.markCompleted(scopeNode.id, summary, verifiedFacts);

        st.pendingAgentCheckpoint = {
          scopeId: scopeNode.id,
          mode,
          label,
          summary,
          verifiedFacts,
          purgedAssumptions,
        };

        const checkpointText = formatScopeCheckpointBlock({
          scopeId: scopeNode.id,
          mode,
          label,
          summary,
          verifiedFacts,
          purgedAssumptions,
          recentBlocks: st.orderedBlocks.slice(-12),
        });
        st.flushMetricsFile(opts.metricsFile);
        return {
          content: [{ type: "text", text: checkpointText }],
          details: { scopeId: scopeNode.id, mode },
        };
      },
    });

    // 2. Interactive `/cpat` status & `/cpat-fold` explicit boundary commands
    pi.registerCommand?.("cpat", {
      description: "Show CPAT ContextHarness block, token, cache-fork, and boundary-fold metrics",
      handler: async (_args, ctx) => {
        const m = getState(ctx).snapshotMetrics();
        ctx.ui?.notify?.(
          `CPAT: ${m.indexedBlocks} blocks (${m.compactedBlocks} compacted, saved ~${m.estimatedSavedTokens.toLocaleString()} tok), folds=${m.boundaryFolds}, forkSummaries=${m.cacheForkSummaries}, queries=${m.contextQueries}, sidecar=$${m.sidecarCostUsd.toFixed(4)}`,
          "info",
        );
      },
    });

    pi.registerCommand?.("cpat-fold", {
      description: "Fold prior exploratory/polluted turns at the next prompt boundary while keeping raw blocks in SearchIndex",
      handler: async (args, ctx) => {
        const st = getState(ctx);
        st.manualFoldNote = args.trim() || "Manual /cpat-fold task boundary";
        ctx.ui?.notify?.(
          `CPAT: armed zero-suffix ScopeTree boundary fold for next prompt (${st.manualFoldNote})`,
          "info",
        );
      },
    });

    // 2b. Read-only wire prefix & affinity header observer (`before_provider_headers` + `before_provider_request`):
    //     Captures the exact warm payload (`thinking`, `reasoning_effort`, `tools`, `messages`)
    //     and `x-session-affinity` header sent to the LLM so `session_before_compact` at 384K
    //     hits ~95%+ KV Cache ($0.003/M instead of $0.15/M)!
    pi.on("before_provider_headers", (event: { headers?: Record<string, unknown> }, ctx) => {
      if (event?.headers && typeof event.headers === "object") {
        const st = getState(ctx);
        const captured: Record<string, string> = {};
        for (const [k, v] of Object.entries(event.headers)) {
          const low = k.toLowerCase();
          if (
            typeof v === "string" &&
            v.length > 0 &&
            (low === "x-session-affinity" || low === "x-client-request-id" || low === "session_id" || low === "x-session-id")
          ) {
            captured[k] = v;
          }
        }
        st.lastAffinityHeaders = captured;
      }
      return undefined;
    });

    pi.on("before_provider_request", (event: PiBeforeProviderRequestEvent, ctx) => {
      if (event?.payload && typeof event.payload === "object") {
        const p = event.payload as Record<string, unknown>;
        if (typeof p.model === "string" && Array.isArray(p.messages) && p.messages.length > 0) {
          const st = getState(ctx);
          st.lastWireRequest = {
            model: p.model,
            messages: p.messages.slice(),
            ...(p.tools !== undefined ? { tools: p.tools } : {}),
            rawPayload: { ...p },
            affinityHeaders: {
              "x-client-request-id": st.sessionId,
              "x-session-affinity": st.sessionId,
              ...st.lastAffinityHeaders,
            },
          };
        }
      }
      return undefined;
    });

    // 3. Inject system hint + detect Explicit User Task-Boundary / Pollution-Reset (`before_agent_start`)
    pi.on("before_agent_start", (event: PiBeforeAgentStartEvent, ctx) => {
      const st = getState(ctx);
      const promptText = typeof event.prompt === "string" ? event.prompt.trim() : "";
      if (promptText) {
        st.latestTaskHint = promptText.slice(0, 500);
      }
      const cur = event.systemPromptOptions.appendSystemPrompt;
      if (Array.isArray(cur)) {
        if (!cur.some((s) => s.includes("[CPAT Context Harness]"))) {
          event.systemPromptOptions.appendSystemPrompt = [...cur, CPAT_SYSTEM_HINT];
        }
      } else {
        const text = typeof cur === "string" ? cur : "";
        if (!text.includes("[CPAT Context Harness]")) {
          event.systemPromptOptions.appendSystemPrompt = text
            ? `${text}\n\n${CPAT_SYSTEM_HINT}`
            : CPAT_SYSTEM_HINT;
        }
      }

      // Check for explicit user task-pivot or context-pollution reset BEFORE the new user
      // prompt is appended (`suffixCachedTokens == 0`!).
      const manualNote = st.manualFoldNote;
      st.manualFoldNote = undefined;
      const boundaryIntent = manualNote
        ? ({ mode: "task_pivot" as const, explicitTag: true })
        : detectExplicitBoundaryIntent(promptText);

      if (!boundaryIntent) return undefined;

      const projection = ctx.sessionManager.buildSessionProjection?.();
      const preEntries = projection?.entries ?? [];
      if (preEntries.length === 0) return undefined;

      // Locate entries strictly after `lastFoldAnchorEntryId` so already-checkpointed
      // parent scopes stay 100% frozen in Prefix Cache!
      let startEntryIdx = 0;
      if (st.lastFoldAnchorEntryId) {
        const found = preEntries.findIndex((e) => e.sourceEntry.id === st.lastFoldAnchorEntryId);
        if (found >= 0) startEntryIdx = found + 1;
      }

      const targetEntryIds: string[] = [];
      let collapsibleTokens = 0;
      // IMPORTANT: `pi.on("context")` receives `messages` WITHOUT `role === "system"`!
      // Count strictly non-system messages so `startNonSysIdx..endNonSysIdx` never shifts by +1!
      let nonSysCursor = 0;
      let startNonSysIdx = 0;

      for (let i = 0; i < preEntries.length; i++) {
        const pe = preEntries[i];
        const msgs = (pe.messages ?? []).filter((m) => m.role !== "system");
        if (i === startEntryIdx) {
          startNonSysIdx = nonSysCursor;
        }
        if (i >= startEntryIdx) {
          for (let mIdx = 0; mIdx < msgs.length; mIdx++) {
            const msg = msgs[mIdx];
            const text = extractMessageText(msg);
            if (
              (msg.role === "toolResult" || msg.role === "user") &&
              text.length >= 120
            ) {
              st.ingestRaw({
                entryId: pe.sourceEntry.id,
                role: msg.role === "toolResult" ? "tool" : "user",
                description: `${msg.role} entry ${pe.sourceEntry.id}`,
                text,
                turnIndex: 0,
              });
            }
          }

          const firstRole = msgs[0]?.role ?? pe.sourceEntry.message?.role;
          const firstText = msgs[0] ? extractMessageText(msgs[0]) : "";
          const isScopeCheckpointMsg = firstText.startsWith("[cpat:scope_checkpoint #");
          if (
            (firstRole === "assistant" || firstRole === "toolResult") &&
            !isScopeCheckpointMsg &&
            !st.foldedEntryIds.has(pe.sourceEntry.id)
          ) {
            targetEntryIds.push(pe.sourceEntry.id);
            collapsibleTokens += estimateMessagesTokens(msgs);
          }
        }
        nonSysCursor += msgs.length;
      }
      if (startEntryIdx >= preEntries.length) {
        startNonSysIdx = nonSysCursor;
      }
      const endNonSysIdx = nonSysCursor;

      const minFoldTokens = boundaryIntent.explicitTag
        ? 400
        : (opts.boundaryFoldMinTokens ??
          (Number(process.env.CPAT_BOUNDARY_FOLD_MIN_TOKENS) || 8_000));

      if (targetEntryIds.length === 0 || collapsibleTokens < minFoldTokens) {
        return undefined;
      }

      const scopeNode =
        st.scopes.activeScope() ??
        st.scopes.openScope({
          label: boundaryIntent.mode === "pollution_reset" ? "polluted-scope-reset" : "completed-task-scope",
          goal: promptText.slice(0, 200),
        });
      const summaryLine =
        manualNote ??
        (boundaryIntent.mode === "pollution_reset"
          ? "AC3 Stage-1 Authoritative-Only Reset: Purged prior exploratory/polluted assistant trajectory upon explicit user correction while retaining 100% lossless raw observations in SearchIndex."
          : "Folded completed prior task scope at explicit user task boundary while retaining 100% lossless raw observations in SearchIndex.");
      st.scopes.markCompleted(scopeNode.id, summaryLine, [], undefined, collapsibleTokens);

      // AC3 Stage-1 Authoritative-Only Rule: include ONLY ground-truth raw tool/user landmarks,
      // NEVER tainted assistant conclusions!
      const authoritativeFacts = st.orderedBlocks
        .filter((b) => b.role === "tool")
        .slice(-15)
        .map(
          (b) =>
            `#${b.blockId}: ${b.description} (${b.rawTokens.toLocaleString()} raw tok${b.symbolLineSummary ? ` | ${b.symbolLineSummary.slice(0, 110)}` : ""})`,
        );

      const checkpointText = formatScopeCheckpointBlock({
        scopeId: scopeNode.id,
        mode: boundaryIntent.mode === "pollution_reset" ? "pollution_purge" : "subtask_complete",
        label: scopeNode.label,
        summary: summaryLine,
        verifiedFacts: authoritativeFacts,
        purgedAssumptions:
          boundaryIntent.mode === "pollution_reset"
            ? ["All unverified hypotheses and assistant conclusions from the prior scope have been purged from active context; rely strictly on authoritative user directives and raw tool evidence."]
            : [],
        recentBlocks: st.orderedBlocks.slice(-10),
      });

      st.pendingUserBoundaryFold = {
        scopeId: scopeNode.id,
        mode: boundaryIntent.mode,
        targetEntryIds,
        startNonSysIdx,
        endNonSysIdx,
        savedTokens: Math.max(0, collapsibleTokens - estimateTokens(checkpointText)),
      };

      return {
        message: {
          customType: "cpat:scope_checkpoint",
          content: [{ type: "text", text: checkpointText }],
          display: false,
          details: { scopeId: scopeNode.id, mode: boundaryIntent.mode, foldedEntries: targetEntryIds.length },
        },
      };
    });

    // 3b. Pre-Cache Oversized `tool_result` Gate (`pi.on("tool_result")`):
    //     - Intercepts giant or host-truncated tool outputs BEFORE Turn t+1 ever
    //       prefills or caches them (100% append-only, zero retroactive cache break!).
    //     - Recovers full untruncated `bash` (`details.fullOutputPath`) & `read`
    //       outputs that Pi's 50KB/2,000-line limit dropped, indexing 100% of all
    //       lines into `SearchIndex` and surfacing early `ERROR`/`Traceback` hunks
    //       + full-file `[Lstart-Lend]` symbol spans inline!
    pi.on("tool_result", async (event: PiToolResultEvent, ctx) => {
      const toolName = String(event.toolName ?? "");
      if (!toolName || toolName === "context_query" || toolName === "context_checkpoint") {
        return undefined;
      }

      const st = getState(ctx);
      const { description, baseLineOffset } = describeToolCall(toolName, event.input);
      const modelFacingText = extractMessageText({ role: "toolResult", content: event.content });

      let fullRaw = modelFacingText;
      let hostWasTruncated = false;

      // 1. Recover untruncated bash output from `details.fullOutputPath` (up to 8 MiB safety cap)
      const fullOutPath = event.details?.fullOutputPath;
      if (typeof fullOutPath === "string" && fullOutPath.length > 0) {
        try {
          const stat = fs.statSync(fullOutPath);
          if (stat.isFile() && stat.size > 0 && stat.size <= 8 * 1024 * 1024) {
            fullRaw = fs.readFileSync(fullOutPath, "utf8");
            hostWasTruncated = true;
          }
        } catch {
          // fallback to modelFacingText
        }
      }

      // 2. Recover untruncated file lines when Pi's `read` hit its 50KB / 2,000-line head limit
      if (
        !hostWasTruncated &&
        toolName === "read" &&
        event.details?.truncation?.truncated === true &&
        typeof event.input?.path === "string" &&
        event.input?.limit === undefined
      ) {
        try {
          const absPath = path.resolve(ctx.cwd, event.input.path);
          const stat = fs.statSync(absPath);
          if (stat.isFile() && stat.size > 0 && stat.size <= 8 * 1024 * 1024) {
            const allLines = fs.readFileSync(absPath, "utf8").split("\n");
            fullRaw = allLines.slice(Math.max(0, baseLineOffset - 1)).join("\n");
            hostWasTruncated = true;
          }
        } catch {
          // fallback to modelFacingText
        }
      }

      const meta = st.ingestRaw({
        toolCallId: event.toolCallId,
        role: "tool",
        description,
        text: fullRaw,
        baseLineOffset,
        turnIndex: 0,
      });
      if (!meta) return undefined;

      const codingSoftCap =
        opts.codingToolSoftCapTokens ??
        (Number(process.env.CPAT_CODING_TOOL_CAP_TOKENS) || 24_000);
      const redundantCap =
        opts.genericRedundantCapTokens ??
        (Number(process.env.CPAT_REDUNDANT_TOOL_CAP_TOKENS) || 28_000);
      const hardCap =
        opts.hardInlineCapTokens ??
        (Number(process.env.CPAT_HARD_TOOL_CAP_TOKENS) || 96_000);

      const proj = projectOversizedToolResult({
        blockId: meta.blockId,
        toolName,
        description,
        rawText: fullRaw,
        baseLineOffset,
        hostWasTruncated,
        codingToolSoftCapTokens: codingSoftCap,
        genericRedundantCapTokens: redundantCap,
        hardInlineCapTokens: hardCap,
      });

      if (!proj.shouldProject || !proj.projectedText) {
        return undefined;
      }

      const origVisibleTokens = estimateTokens(modelFacingText);
      st.preCacheToolProjections += 1;
      st.preCacheSavedTokens += Math.max(0, origVisibleTokens - proj.projectedTokens);
      meta.compacted = true;
      meta.compactText = proj.projectedText;
      meta.activeTokens = proj.projectedTokens;
      st.flushMetricsFile(opts.metricsFile);

      return {
        content: [{ type: "text", text: proj.projectedText }],
      };
    });

    // 4. Pre-call guard (`context`):
    //    - Applies one-shot `pendingUserBoundaryFold` on Turn 1 of an explicit user task
    //      pivot / pollution reset (matching the exact `context_edit`s committed at Turn 1
    //      `turn_end`, so Turn 1 -> Turn 2 enjoys 100% Prefix Cache Hit!).
    //    - Otherwise zero-touch (`return undefined`) during normal multi-turn operation!
    pi.on("context", async (event: PiContextEvent, ctx) => {
      const st = getState(ctx);
      const effectiveWin = st.lastEffectiveWindow;

      // Ensure all messages in `event.messages` are losslessly indexed so `context_query`
      // can always search pre-seeded user/assistant transcripts even on Turn 1!
      for (let i = 0; i < event.messages.length; i++) {
        const m = event.messages[i];
        if (m.role !== "user" && m.role !== "toolResult" && m.role !== "assistant") continue;
        const text = extractMessageText(m);
        if (text.length < 240) continue;
        if (m.role === "user" && i === event.messages.length - 1) {
          st.latestTaskHint = text.slice(0, 500);
          continue;
        }
        st.ingestRaw({
          role: m.role === "toolResult" ? "tool" : "user",
          description: `${m.role} message #${i + 1}`,
          text,
          turnIndex: 0,
        });
      }

      if (st.pendingUserBoundaryFold && st.pendingUserBoundaryFold.endNonSysIdx > st.pendingUserBoundaryFold.startNonSysIdx) {
        const { startNonSysIdx, endNonSysIdx } = st.pendingUserBoundaryFold;
        const filtered = event.messages.filter((m, idx) => {
          if (idx < startNonSysIdx || idx >= endNonSysIdx) return true;
          if (m.role !== "assistant" && m.role !== "toolResult") return true;
          if (extractMessageText(m).startsWith("[cpat:scope_checkpoint #")) return true;
          return false;
        });
        return { messages: filtered };
      }

      const estTokens = estimateMessagesTokens(event.messages);
      const strategy = resolveStrategy();
      // In `cache-first` mode, once multi-turn execution is underway (`hasAssistantTurn`),
      // NEVER rewrite `event.messages` ephemerally in `context` (which shifts cuts each turn
      // and uses rough char*0.3 estimates). Let Pi's exact API-usage threshold trigger a
      // single PERSISTED `session_before_compact` checkpoint at `contextWindow - reserveTokens`!
      const hasAssistantTurn = event.messages.some((m) => m.role === "assistant");
      if (strategy === "cache-first" && hasAssistantTurn) {
        return undefined;
      }
      const emergencyBudget = Math.floor(
        effectiveWin * (strategy === "cache-first" ? 0.95 : 0.78),
      );
      if (estTokens <= emergencyBudget) {
        return undefined; // 100% untouched prefix!
      }

      st.preCallProjections += 1;
      const targetTokens = Math.floor(effectiveWin * 0.54);
      const useLandmark = estTokens > effectiveWin * 2.2;
      let projectedTokens = estTokens;
      const nextMessages = event.messages.map((m) => ({ ...m }));

      const protectTailCount = Math.min(3, nextMessages.length);
      const cutoff = Math.max(0, nextMessages.length - protectTailCount);

      for (let i = 0; i < cutoff && projectedTokens > targetTokens; i++) {
        const m = nextMessages[i];
        if (m.role !== "user" && m.role !== "assistant" && m.role !== "toolResult") continue;
        if (
          m.role === "assistant" &&
          Array.isArray(m.content) &&
          m.content.some((p) => typeof p === "object" && p !== null && (p as { type?: string }).type === "toolCall")
        ) {
          continue;
        }
        const text = extractMessageText(m);
        const rawTok = estimateTokens(text);
        if (rawTok < 160) continue;

        const meta = st.byFingerprint.get(fingerprintOf(m.role === "toolResult" ? "tool" : "user", text));
        const blockId = meta?.blockId ?? `raw_${i + 1}`;
        const compact = useLandmark
          ? buildLandmark(blockId, `${m.role} turn #${i + 1}`, rawTok, text.split("\n").length)
          : extractSkeleton(blockId, `${m.role} turn #${i + 1}`, text, 850, meta?.baseLineOffset ?? 1);
        const newTok = estimateTokens(compact);
        if (newTok >= rawTok * 0.85) continue;

        if (typeof m.content === "string") {
          m.content = compact;
        } else {
          m.content = [{ type: "text", text: compact }];
        }
        projectedTokens -= rawTok - newTok;
      }

      st.flushMetricsFile(opts.metricsFile);
      return { messages: nextMessages };
    });

    // 5. Turn-boundary lossless ingest + Explicit ScopeTree Folding + Sliding GC (`turn_end`):
    pi.on("turn_end", async (event: PiTurnEndEvent, ctx) => {
      if (event.outcome !== "completed") return undefined;
      const st = getState(ctx);
      const effectiveWin = st.lastEffectiveWindow;

      const projEntries = event.context?.contextEntries ?? ctx.sessionManager.buildSessionProjection?.()?.entries ?? [];
      for (let i = 0; i < projEntries.length; i++) {
        const pe = projEntries[i];
        const msg = pe.messages?.[0] ?? pe.sourceEntry.message;
        if (!msg || msg.role !== "user") continue;
        const text = extractMessageText(msg);
        if (i === projEntries.length - 1 || i === 0) {
          if (text.trim()) st.latestTaskHint = text.slice(0, 500);
          continue;
        }
        if (estimateTokens(text) >= 400) {
          st.ingestRaw({
            entryId: pe.sourceEntry.id,
            role: "user",
            description: `user chunk entry ${pe.sourceEntry.id}`,
            text,
            turnIndex: Math.max(0, event.turnIndex - 1),
          });
        }
      }

      const callInfoById = new Map<string, { name: string; args: Record<string, unknown> }>();
      if (Array.isArray(event.message?.content)) {
        for (const part of event.message.content) {
          if (typeof part === "object" && part !== null && (part as { type?: string }).type === "toolCall") {
            const tc = part as PiToolCallPart;
            callInfoById.set(tc.id, { name: tc.name, args: tc.arguments ?? {} });
          }
        }
      }

      for (let i = 0; i < event.toolResults.length; i++) {
        const tr = event.toolResults[i];
        const entryId = event.toolResultEntryIds?.[i];
        const toolName = String(tr.toolName ?? callInfoById.get(String(tr.toolCallId ?? ""))?.name ?? "tool");
        if (toolName === "context_query" || toolName === "context_checkpoint") continue;
        const callIdStr = String(tr.toolCallId ?? "");
        const callMeta = callInfoById.get(callIdStr);
        const { description, baseLineOffset } = describeToolCall(toolName, callMeta?.args);
        const text = extractMessageText(tr);
        if (callIdStr && st.byToolCallId.has(callIdStr)) {
          const existing = st.byToolCallId.get(callIdStr)!;
          existing.turnIndex = event.turnIndex;
          if (entryId && !existing.entryId) {
            existing.entryId = entryId;
            st.byEntryId.set(entryId, existing);
          }
          continue;
        }
        st.ingestRaw({
          entryId,
          toolCallId: callIdStr || undefined,
          role: "tool",
          description,
          text,
          baseLineOffset,
          turnIndex: event.turnIndex,
        });
      }

      // 5a. Commit Explicit Boundary / Pollution Scope Fold (`suffixCachedTokens == 0`!)
      const boundaryDrafts: Array<PiContextEditDraft | PiCustomEntryDraft> = [];
      let boundarySaved = 0;

      if (st.pendingUserBoundaryFold) {
        const fold = st.pendingUserBoundaryFold;
        st.pendingUserBoundaryFold = undefined;
        for (const targetId of fold.targetEntryIds) {
          if (st.foldedEntryIds.has(targetId)) continue;
          st.foldedEntryIds.add(targetId);
          boundaryDrafts.push({
            type: "context_edit",
            targetId,
            replacement: null,
          });
          st.contextEditsEmitted += 1;
        }
        if (boundaryDrafts.length > 0) {
          st.boundaryFolds += 1;
          st.boundaryFoldSavedTokens += fold.savedTokens;
          boundarySaved += fold.savedTokens;
          st.lastFoldAnchorEntryId =
            event.toolResultEntryIds?.[event.toolResultEntryIds.length - 1] ?? event.messageEntryId;
        }
      }

      if (st.pendingAgentCheckpoint) {
        const ckpt = st.pendingAgentCheckpoint;
        st.pendingAgentCheckpoint = undefined;

        let startIdx = 0;
        if (st.lastFoldAnchorEntryId) {
          const anchorPos = projEntries.findIndex((e) => e.sourceEntry.id === st.lastFoldAnchorEntryId);
          if (anchorPos >= 0) startIdx = anchorPos + 1;
        }
        const curMsgPos = projEntries.findIndex((e) => e.sourceEntry.id === event.messageEntryId);
        const stopIdx = curMsgPos >= 0 ? curMsgPos : projEntries.length;

        let savedInAgentFold = 0;
        let emittedInAgentFold = 0;
        for (let i = startIdx; i < stopIdx; i++) {
          const pe = projEntries[i];
          const entryId = pe.sourceEntry.id;
          if (!entryId || st.foldedEntryIds.has(entryId)) continue;
          const msgs = pe.messages ?? [];
          const firstRole = msgs[0]?.role ?? pe.sourceEntry.message?.role;
          const firstText = msgs[0] ? extractMessageText(msgs[0]) : "";
          if (firstText.startsWith("[cpat:scope_checkpoint #")) continue;
          if (firstRole !== "assistant" && firstRole !== "toolResult") continue;

          // Ensure raw toolResult is indexed before folding its entry
          if (firstRole === "toolResult" && firstText.length >= 80) {
            st.ingestRaw({
              entryId,
              role: "tool",
              description: `toolResult entry ${entryId}`,
              text: firstText,
              turnIndex: Math.max(0, event.turnIndex - 1),
            });
          }

          st.foldedEntryIds.add(entryId);
          boundaryDrafts.push({
            type: "context_edit",
            targetId: entryId,
            replacement: null,
          });
          st.contextEditsEmitted += 1;
          emittedInAgentFold += 1;
          savedInAgentFold += estimateMessagesTokens(msgs);
        }

        st.lastFoldAnchorEntryId =
          event.toolResultEntryIds?.[event.toolResultEntryIds.length - 1] ?? event.messageEntryId;
        if (emittedInAgentFold > 0) {
          st.boundaryFolds += 1;
          st.boundaryFoldSavedTokens += savedInAgentFold;
          boundarySaved += savedInAgentFold;
        }
        boundaryDrafts.push({
          type: "custom",
          customType: "cpat:scope_fold",
          data: {
            scopeId: ckpt.scopeId,
            mode: ckpt.mode,
            label: ckpt.label,
            editedCount: emittedInAgentFold,
            savedTokens: savedInAgentFold,
          },
        });
      }

      if (boundaryDrafts.length > 0 && resolveStrategy() === "cache-first") {
        ctx.ui?.setStatus?.(
          "cpat",
          `CPAT Scope Fold: -${boundarySaved.toLocaleString()} tok (${st.boundaryFolds} folds, ${st.orderedBlocks.length} blocks indexed)`,
        );
        st.flushMetricsFile(opts.metricsFile);
        return {
          entries: [...event.entries, ...boundaryDrafts],
        };
      }

      // In `cache-first` mode (default): NEVER mutate already-cached prefix bytes
      // during normal turns (`0 -> ~90% window`)! Ride 80%-98% native Prefix Cache
      // at $0.003/M, and let `session_before_compact` perform a single lossless
      // structural + Cache-Forked checkpoint ONLY if the session hits the window ceiling.
      if (resolveStrategy() === "cache-first") {
        st.flushMetricsFile(opts.metricsFile);
        return undefined;
      }

      const activeTokens = estimateMessagesTokens(event.context.contextMessages);
      const envTail = Number(process.env.CPAT_WORKING_TAIL_TOKENS);
      const workingTailBudget =
        Number.isFinite(envTail) && envTail >= 2_000
          ? Math.floor(envTail)
          : Math.min(48_000, Math.max(4_200, Math.floor(effectiveWin * 0.38)));
      const highWindowTrigger = Math.floor(effectiveWin * 0.48);

      const uncompactedWithEntry = st.orderedBlocks.filter(
        (b) => !b.compacted && b.entryId && b.rawTokens >= 260 && !st.foldedEntryIds.has(b.entryId),
      );
      const uncompactedBulkyTokens = uncompactedWithEntry.reduce((s, b) => s + b.rawTokens, 0);

      const burstSingleTurn = activeTokens > Math.floor(effectiveWin * 0.66);
      const eligible = uncompactedWithEntry.filter((b) =>
        burstSingleTurn ? true : b.turnIndex < event.turnIndex,
      );
      if (burstSingleTurn && eligible.length > 1) {
        eligible.pop();
      }

      const singleMonsterThreshold = Math.min(28_000, Math.max(2_200, Math.floor(workingTailBudget * 0.55)));
      const shouldDrain =
        eligible.length > 0 &&
        (activeTokens >= highWindowTrigger ||
          (uncompactedBulkyTokens >= workingTailBudget &&
            (eligible.length >= 2 || eligible[0].rawTokens >= singleMonsterThreshold)));

      if (!shouldDrain) {
        st.flushMetricsFile(opts.metricsFile);
        return boundaryDrafts.length > 0 ? { entries: [...event.entries, ...boundaryDrafts] } : undefined;
      }

      await st.compactTargets(eligible, "digest");

      const newDrafts: Array<PiContextEditDraft | PiCustomEntryDraft> = [...boundaryDrafts];
      let savedThisDrain = boundarySaved;
      for (const item of eligible) {
        if (!item.compacted || !item.entryId || !item.compactText) continue;
        newDrafts.push({
          type: "context_edit",
          targetId: item.entryId,
          replacement: { content: [{ type: "text", text: item.compactText }] },
        });
        st.contextEditsEmitted += 1;
        savedThisDrain += Math.max(0, item.rawTokens - item.activeTokens);
      }

      if (newDrafts.length === 0) {
        st.flushMetricsFile(opts.metricsFile);
        return undefined;
      }

      newDrafts.push({
        type: "custom",
        customType: "cpat:gc",
        data: {
          turnIndex: event.turnIndex,
          editedCount: newDrafts.length,
          savedTokens: savedThisDrain,
          metrics: st.snapshotMetrics(),
        },
      });

      ctx.ui?.setStatus?.(
        "cpat",
        `CPAT: -${savedThisDrain.toLocaleString()} tok (${st.orderedBlocks.length} blocks indexed)`,
      );
      st.flushMetricsFile(opts.metricsFile);

      return {
        entries: [...event.entries, ...newDrafts],
      };
    });

    // 6. High-Watermark Lossless Compaction (`session_before_compact`):
    //    Combines:
    //      (a) Cache-Forked Macro-Narrative Summary (reusing the exact warm `before_provider_request`
    //          wire prefix so even a 384K summary hits ~95%+ KV Cache at $0.003/M, ~$0.001!)
    //      (b) Multi-Epoch Lossless `SearchIndex` Block Directory (`#b_001..#b_N`)
    pi.on("session_before_compact", async (event: PiSessionBeforeCompactEvent, ctx) => {
      const st = getState(ctx);
      const effectiveWin = st.lastEffectiveWindow;
      const strategy = resolveStrategy();
      const { preparation } = event;

      if (
        strategy === "sliding" &&
        event.reason === "threshold" &&
        preparation.tokensBefore < Math.floor(effectiveWin * 0.82)
      ) {
        st.compactionsCancelled += 1;
        st.flushMetricsFile(opts.metricsFile);
        return { cancel: true };
      }

      const allMsgs = [...preparation.messagesToSummarize, ...preparation.turnPrefixMessages];
      for (let i = 0; i < allMsgs.length; i++) {
        const m = allMsgs[i];
        const text = extractMessageText(m);
        if (text.length >= 160) {
          st.ingestRaw({
            role: m.role === "toolResult" ? "tool" : "user",
            description: `archived ${m.role} #${i + 1}`,
            text,
            turnIndex: 0,
          });
        }
      }

      st.epochCompactions += 1;
      const blockDirectory = buildMultiEpochBlockDirectory(st);

      let llmSummary = "";
      let usedCacheFork = false;
      let piUsage: PiUsage | undefined;

      const allowForkSummary =
        strategy === "cache-first" &&
        opts.enableCacheForkSummary !== false &&
        process.env.CPAT_DISABLE_FORK_SUMMARY !== "1" &&
        st.lastWireRequest !== undefined &&
        st.lastWireRequest.messages.length >= 2;

      if (allowForkSummary && st.lastWireRequest) {
        const forkOut = await runCacheForkEpochSummary({
          wire: st.lastWireRequest,
          previousSummary: preparation.previousSummary,
          tailMessages: allMsgs.slice(-2),
          epochNumber: st.epochCompactions,
          signal: event.signal,
          opts,
        });
        st.lastWireRequest = undefined;
        if (forkOut) {
          usedCacheFork = true;
          llmSummary = forkOut.summary;
          st.sidecarLatencyMs += forkOut.latencyMs;
          st.sidecarCachedTokens += forkOut.usage.prompt_cache_hit_tokens ?? 0;
          st.cacheForkSummaries += 1;
          st.cacheForkPromptTokens += forkOut.usage.prompt_tokens;
          st.cacheForkCachedTokens += forkOut.usage.prompt_cache_hit_tokens ?? 0;
          st.cost.recordSidecar(forkOut.usage);
          st.bcUnifiedCost.recordSidecar(forkOut.usage);
          piUsage = toPiUsage(forkOut.usage);
        }
      }

      // Fallback LLM summary path for `sliding` mode or explicit `CPAT_COMPACT_LLM=1`
      const useColdSidecarCompact =
        !usedCacheFork &&
        st.sidecarClient &&
        (strategy === "sliding" || process.env.CPAT_COMPACT_LLM === "1");

      if (useColdSidecarCompact && st.sidecarClient) {
        const serialized = allMsgs
          .map((m) => `[${m.role}]: ${extractMessageText(m).slice(0, 1400)}`)
          .join("\n\n")
          .slice(-28_000);
        const t0 = Date.now();
        try {
          const resp = await st.sidecarClient.chat({
            model: st.sidecarModel,
            messages: [
              {
                role: "system",
                content:
                  "Summarize the completed agent trajectory concisely while preserving EVERY exact file path, line range (Lstart-Lend), identifier, numeric value, and verified conclusion. Do not drop code locations.",
              },
              {
                role: "user",
                content:
                  (preparation.previousSummary ? `Previous summary:\n${preparation.previousSummary}\n\n` : "") +
                  `Trajectory to compact:\n${serialized}`,
              },
            ],
            temperature: 0,
            maxTokens: 1600,
            reasoningEffort: "low",
          });
          st.sidecarLatencyMs += Date.now() - t0;
          st.cost.recordSidecar(resp.usage);
          llmSummary = resp.message.content?.trim() ?? "";
          piUsage = toPiUsage(resp.usage);
        } catch {
          // Fall back to deterministic manifest summary below
        }
      }

      const summary = [
        `## CPAT Cache-First Lossless Checkpoint (Epoch #${st.epochCompactions})`,
        ...(!usedCacheFork && preparation.previousSummary ? [preparation.previousSummary, ""] : []),
        llmSummary
          ? `### Epoch Macro-Narrative & Causal Decision Ledger\n${llmSummary}`
          : `Archived ${st.orderedBlocks.length} raw blocks (~${st.orderedBlocks.reduce((s, b) => s + b.rawTokens, 0).toLocaleString()} tokens) into CPAT's lossless in-memory SearchIndex at the window high-watermark.`,
        "",
        "### Indexed Raw Blocks (100% lossless & searchable via `context_query`)",
        blockDirectory || "- (all prior blocks indexed)",
        "",
        "### ⚡ ACTIVE CONTINUATION PROTOCOL (DO NOT STOP EARLY)",
        "1. Check the kept recent tool result(s) immediately following this checkpoint: if stream/file ingestion is NOT finished yet (e.g. `END_OF_STREAM` has not appeared), immediately continue calling `read_next_batch` / exploration tools until complete.",
        "2. BEFORE answering any question about earlier sessions, arcs, documents, or files archived above, ALWAYS call `context_query` (`query` keywords, `regex`, or `block_ids` + `start_line`/`end_line`) to retrieve exact verbatim names, dates, numbers, and code spans!",
      ].join("\n");

      st.flushMetricsFile(opts.metricsFile);
      return {
        compaction: {
          summary,
          firstKeptEntryId: preparation.firstKeptEntryId,
          tokensBefore: preparation.tokensBefore,
          ...(piUsage ? { usage: piUsage } : {}),
          details: {
            cpatIndexedBlocks: st.orderedBlocks.length,
            epoch: st.epochCompactions,
            usedCacheFork,
            strategy,
          },
        },
      };
    });

    pi.on("agent_settled", (_event: unknown, ctx) => {
      getState(ctx).flushMetricsFile(opts.metricsFile);
    });

    pi.on("session_shutdown", (_event: unknown, ctx) => {
      getState(ctx).flushMetricsFile(opts.metricsFile);
    });
  };
}

function detectExplicitBoundaryIntent(
  promptText: string,
): { mode: "pollution_reset" | "task_pivot"; explicitTag: boolean } | undefined {
  if (!promptText) return undefined;

  if (/\[(?:RESET_CONTEXT|POLLUTION_RESET)\]/i.test(promptText)) {
    return { mode: "pollution_reset", explicitTag: true };
  }
  if (/\[(?:TASK_BOUNDARY|NEW_TASK|FOLD_SCOPE)\]/i.test(promptText)) {
    return { mode: "task_pivot", explicitTag: true };
  }

  const pollutionEn =
    /\b(?:ignore|disregard|forget|discard|abandon|drop)\s+(?:all\s+)?(?:the\s+|our\s+|your\s+)?(?:previous|earlier|prior|above|wrong|legacy|initial)\s+(?:hypothesis|hypotheses|assumption|assumptions|direction|approach|investigation|analysis|reasoning|theory|conclusions?|path)\b/i.test(
      promptText,
    ) ||
    /\b(?:previous|earlier|prior|initial)\s+(?:hypothesis|assumption|direction|approach|reasoning|theory)\s+(?:was|is|turned out(?:\s+to\s+be)?)\s+(?:wrong|incorrect|invalid|false|a\s+dead[\s-]end|obsolete|deprecated|misleading)\b/i.test(
      promptText,
    );
  const pollutionZh =
    /(?:忽略|忘掉|放弃|推翻|丢弃|不要管)(?:刚才|之前|前面|先前|上述)(?:所有)?(?:的)?(?:错误)?(?:假设|方向|思路|推理|分析|结论|路径)/.test(
      promptText,
    ) ||
    /(?:前面|之前|刚才)的(?:假设|方向|思路|推理)(?:完全)?(?:错了|是错的|不对|走偏了|已废弃)/.test(
      promptText,
    );
  if (pollutionEn || pollutionZh) {
    return { mode: "pollution_reset", explicitTag: false };
  }

  const taskPivotEn =
    /^(?:\s*(?:new\s+task|next\s+independent\s+task|switching\s+tasks?|task\s+switch)\s*[:—-])/i.test(
      promptText,
    ) ||
    /\b(?:now\s+switch(?:ing)?\s+to\s+a\s+(?:completely\s+)?new\s+(?:unrelated\s+)?task|previous\s+(?:sub)?task\s+is\s+complete[.,;\s]+now\s+start)\b/i.test(
      promptText,
    );
  const taskPivotZh =
    /(?:切(?:换到)?新(?:需求|任务)|换一个(?:全新的|独立的)?新?(?:需求|任务)|上一个(?:子)?任务已?(?:完结|完成)[，,。;\s]*现在开始)/.test(
      promptText,
    );
  if (taskPivotEn || taskPivotZh) {
    return { mode: "task_pivot", explicitTag: false };
  }

  return undefined;
}

function formatScopeCheckpointBlock(params: {
  scopeId: string;
  mode: "subtask_complete" | "pollution_purge";
  label: string;
  summary: string;
  verifiedFacts: string[];
  purgedAssumptions: string[];
  recentBlocks: IndexedEntryMeta[];
}): string {
  const lines: string[] = [
    `[cpat:scope_checkpoint #${params.scopeId} | mode=${params.mode} | scope="${params.label}"]`,
    `Summary: ${params.summary}`,
  ];
  if (params.verifiedFacts.length > 0) {
    lines.push("Verified Ground-Truth Facts / Landmarks:");
    for (const f of params.verifiedFacts.slice(0, 16)) {
      lines.push(`  - ${f}`);
    }
  }
  if (params.purgedAssumptions.length > 0) {
    lines.push("PURGED DEAD-END ASSUMPTIONS (DO NOT REUSE):");
    for (const p of params.purgedAssumptions.slice(0, 8)) {
      lines.push(`  - [DISCARDED] ${p}`);
    }
  }
  if (params.recentBlocks.length > 0) {
    const ids = params.recentBlocks.map((b) => `#${b.blockId}`).join(", ");
    lines.push(
      `Lossless Raw SearchIndex Blocks Retained: ${ids} (use context_query for exact verbatim lines/facts)`,
    );
  }
  return lines.join("\n");
}

function buildMultiEpochBlockDirectory(st: SessionHarnessState): string {
  const all = st.orderedBlocks;
  if (all.length === 0) return "";
  const recentLimit = 40;
  const lines: string[] = [];

  if (all.length > recentLimit) {
    const older = all.slice(0, all.length - recentLimit);
    const chunkSize = 25;
    for (let i = 0; i < older.length; i += chunkSize) {
      const slice = older.slice(i, i + chunkSize);
      const firstId = slice[0].blockId;
      const lastId = slice[slice.length - 1].blockId;
      const sumTok = slice.reduce((s, b) => s + b.rawTokens, 0);
      const sampleDescs = slice
        .slice(0, 4)
        .map((b) => b.description.slice(0, 48))
        .join("; ");
      lines.push(
        `- Earlier Epoch Range #${firstId}..#${lastId} (${slice.length} blocks, ${sumTok.toLocaleString()} raw tok): ${sampleDescs}${slice.length > 4 ? "; ..." : ""} [100% searchable via context_query]`,
      );
    }
  }

  const recent = all.slice(-recentLimit);
  for (const b of recent) {
    const doc = st.index.getDoc(b.blockId);
    const structuralHeaders = doc
      ? doc.lines
          .filter((l) => /^(?:===\s+|---\s+(?:ARC|SESSION|BATCH)|#{1,3}\s+|Document\s+\d+:)/i.test(l.trim()))
          .slice(0, 8)
          .map((l) => l.trim().slice(0, 95))
      : [];
    const headerNote =
      structuralHeaders.length > 0 ? ` | Sections: ${structuralHeaders.join(" ; ")}` : "";
    const symNote = b.symbolLineSummary ? ` | ${b.symbolLineSummary.slice(0, 140)}` : "";
    lines.push(
      `- #${b.blockId}: ${b.description} (${b.rawTokens.toLocaleString()} raw tok${symNote}${headerNote})`,
    );
  }
  return lines.join("\n");
}

function resolveForkEndpoint(opts: CpatPiOptions): { apiKey: string; baseUrl: string } | undefined {
  const explicitKey =
    opts.cacheForkApiKey ||
    process.env.OPENAI_API_KEY ||
    process.env.DEEPSEEK_API_KEY ||
    process.env.SIDECAR_API_KEY ||
    process.env.API_KEY;
  const rawBase =
    opts.cacheForkBaseUrl ||
    process.env.OPENAI_BASE_URL ||
    process.env.DEEPSEEK_BASE_URL ||
    process.env.SIDECAR_BASE_URL;

  if (explicitKey) {
    return {
      apiKey: explicitKey,
      baseUrl: (rawBase || "https://api.deepseek.com").replace(/\/+$/, ""),
    };
  }
  try {
    const cfg = loadSidecarEnv();
    return { apiKey: cfg.apiKey, baseUrl: cfg.baseUrl.replace(/\/+$/, "") };
  } catch {
    if (opts.cacheForkFetch) {
      return { apiKey: "mock-fork-key", baseUrl: "https://api.deepseek.com" };
    }
    return undefined;
  }
}

async function runCacheForkEpochSummary(params: {
  wire: WireRequestSnapshot;
  previousSummary?: string;
  tailMessages: PiAgentMessage[];
  epochNumber: number;
  signal: AbortSignal;
  opts: CpatPiOptions;
}): Promise<{ summary: string; usage: ChatUsage; latencyMs: number } | undefined> {
  const endpoint = resolveForkEndpoint(params.opts);
  if (!endpoint) return undefined;
  const fetchFn = params.opts.cacheForkFetch ?? globalThis.fetch;
  if (typeof fetchFn !== "function") return undefined;

  const tailDelta = params.tailMessages
    .map((m) => `[${m.role}]: ${extractMessageText(m).slice(0, 1800)}`)
    .join("\n")
    .slice(0, 4200);

  const forkPrompt = [
    `[CPAT CACHE-FORKED EPOCH #${params.epochNumber} CHECKPOINT — DO NOT CALL ANY TOOLS]`,
    "You are generating a concise Macro-Narrative & Causal Decision Ledger for a high-watermark context checkpoint. All raw verbatim lines remain indexed in CPAT's lossless SearchIndex (`context_query`).",
    ...(params.previousSummary
      ? [
          "Carry forward essential unresolved state and verified facts from the Previous Epoch Ledger below so multi-epoch continuity stays unbroken:",
          `<previous_epoch_ledger>\n${params.previousSummary.slice(0, 5000)}\n</previous_epoch_ledger>`,
        ]
      : []),
    ...(tailDelta
      ? [`Latest turn delta right after wire prefix:\n<latest_turn_delta>\n${tailDelta}\n</latest_turn_delta>`]
      : []),
    "Respond in plain Markdown (under 550 tokens, NO tool calls) with:",
    "1. **Goal & Current Execution State**: Exact user objective, current milestone/batch/step reached, and whether stream/file exploration is still in progress.",
    "2. **Causal Decisions & Verified Facts**: Key reasoning steps, architectural decisions, exact file paths, `Lstart-Lend` code spans, identifiers, dates, and numbers established so far.",
    "3. **Falsified Hypotheses & Guardrails**: Any tested dead-ends or constraints that future turns must NOT repeat.",
  ].join("\n\n");

  // Preserve exact provider template fields (`thinking`, `reasoning_effort`, `tools`,
  // `prompt_cache_key`, etc.) from `rawPayload` so chat-template tokenization matches
  // the warm main-session prefix byte-for-byte!
  const {
    stream: _stream,
    stream_options: _streamOpts,
    max_tokens: _maxTok,
    max_completion_tokens: _maxCompTok,
    messages: _origMsgs,
    ...preservedWireFields
  } = params.wire.rawPayload;

  const body: Record<string, unknown> = {
    ...preservedWireFields,
    model: params.wire.model,
    ...(params.wire.tools !== undefined ? { tools: params.wire.tools } : {}),
    messages: [...params.wire.messages, { role: "user", content: forkPrompt }],
    max_tokens: 2200,
    stream: false,
  };

  const t0 = Date.now();
  try {
    const timeoutSig = AbortSignal.timeout(35_000);
    const combinedSignal = params.signal ? AbortSignal.any([params.signal, timeoutSig]) : timeoutSig;
    const res = await fetchFn(`${endpoint.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${endpoint.apiKey}`,
        ...params.wire.affinityHeaders,
      },
      body: JSON.stringify(body),
      signal: combinedSignal,
    });
    if (!res.ok) return undefined;
    const data = (await res.json()) as Record<string, any>;
    const text = String(data?.choices?.[0]?.message?.content ?? "").trim();
    if (!text) return undefined;

    const u = data?.usage ?? {};
    const promptTokens = Number(u.prompt_tokens ?? 0);
    const hitTokens = Number(
      u.prompt_cache_hit_tokens ?? u.prompt_tokens_details?.cached_tokens ?? 0,
    );
    const missTokens = Number(
      u.prompt_cache_miss_tokens ?? Math.max(0, promptTokens - hitTokens),
    );
    const completionTokens = Number(u.completion_tokens ?? 0);
    const reasoningTokens = Number(u.completion_tokens_details?.reasoning_tokens ?? 0);

    return {
      summary: text,
      latencyMs: Date.now() - t0,
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: Number(u.total_tokens ?? promptTokens + completionTokens),
        prompt_cache_hit_tokens: hitTokens,
        prompt_cache_miss_tokens: missTokens,
        reasoning_tokens: reasoningTokens,
      },
    };
  } catch {
    return undefined;
  }
}

function toPiUsage(u: ChatUsage): PiUsage {
  const cached = u.prompt_cache_hit_tokens ?? 0;
  const uncached = u.prompt_cache_miss_tokens ?? Math.max(0, u.prompt_tokens - cached);
  const inputCost = (uncached / 1_000_000) * 0.15;
  const cacheCost = (cached / 1_000_000) * 0.003;
  const outputCost = (u.completion_tokens / 1_000_000) * 0.6;
  return {
    input: uncached,
    output: u.completion_tokens,
    cacheRead: cached,
    cacheWrite: 0,
    reasoning: u.reasoning_tokens ?? 0,
    totalTokens: uncached + cached + u.completion_tokens,
    cost: {
      input: inputCost,
      output: outputCost,
      cacheRead: cacheCost,
      cacheWrite: 0,
      total: inputCost + cacheCost + outputCost,
    },
  };
}

export default createCpatPiExtension();
