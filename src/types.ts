/**
 * CPAT — Context Patch as Tool.
 *
 * Storage model (single-copy):
 *   ContentStore  — every payload is stored exactly once at ingestion,
 *                   addressed as artifact://<key>. The only recovery channel.
 *   Journal       — append-only event log (ingest / patch / llm_call).
 *                   Metadata + content keys only, never full payload copies.
 *   ContextBlock  — patchable working state, addressable by id.
 *   ContextView   — the block list actually rendered into the next LLM call.
 *
 * payload_offload is therefore a pure view-level operation: the block stops
 * inlining its content and renders an ArtifactRef instead. Zero copy.
 */

export type Role = "user" | "assistant" | "tool" | "runtime";

export type BlockKind =
  | "system_prompt" // runtime extension: system prompt as a block, never patchable
  | "user_message"
  | "assistant_message"
  | "reasoning_trace"
  | "tool_result"
  | "summary"
  | "artifact_ref"
  | "task_state"
  | "budget_report";

/**
 * "model"        — rendered into the next context view.
 * "archived"     — kept in the block store, not rendered; recoverable.
 * "hidden"       — not rendered and not listed in the manifest.
 * "api_required" — runtime-internal: DeepSeek thinking mode requires
 *                  reasoning_content to be replayed while a tool-call chain is
 *                  open. The agent cannot patch api_required blocks; the
 *                  runtime downgrades them to "hidden" once the chain closes.
 */
export type Visibility = "model" | "archived" | "hidden" | "api_required";

export type Retention = "ephemeral" | "session" | "persistent";

export type ArtifactStoreKind = "file" | "object_store" | "sqlite" | "s3";

/**
 * Multi-resolution rendering tiers for ContextHarness (CPAT v2):
 *   "full"     — exact verbatim text from ContentStore (<id>@v<version>).
 *   "skeleton" — deterministic structural outline + regex-extracted identifiers (<2ms, 0 LLM cost).
 *   "digest"   — sidecar-computed FactDigest (verbatim facts + ruled-out leads + query hint).
 *   "landmark" — 1-line byte-stable address handle (~20 tokens).
 */
export type ResolutionLevel = "full" | "skeleton" | "digest" | "landmark";

/**
 * Epistemic authority tiers (inspired by AC3 & Hindsight):
 *   "authoritative" — system prompt & user specifications; never overwritten by subjective turns.
 *   "observation"   — objective tool outputs & environment observations; losslessly indexed.
 *   "subjective"    — assistant reasoning, hypotheses, and plans; eligible for pollution audit.
 */
export type EpistemicTier = "authoritative" | "observation" | "subjective";

export interface FactDigest {
  summary: string;
  verbatim_facts: string[];
  ruled_out: string[];
  query_hint: string;
  generated_by: "sidecar" | "deterministic";
}

export interface BlockResolutions {
  skeleton?: string;
  digest?: FactDigest;
  landmark: string;
}

export interface ScopeNode {
  id: string;
  parent_id?: string;
  label: string;
  goal: string;
  status: "active" | "completed" | "aborted";
  /** Block id immediately preceding this scope's opening turn (for stack-aligned rollback). */
  entry_after_id?: string;
  block_ids: string[];
  child_scope_ids: string[];
  submitted_result?: string;
  key_facts?: string[];
  summary_block_id?: string;
  created_at: string;
}

export interface ArtifactRef {
  artifact_id: string;
  uri: string; // artifact://<id>
  store: ArtifactStoreKind;
  summary: string;
  retrieval_hint: string;
}

export interface ContextBlock {
  id: string;
  role: Role;
  kind: BlockKind;

  description: string;
  content: string | ArtifactRef;

  source_ids: string[];
  token_count: number;
  /** Original full-payload token count before any offload or resolution degradation. */
  raw_tokens?: number;

  visibility: Visibility;
  retention: Retention;

  /** Harness v2 epistemic authority tier. */
  epistemic?: EpistemicTier;
  /** Harness v2 active rendering tier (defaults to "full"). */
  active_resolution?: ResolutionLevel;
  /** Harness v2 pre-computed multi-resolution projections. */
  resolutions?: BlockResolutions;
  /** Owning ScopeNode id if created inside a subtask scope. */
  scope_id?: string;
  /** Pinned high-utility evidence kept inline across routine batch degradations. */
  pinned?: boolean;

  protected?: boolean;
  version: number;
  created_at: string;

  /**
   * Runtime-internal API replay state. Not part of the semantic context view
   * and never exposed to the agent for patching. Keeps DeepSeek's multi-turn
   * tool-call protocol valid regardless of semantic patches:
   *   - tool_call_id / name: for role:"tool" messages.
   *   - tool_calls: original assistant tool_calls payload.
   *   - reasoning_content: thinking-mode trace that must be replayed while the
   *     tool-call chain is open.
   */
  api?: {
    tool_call_id?: string;
    name?: string;
    tool_calls?: ToolCall[];
    reasoning_content?: string;
  };
}

export interface TokenBudget {
  max: number;
  soft_limit: number;
  used: number;
}

export interface ContextView {
  blocks: string[]; // ordered block ids rendered into the next call
  token_budget: TokenBudget;
}

// ---------------------------------------------------------------------------
// Context operations (the payload of the context_update tool)
// ---------------------------------------------------------------------------

export type ContextOperation =
  | {
      op: "compact";
      ids: string[];
      output: {
        id?: string;
        description: string;
        content: string;
      };
      preserve: string[];
      drop: string[];
    }
  | {
      op: "payload_offload";
      ids: string[];
      store: ArtifactStoreKind;
      replace_with: {
        description: string;
        summary: string;
        retrieval_hint: string;
      };
    }
  | {
      op: "redact";
      ids: string[];
      drop_fields: string[];
      preserve_fields?: string[];
    }
  | {
      op: "set_visibility";
      ids: string[];
      visibility: "model" | "archived" | "hidden";
    }
  | {
      op: "replace";
      id: string;
      content: string;
      description?: string;
    }
  | {
      // Inverse of payload_offload: re-inline an offloaded block's full payload
      // from the content store. Zero-copy (the payload already lives at the
      // block's <id>@v<version> key); version is unchanged.
      op: "restore";
      ids: string[];
    }
  | {
      // Consolidate >=2 semantically overlapping blocks into one canonical
      // block, archiving the sources (the supersede chain). resolution records
      // intent: "update" combines, "contradiction" marks the older info as
      // superseded by the newer.
      op: "merge";
      ids: string[];
      output: {
        description: string;
        content: string;
      };
      resolution: "update" | "contradiction";
    }
  | {
      // Fold a contiguous block range (a finished subtask trajectory) into a
      // single scoped summary, archiving the range. Like compact but for a
      // structurally-marked, recoverable subtask branch (unfold via
      // set_visibility=model on the archived sources).
      op: "fold";
      ids: string[];
      output: {
        description: string;
        content: string;
      };
      scope_label: string;
    }
  | {
      // Harness v2: zero-copy multi-resolution switch without changing kind,
      // version, or tool-call chain membership.
      op: "set_resolution";
      ids: string[];
      resolution: ResolutionLevel;
    }
  | {
      // Harness v2: pin or unpin high-utility blocks from routine batch degradation.
      op: "set_pinned";
      ids: string[];
      pinned: boolean;
    }
  | {
      // Harness v2 (AC3 isolated pollution audit): archive tainted subjective
      // blocks (chains closed) while keeping authoritative & objective evidence,
      // optionally inserting a clean authoritative checkpoint note.
      op: "purge_pollution";
      ids: string[];
      clean_checkpoint?: {
        description: string;
        content: string;
      };
    };

export interface ContextUpdateRequest {
  operations: ContextOperation[];
  reason: string;
}

export interface PatchRejection {
  op_index: number;
  rule: string;
  message: string;
}

export interface PatchResult {
  ok: boolean;
  applied: number;
  rejections: PatchRejection[];
  freed_tokens: number;
  created_block_ids?: string[];
  budget: TokenBudget;
}

export type PatchActor = "agent" | "runtime" | "harness";

export type JournalEvent =
  | {
      type: "ingest";
      seq: number;
      ts: string;
      block_id: string;
      role: Role;
      kind: BlockKind;
      content_key: string;
      token_count: number;
      description: string;
    }
  | {
      type: "patch";
      seq: number;
      ts: string;
      actor: PatchActor;
      reason: string;
      operations: ContextOperation[];
      result: PatchResult;
    }
  | {
      type: "llm_call";
      seq: number;
      ts: string;
      model: string;
      usage: ChatUsage;
      visible_blocks: number;
      est_view_tokens: number;
    }
  | {
      type: "sidecar_call";
      seq: number;
      ts: string;
      purpose: "digest" | "decision" | "sub_reader" | "pollution_audit";
      model: string;
      usage: ChatUsage;
      duration_ms: number;
      target_ids?: string[];
    }
  | {
      type: "decision";
      seq: number;
      ts: string;
      trigger: string;
      phase: string;
      committed: boolean;
      reason: string;
      net_benefit_est: number;
      ops_count: number;
    }
  | {
      type: "scope";
      seq: number;
      ts: string;
      action: "open" | "submit" | "abort";
      scope_id: string;
      label: string;
      freed_tokens?: number;
    };

// ---------------------------------------------------------------------------
// Budget report (injected by the runtime as a block, never auto-compacts)
// ---------------------------------------------------------------------------

export interface BudgetReportContent {
  used_tokens: number;
  max_tokens: number;
  soft_limit: number;
  pressure: "soft" | "must_act" | "critical";
  largest_blocks: Array<{
    id: string;
    kind: BlockKind;
    tokens: number;
    suggested_ops: string[];
  }>;
  required_preserve: string[];
}

// ---------------------------------------------------------------------------
// OpenAI-compatible chat types (the subset DeepSeek needs)
// ---------------------------------------------------------------------------

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning_content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    strict?: boolean;
    parameters: Record<string, unknown>;
  };
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  reasoning_tokens?: number;
}

export interface ChatResponse {
  message: ChatMessage;
  finish_reason: string;
  usage: ChatUsage;
}

// ---------------------------------------------------------------------------
// Pricing & True Cost Accounting (Harness v2)
// ---------------------------------------------------------------------------

export interface ModelPricing {
  /** USD per 1M cache-miss input tokens. */
  inputMissPerM: number;
  /** USD per 1M cache-hit input tokens. */
  inputHitPerM: number;
  /** USD per 1M output (completion + reasoning) tokens. */
  outputPerM: number;
}

export interface CostBreakdown {
  main_prompt_tokens: number;
  main_cache_hit_tokens: number;
  main_cache_miss_tokens: number;
  main_completion_tokens: number;
  sidecar_calls: number;
  sidecar_prompt_tokens: number;
  sidecar_completion_tokens: number;
  main_cache_hit_usd: number;
  main_cache_miss_usd: number;
  main_output_usd: number;
  sidecar_total_usd: number;
  /** Total real dollar spend across main (cache-adjusted) + sidecar. */
  effective_cost_usd: number;
  /** Hypothetical main-only cost if zero prefix cache hits occurred. */
  uncached_main_equivalent_usd: number;
  /** Critical-path wall-clock latency added synchronously by the harness (ms). */
  harness_critical_path_ms: number;
}

// ---------------------------------------------------------------------------
// Runtime & Harness configuration
// ---------------------------------------------------------------------------

export interface CpatConfig {
  model: string;
  /** Optional cheap model used by the independent ContextHarness sidecar. */
  sidecarModel?: string;
  /** Working context budget. Keep far below the real 1M window for experiments. */
  maxContextTokens: number;
  /** Fraction of max at which a budget_report block is injected (default 0.7). */
  softLimitRatio: number;
  /** Fraction at which task turns should stay narrow and boundary maintenance is expected. */
  mustActRatio: number;
  /** Fraction at which the runtime force-offloads largest tool results (default 0.95). */
  criticalRatio: number;
  /**
   * Generational tail rendering + batched, net-benefit-gated offload. When on,
   * the view lays bulky tool-result payloads out at the tail in generations and
   * retires whole generations at once, preserving the cached stable prefix.
   * Off → legacy insertion-order rendering + per-block critical fallback.
   */
  generational: boolean;
  /** MVP gates: replace/redact are off by default per the staged plan. */
  allowReplace: boolean;
  allowRedact: boolean;
  /** Ask DeepSeek for strict JSON-schema tool calls; falls back if rejected. */
  strictTools: boolean;
  maxTurns: number;
  runDir: string;
  verbose: boolean;
}

export type HarnessMode = "cooperative" | "query_only" | "transparent";

export interface HarnessConfig {
  maxContextTokens: number;
  /**
   * Target steady-state raw tail budget before batch-degrading older digested
   * blocks. Defaults to min(14_000, Math.floor(maxContextTokens * 0.55)) so
   * even 200K windows stay lean (~15K–25K active) while micro-batching 2–3
   * reads for 100% intra-batch prefix cache hits.
   */
  workingTailTokens?: number;
  softLimitRatio?: number;
  mustActRatio?: number;
  criticalRatio?: number;
  /** Minimum block tokens worth digesting / degrading (default 320). */
  minDigestTokens?: number;
  /** Number of most-recent tool chains kept at "full" resolution (default 1). */
  keepRecentFullChains?: number;
  /** Expected remaining turns horizon for EconomicCommitGate (default 8). */
  expectedHorizonTurns?: number;
  mode?: HarnessMode;
  sidecarModel?: string;
  mainPricing?: ModelPricing;
  sidecarPricing?: ModelPricing;
  runDir: string;
  verbose?: boolean;
}
