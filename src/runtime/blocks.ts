import type {
  ArtifactRef,
  BlockKind,
  ChatMessage,
  ContextBlock,
  EpistemicTier,
  FactDigest,
  Role,
  ToolCall,
  Visibility,
} from "../types.ts";
import { buildLandmark, extractSkeleton } from "../sidecar/skeleton.ts";
import { estimateTokens, nextId, nowIso } from "../util/misc.ts";
import { ContentStore, Journal } from "./stores.ts";

/**
 * The patchable working-state layer. Blocks are addressable by id, ordered by
 * insertion; the context view is this order filtered by visibility.
 */
export class BlockStore {
  private readonly map = new Map<string, ContextBlock>();
  order: string[] = [];
  private readonly content: ContentStore;
  private readonly journal: Journal;

  /**
   * Generational tail bookkeeping (runtime-only state — not part of the
   * ContextBlock schema, the journal, or anything the agent can patch).
   *
   * In generational rendering, the volatile large tool-result blocks are pushed
   * to the TAIL of the rendered view and grouped into generations. Deletion
   * (offload) only happens by retiring whole generations from the oldest end, so
   * the stable prefix never gets a hole punched in it. `genOf` records which
   * generation a block belongs to; `nextGen` hands out a fresh, strictly larger
   * generation id when survivors are promoted past a retired one.
   */
  private readonly genOf = new Map<string, number>();
  private nextGen = 0;

  constructor(content: ContentStore, journal: Journal) {
    this.content = content;
    this.journal = journal;
  }

  get(id: string): ContextBlock | undefined {
    return this.map.get(id);
  }

  all(): ContextBlock[] {
    return this.order
      .map((id) => this.map.get(id))
      .filter((b): b is ContextBlock => b !== undefined);
  }

  visible(): ContextBlock[] {
    return this.all().filter((b) => b.visibility === "model");
  }

  /** Content-store key for the current version of a block. */
  static contentKey(block: ContextBlock): string {
    return `${block.id}@v${block.version}`;
  }

  renderedText(block: ContextBlock): string {
    if (ContentStore.isRef(block.content)) {
      const ref = block.content;
      return (
        `[offloaded payload] ${ref.summary}\n` +
        `(full payload recoverable at ${ref.uri} — ${ref.retrieval_hint})`
      );
    }
    const res = block.active_resolution ?? "full";
    if (res === "full") {
      return block.content;
    }
    const rawTok = block.raw_tokens ?? block.token_count;
    if (res === "digest") {
      const d = block.resolutions?.digest;
      if (d) {
        return formatDigestProjection(block.id, rawTok, d);
      }
      if (block.resolutions?.skeleton) return block.resolutions.skeleton;
    }
    if (res === "skeleton" && block.resolutions?.skeleton) {
      return block.resolutions.skeleton;
    }
    if (res === "landmark" && block.resolutions?.landmark) {
      return block.resolutions.landmark;
    }
    return block.resolutions?.skeleton ?? block.resolutions?.landmark ?? block.content;
  }

  /** Attach an asynchronously computed FactDigest without mutating rendered bytes until resolution flips. */
  attachDigest(blockId: string, digest: FactDigest): void {
    const block = this.map.get(blockId);
    if (!block) return;
    if (!block.resolutions) {
      const text = typeof block.content === "string" ? block.content : "";
      block.resolutions = {
        landmark: buildLandmark(block.id, block.description, block.raw_tokens ?? block.token_count, text.split("\n").length),
      };
    }
    // Preserve byte stability: if already rendering as digest, do not overwrite.
    if (block.active_resolution === "digest" && block.resolutions.digest) return;
    block.resolutions.digest = digest;
  }

  create(opts: {
    idPrefix: string;
    role: Role;
    kind: BlockKind;
    description: string;
    content: string;
    sourceIds?: string[];
    visibility?: Visibility;
    retention?: ContextBlock["retention"];
    epistemic?: EpistemicTier;
    scopeId?: string;
    protected?: boolean;
    api?: ContextBlock["api"];
    insertAfter?: string;
  }): ContextBlock {
    const id = nextId(opts.idPrefix);
    const rawTok = estimateTokens(opts.content);
    const lineCount = opts.content.split("\n").length;
    const epistemic: EpistemicTier =
      opts.epistemic ?? defaultEpistemicTier(opts.kind);
    const shouldPrecomputeSkeleton =
      rawTok >= 220 &&
      (opts.kind === "tool_result" || opts.kind === "assistant_message");

    const block: ContextBlock = {
      id,
      role: opts.role,
      kind: opts.kind,
      description: opts.description,
      content: opts.content,
      source_ids: opts.sourceIds ?? [],
      token_count: rawTok,
      raw_tokens: rawTok,
      visibility: opts.visibility ?? "model",
      retention: opts.retention ?? "session",
      epistemic,
      active_resolution: "full",
      resolutions: {
        landmark: buildLandmark(id, opts.description, rawTok, lineCount),
        ...(shouldPrecomputeSkeleton
          ? { skeleton: extractSkeleton(id, opts.description, opts.content) }
          : {}),
      },
      ...(opts.scopeId ? { scope_id: opts.scopeId } : {}),
      protected: opts.protected,
      version: 1,
      created_at: nowIso(),
      ...(opts.api ? { api: opts.api } : {}),
    };
    this.map.set(id, block);
    if (opts.insertAfter) {
      const at = this.order.indexOf(opts.insertAfter);
      this.order.splice(at === -1 ? this.order.length : at + 1, 0, id);
    } else {
      this.order.push(id);
    }
    const key = this.content.put(BlockStore.contentKey(block), opts.content);
    this.journal.append("ingest", {
      block_id: id,
      role: block.role,
      kind: block.kind,
      content_key: key,
      token_count: block.token_count,
      description: block.description,
    });
    return block;
  }

  // -- ingestion helpers ----------------------------------------------------

  createSystem(content: string): ContextBlock {
    return this.create({
      idPrefix: "system",
      role: "runtime",
      kind: "system_prompt",
      description: "System prompt and tool policy. Never patchable.",
      content,
      visibility: "model",
      retention: "persistent",
      protected: true,
    });
  }

  createUserMessage(content: string): ContextBlock {
    return this.create({
      idPrefix: "user",
      role: "user",
      kind: "user_message",
      description: describeText("User message", content),
      content,
      retention: "persistent",
      protected: true,
    });
  }

  /**
   * Splits an assistant API message into an assistant_message block plus, when
   * thinking mode returned reasoning alongside tool calls, an api_required
   * reasoning_trace block (DeepSeek requires replay while the chain is open).
   */
  createAssistantMessage(msg: ChatMessage, scopeId?: string): {
    block: ContextBlock;
    reasoning?: ContextBlock;
  } {
    const block = this.create({
      idPrefix: "assistant",
      role: "assistant",
      kind: "assistant_message",
      description: describeText(
        msg.tool_calls?.length
          ? `Assistant turn calling ${msg.tool_calls.map((t) => t.function.name).join(", ")}`
          : "Assistant message",
        msg.content ?? "",
      ),
      content: msg.content ?? "",
      scopeId,
      api: msg.tool_calls?.length ? { tool_calls: msg.tool_calls } : undefined,
    });
    let reasoning: ContextBlock;
    if (msg.reasoning_content && msg.tool_calls?.length) {
      reasoning = this.create({
        idPrefix: "reasoning",
        role: "assistant",
        kind: "reasoning_trace",
        description: `Thinking-mode trace for ${block.id}; API-required while its tool-call chain is open.`,
        content: msg.reasoning_content,
        sourceIds: [block.id],
        visibility: "api_required",
        retention: "ephemeral",
        scopeId,
      });
      block.api = { ...block.api, reasoning_content: msg.reasoning_content };
      return { block, reasoning };
    }
    return { block };
  }

  createToolResult(
    toolCall: ToolCall,
    output: string,
    description: string,
    scopeId?: string,
  ): ContextBlock {
    return this.create({
      idPrefix: "tool",
      role: "tool",
      kind: "tool_result",
      description,
      content: output,
      scopeId,
      api: { tool_call_id: toolCall.id, name: toolCall.function.name },
    });
  }

  /**
   * All blocks belonging to the same tool-call chain as `block`. Membership is
   * decided by api.tool_calls / api.tool_call_id, NOT by kind — offloaded
   * blocks still render as role:"tool" messages and stay chain members.
   */
  chainOf(block: ContextBlock): ContextBlock[] {
    const blocks = this.all();
    let head: ContextBlock | undefined;
    if (block.api?.tool_calls?.length) {
      head = block;
    } else if (block.api?.tool_call_id) {
      const idx = blocks.findIndex((b) => b.id === block.id);
      for (let i = idx - 1; i >= 0; i--) {
        const b = blocks[i];
        if (b.api?.tool_calls?.length) {
          const ids = new Set(b.api.tool_calls.map((t) => t.id));
          if (ids.has(block.api.tool_call_id)) head = b;
          break;
        }
      }
    }
    if (!head) return [block];
    const callIds = new Set(head.api!.tool_calls!.map((t) => t.id));
    const chain = [head];
    for (const b of blocks) {
      if (b.api?.tool_call_id && callIds.has(b.api.tool_call_id)) {
        chain.push(b);
      }
    }
    return chain;
  }

  /**
   * Partition the given blocks into ordered, non-overlapping chain groups,
   * preserving insertion order. A group is either a single standalone block or a
   * whole tool-call chain (assistant tool_calls head + its tool_result members).
   *
   * Groups are the atomic unit of generational reordering: DeepSeek requires an
   * assistant tool_calls message to be immediately followed by its tool results,
   * so we may reorder GROUPS relative to each other but never split one.
   */
  chainGroups(blocks: ContextBlock[]): ContextBlock[][] {
    const seen = new Set<string>();
    const groups: ContextBlock[][] = [];
    const inScope = new Set(blocks); // O(1) membership vs O(N) array includes
    for (const b of blocks) {
      if (seen.has(b.id)) continue;
      const chain = this.chainOf(b).filter((c) => inScope.has(c));
      for (const c of chain) seen.add(c.id);
      groups.push(chain);
    }
    return groups;
  }

  /** Expand any set of block ids into complete tool-call chains in block order. */
  expandToChainClosure(ids: string[]): string[] {
    const expanded = new Set<string>();
    for (const id of ids) {
      const b = this.get(id);
      if (!b) continue;
      for (const member of this.chainOf(b)) {
        expanded.add(member.id);
      }
    }
    return this.order.filter((id) => expanded.has(id));
  }

  // -- generational tail bookkeeping -----------------------------------------

  /** Generation id a block currently belongs to, or undefined if untracked. */
  generationOf(id: string): number | undefined {
    return this.genOf.get(id);
  }

  /** Place a block into a specific generation (used when promoting survivors). */
  setGeneration(id: string, gen: number): void {
    this.genOf.set(id, gen);
  }

  /** Allocate the next strictly-larger generation id. */
  allocGeneration(): number {
    return this.nextGen++;
  }
}

function defaultEpistemicTier(kind: BlockKind): EpistemicTier {
  switch (kind) {
    case "system_prompt":
    case "user_message":
    case "task_state":
      return "authoritative";
    case "tool_result":
    case "artifact_ref":
    case "budget_report":
      return "observation";
    case "assistant_message":
    case "reasoning_trace":
    case "summary":
      return "subjective";
  }
}

export function formatDigestProjection(
  blockId: string,
  rawTokens: number,
  d: FactDigest,
): string {
  const lines = [
    `[context:digest #${blockId} | raw ${rawTokens} tok indexed — call context_query(block_ids:["${blockId}"], ...) for exact lines]`,
    `Summary: ${d.summary}`,
  ];
  if (d.verbatim_facts.length > 0) {
    lines.push(`Verbatim facts: ${d.verbatim_facts.join(" | ")}`);
  }
  if (d.ruled_out.length > 0) {
    lines.push(`Ruled out: ${d.ruled_out.join(" | ")}`);
  }
  if (d.query_hint) {
    lines.push(`Query hint: ${d.query_hint}`);
  }
  return lines.join("\n");
}

export function describeText(prefix: string, content: string): string {
  const flat = content.replace(/\s+/g, " ").trim();
  return flat ? `${prefix}: ${flat.slice(0, 110)}${flat.length > 110 ? "…" : ""}` : prefix;
}
