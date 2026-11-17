import type { ChatClient } from "../deepseek/client.ts";
import type { CostTracker } from "../harness/cost.ts";
import type { BlockStore } from "../runtime/blocks.ts";
import type { Journal } from "../runtime/stores.ts";
import type { ContextBlock, FactDigest } from "../types.ts";

const SINGLE_DIGEST_SYSTEM_PROMPT = `You are a precision context-digestion sidecar for an autonomous research and coding agent.
Extract a high-density, zero-hallucination JSON digest of the given tool block so the raw block can be safely offloaded from active working memory.

STRICT RULES:
1. "verbatim_facts": Copy EVERY exact identifier, codename, person/lead name, date/timestamp, numeric metric/percentage, incident/trace/error/commit code (e.g. INC-..., TRC-..., SHA-...), dependency target, file:line path, and formula CHARACTER-FOR-CHARACTER. Never round numbers or paraphrase IDs.
2. "ruled_out": List any checked paths, hypotheses, or components proven absent, staging-only, or irrelevant (empty array if none).
3. "summary": 1-2 tight sentences stating what resource was inspected and its core outcome.
4. "query_hint": 1 short clause stating what granular lines remain in the raw indexed block.
Return ONLY valid JSON matching:
{"summary":string,"verbatim_facts":string[],"ruled_out":string[],"query_hint":string}`;

const BATCH_DIGEST_SYSTEM_PROMPT = `You are a precision batch context-digestion sidecar for an autonomous agent.
You will receive 2..4 raw tool blocks. For EACH block_id, extract a high-density, zero-hallucination JSON digest.

STRICT RULES:
1. "verbatim_facts": Copy EVERY exact identifier, codename, person/lead name, date/timestamp, numeric metric/percentage, incident/trace/error/commit code, dependency target, file:line path, and formula CHARACTER-FOR-CHARACTER.
2. "ruled_out": List any checked paths, hypotheses, or components proven absent or irrelevant (empty array if none).
3. "summary": 1-2 tight sentences stating what resource was inspected and its outcome.
4. "query_hint": 1 short clause stating what granular lines remain in the raw indexed block.
Return ONLY valid JSON matching:
{"digests":{"<block_id>":{"summary":string,"verbatim_facts":string[],"ruled_out":string[],"query_hint":string}}}`;

export class ShadowDigester {
  private readonly client?: ChatClient;
  private readonly model: string;
  private readonly blocks: BlockStore;
  private readonly journal: Journal;
  private readonly cost: CostTracker;
  private readonly minTokens: number;
  private readonly maxConcurrent: number;

  private activeCount = 0;
  private readonly queue: Array<() => void> = [];
  private readonly inFlight = new Map<string, Promise<void>>();

  constructor(opts: {
    client?: ChatClient;
    model: string;
    blocks: BlockStore;
    journal: Journal;
    cost: CostTracker;
    minTokens?: number;
    maxConcurrent?: number;
  }) {
    this.client = opts.client;
    this.model = opts.model;
    this.blocks = opts.blocks;
    this.journal = opts.journal;
    this.cost = opts.cost;
    this.minTokens = opts.minTokens ?? 450;
    this.maxConcurrent = opts.maxConcurrent ?? 3;
  }

  /** Optional eager single-block enqueue (kept for explicit background pre-warming). */
  enqueue(block: ContextBlock, taskHint?: string): void {
    if (!this.client) return;
    if ((block.raw_tokens ?? block.token_count) < this.minTokens) return;
    if (block.kind !== "tool_result") return;
    if (typeof block.content !== "string" || !block.content.trim()) return;
    if (block.resolutions?.digest || this.inFlight.has(block.id)) return;

    const p = this.runBatchCall([block], taskHint);
    this.inFlight.set(block.id, p);
  }

  /**
   * Lazy Just-in-Time Batched Digestion (v2.1):
   *
   * Packs multiple blocks that are ACTUALLY being degraded right now into
   * multi-block sidecar requests (up to 4 blocks / ~28K chars per call),
   * eliminating wasted calls on tail blocks that never get compressed and
   * slashing system-prompt/API round-trip overhead by 3x-4x!
   *
   * Bounded by `timeoutMs`: if a massive burst exceeds `timeoutMs`, returns
   * immediately so the caller can zero-wait degrade remaining blocks to
   * deterministic `skeleton` while the background promise finishes cleanly.
   */
  async digestBatch(
    targets: ContextBlock[],
    taskHint?: string,
    timeoutMs = 3800,
  ): Promise<number> {
    if (!this.client || targets.length === 0) return 0;

    const todo: ContextBlock[] = [];
    const waitList: Promise<void>[] = [];

    for (const b of targets) {
      if (b.resolutions?.digest) continue;
      const existing = this.inFlight.get(b.id);
      if (existing) {
        waitList.push(existing);
        continue;
      }
      if (
        b.kind === "tool_result" &&
        typeof b.content === "string" &&
        b.content.trim().length > 0 &&
        (b.raw_tokens ?? b.token_count) >= this.minTokens
      ) {
        todo.push(b);
      }
    }

    // Partition `todo` into multi-block chunks (<= 4 blocks and <= 56,000 chars per chunk)
    const chunks: ContextBlock[][] = [];
    let cur: ContextBlock[] = [];
    let curChars = 0;
    for (const b of todo) {
      const len = Math.min(String(b.content).length, 28_000);
      if (cur.length >= 4 || (cur.length > 0 && curChars + len > 56_000)) {
        chunks.push(cur);
        cur = [];
        curChars = 0;
      }
      cur.push(b);
      curChars += len;
    }
    if (cur.length > 0) chunks.push(cur);

    for (const chunk of chunks) {
      const p = this.runBatchCall(chunk, taskHint);
      for (const b of chunk) this.inFlight.set(b.id, p);
      waitList.push(p);
    }

    if (waitList.length === 0) return 0;

    const t0 = Date.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        Promise.allSettled(waitList),
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, timeoutMs);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
    return Date.now() - t0;
  }

  async awaitPending(blockIds: string[], timeoutMs = 2800): Promise<number> {
    const pending = blockIds
      .map((id) => this.inFlight.get(id))
      .filter((p): p is Promise<void> => p !== undefined);
    if (pending.length === 0) return 0;

    const t0 = Date.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        Promise.allSettled(pending),
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, timeoutMs);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
    return Date.now() - t0;
  }

  private runBatchCall(chunk: ContextBlock[], taskHint?: string): Promise<void> {
    return this.withConcurrencySlot(async () => {
      const t0 = Date.now();
      const ids = chunk.map((b) => b.id);
      try {
        if (chunk.length === 1) {
          const b = chunk[0];
          const payloadSlice = sliceForDigest(String(b.content), 52_000);
          const userPrompt =
            (taskHint ? `Active task context: ${taskHint.slice(0, 450)}\n\n` : "") +
            `Block ID: ${b.id}\nDescription: ${b.description}\n--- RAW PAYLOAD ---\n${payloadSlice}`;
          const resp = await this.client!.chat({
            model: this.model,
            messages: [
              { role: "system", content: SINGLE_DIGEST_SYSTEM_PROMPT },
              { role: "user", content: userPrompt },
            ],
            temperature: 0,
            maxTokens: 1200,
            reasoningEffort: "low",
            responseFormat: { type: "json_object" },
          });
          const ms = Date.now() - t0;
          this.cost.recordSidecar(resp.usage);
          this.journal.append("sidecar_call", {
            purpose: "digest",
            model: this.model,
            usage: resp.usage,
            duration_ms: ms,
            target_ids: ids,
          });
          const parsed = parseDigestJson(resp.message.content ?? "");
          if (parsed) this.blocks.attachDigest(b.id, parsed);
          return;
        }

        // Multi-block batched call (2..4 blocks in ONE request)
        const perBlockMax = Math.floor(56_000 / chunk.length);
        const sections = chunk
          .map(
            (b) =>
              `=== BLOCK ${b.id} (${b.description}) ===\n${sliceForDigest(String(b.content), perBlockMax)}`,
          )
          .join("\n\n");
        const userPrompt =
          (taskHint ? `Active task context: ${taskHint.slice(0, 450)}\n\n` : "") +
          `Digest all ${chunk.length} blocks (${ids.join(", ")}):\n\n${sections}`;

        const resp = await this.client!.chat({
          model: this.model,
          messages: [
            { role: "system", content: BATCH_DIGEST_SYSTEM_PROMPT },
            { role: "user", content: userPrompt },
          ],
          temperature: 0,
          maxTokens: Math.min(2800, 650 * chunk.length),
          reasoningEffort: "low",
          responseFormat: { type: "json_object" },
        });
        const ms = Date.now() - t0;
        this.cost.recordSidecar(resp.usage);
        this.journal.append("sidecar_call", {
          purpose: "digest",
          model: this.model,
          usage: resp.usage,
          duration_ms: ms,
          target_ids: ids,
        });
        const batchParsed = parseBatchDigestJson(resp.message.content ?? "", ids);
        for (const [id, digest] of batchParsed) {
          this.blocks.attachDigest(id, digest);
        }
      } catch {
        // Resilient fallback: deterministic `skeleton` + `SearchIndex` stay intact.
      } finally {
        for (const id of ids) this.inFlight.delete(id);
      }
    });
  }

  private async withConcurrencySlot<T>(fn: () => Promise<T>): Promise<T> {
    if (this.activeCount >= this.maxConcurrent) {
      await new Promise<void>((resolve) => this.queue.push(resolve));
    }
    this.activeCount += 1;
    try {
      return await fn();
    } finally {
      this.activeCount -= 1;
      const next = this.queue.shift();
      if (next) next();
    }
  }
}

function sliceForDigest(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const headChars = Math.floor(maxChars * 0.55);
  const tailChars = Math.floor(maxChars * 0.35);
  return (
    text.slice(0, headChars) +
    `\n\n...[middle ${text.length - headChars - tailChars} chars omitted in sidecar preview; head & tail preserved]...\n\n` +
    text.slice(text.length - tailChars)
  );
}

function toDigestObj(obj: Record<string, unknown>): FactDigest | undefined {
  const summary = typeof obj.summary === "string" ? obj.summary.trim() : "";
  if (!summary) return undefined;
  const toStrArr = (v: unknown): string[] =>
    Array.isArray(v)
      ? v.map((x) => String(x).trim()).filter(Boolean).slice(0, 24)
      : [];
  return {
    summary,
    verbatim_facts: toStrArr(obj.verbatim_facts),
    ruled_out: toStrArr(obj.ruled_out),
    query_hint: typeof obj.query_hint === "string" ? obj.query_hint.trim() : "",
    generated_by: "sidecar",
  };
}

export function parseDigestJson(raw: string): FactDigest | undefined {
  const cleaned = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  if (!cleaned) return undefined;
  try {
    const obj = JSON.parse(cleaned) as Record<string, unknown>;
    return toDigestObj(obj);
  } catch {
    return undefined;
  }
}

export function parseBatchDigestJson(
  raw: string,
  expectedIds: string[],
): Map<string, FactDigest> {
  const out = new Map<string, FactDigest>();
  const cleaned = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  if (!cleaned) return out;
  try {
    const root = JSON.parse(cleaned) as Record<string, unknown>;
    const mapCandidate =
      typeof root.digests === "object" && root.digests !== null
        ? (root.digests as Record<string, unknown>)
        : root;
    for (const id of expectedIds) {
      const entry = mapCandidate[id] ?? mapCandidate[`#${id}`];
      if (typeof entry === "object" && entry !== null) {
        const d = toDigestObj(entry as Record<string, unknown>);
        if (d) out.set(id, d);
      }
    }
  } catch {
    // ignore malformed json
  }
  return out;
}
