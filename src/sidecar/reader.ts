import type { ChatClient } from "../deepseek/client.ts";
import type { CostTracker } from "../harness/cost.ts";
import type { SearchIndex, SnippetHit } from "../runtime/index.ts";
import type { Journal } from "../runtime/stores.ts";

export interface ContextQueryArgs {
  query: string;
  mode?: "snippets" | "grep" | "ask" | "raw_range";
  block_ids?: string[];
  start_line?: number;
  end_line?: number;
}

const SUB_READER_SYSTEM_PROMPT = `You are a lossless evidence extraction sub-reader over an agent's indexed session archive.
Answer the query strictly from the retrieved raw snippets below.
Quote EVERY exact code, number, percentage, name, parameter, and formula verbatim, and cite [block_id:Lstart-Lend] for each fact. If the snippets do not contain the answer, say so explicitly.`;

export class SubReader {
  private readonly index: SearchIndex;
  private readonly client?: ChatClient;
  private readonly model: string;
  private readonly journal: Journal;
  private readonly cost: CostTracker;

  constructor(opts: {
    index: SearchIndex;
    client?: ChatClient;
    model: string;
    journal: Journal;
    cost: CostTracker;
  }) {
    this.index = opts.index;
    this.client = opts.client;
    this.model = opts.model;
    this.journal = opts.journal;
    this.cost = opts.cost;
  }

  async query(args: ContextQueryArgs): Promise<string> {
    const mode = args.mode ?? "snippets";
    const blockIds = args.block_ids?.map((id) => id.replace(/^#/, "").trim()).filter(Boolean);
    const q = (args.query ?? "").trim();

    if (mode === "raw_range") {
      const targetId = blockIds?.[0];
      if (!targetId) {
        return `error: mode="raw_range" requires block_ids: ["<block_id>"]`;
      }
      const slice = this.index.sliceLines(
        targetId,
        args.start_line ?? 1,
        args.end_line ?? (args.start_line ? args.start_line + 60 : 80),
      );
      if (!slice) return `No indexed block found with id "${targetId}".`;
      return formatHits([slice]);
    }

    if (!q) {
      return `error: query must be non-empty`;
    }

    if (mode === "grep") {
      let hits = this.index.grep(q, { blockIds, contextLines: 2, maxHits: 10 });
      if (hits.length === 0) {
        hits = this.index.searchSnippets(q, { blockIds, topK: 6 });
      }
      return hits.length > 0
        ? formatHits(hits)
        : `No matches found in session archive for "${q}".`;
    }

    // Combine exact identifier grep (for multi-code queries like "TRC-0209 TRC-0511")
    // with windowed BM25 retrieval so exact needles always rank first.
    const codeTokens = q.match(/\b[A-Za-z]{2,8}-\d[\w-]*\b/g) ?? [];
    const grepHits =
      codeTokens.length > 0
        ? this.index.grep(codeTokens.map(escapeRegex).join("|"), {
            blockIds,
            contextLines: 2,
            maxHits: 8,
          })
        : [];
    const bm25Hits = this.index.searchSnippets(q, { blockIds, topK: 8 });
    const merged = mergeHits(grepHits, bm25Hits, 10);

    if (merged.length === 0) {
      return `No matching evidence found in session archive for "${q}".`;
    }

    if (mode === "ask" && this.client) {
      const t0 = Date.now();
      const formatted = formatHits(merged);
      try {
        const resp = await this.client.chat({
          model: this.model,
          messages: [
            { role: "system", content: SUB_READER_SYSTEM_PROMPT },
            {
              role: "user",
              content: `Query: ${q}\n\n--- RETRIEVED ARCHIVE SNIPPETS ---\n${formatted}`,
            },
          ],
          temperature: 0,
          maxTokens: 500,
          reasoningEffort: "low",
        });
        const ms = Date.now() - t0;
        this.cost.recordSidecar(resp.usage);
        this.cost.addCriticalPathMs(ms);
        this.journal.append("sidecar_call", {
          purpose: "sub_reader",
          model: this.model,
          usage: resp.usage,
          duration_ms: ms,
          target_ids: Array.from(new Set(merged.map((h) => h.blockId))),
        });
        if (resp.message.content?.trim()) {
          return resp.message.content.trim();
        }
      } catch {
        // Fall back to deterministic verbatim snippets below.
      }
    }

    return formatHits(merged);
  }
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function mergeHits(a: SnippetHit[], b: SnippetHit[], limit: number): SnippetHit[] {
  const out: SnippetHit[] = [];
  for (const h of [...a, ...b]) {
    if (out.length >= limit) break;
    const dup = out.some(
      (x) =>
        x.blockId === h.blockId &&
        h.startLine >= x.startLine - 1 &&
        h.endLine <= x.endLine + 1,
    );
    if (!dup) out.push(h);
  }
  return out;
}

function formatHits(hits: SnippetHit[], maxChars = 3400): string {
  const blocks = hits.map(
    (h) =>
      `[#${h.blockId}:L${h.startLine}-L${h.endLine} | ${h.description}]\n${h.text}`,
  );
  const joined = blocks.join("\n\n");
  if (joined.length <= maxChars) return joined;
  return (
    joined.slice(0, maxChars) +
    `\n…[snippet output capped at ${maxChars} chars; narrow query or use mode="raw_range" with start_line/end_line for specific lines]`
  );
}
