/**
 * Zero-dependency lossless SearchIndex over ingested ContextBlocks.
 *
 * Prevents the "verbatim needle recall cliff" (MemoryAgentBench 69% -> 27% on
 * pure lossy summarization) by indexing every block's numbered lines at ingest
 * time with:
 *   1. Exact substring / regex line grep (`grep`)
 *   2. Direct line-range projection (`sliceLines`)
 *   3. Windowed BM25 + exact-code boost (`searchSnippets`)
 *
 * Works across ALL stored blocks regardless of current active_resolution or
 * visibility (archived / degraded blocks remain 100% searchable).
 */

export interface IndexedDoc {
  blockId: string;
  kind: string;
  description: string;
  lines: string[];
  baseLineOffset: number; // 1-based starting line number in source file (default 1)
  totalChars: number;
  windows: WindowEntry[];
}

interface WindowEntry {
  startLine: number; // 1-based inclusive
  endLine: number;   // 1-based inclusive
  tf: Map<string, number>;
  len: number;
  rawLower: string;
}

export interface SnippetHit {
  blockId: string;
  description: string;
  startLine: number;
  endLine: number;
  score: number;
  text: string;
}

const WINDOW_LINES = 6;
const WINDOW_STRIDE = 3;
const K1 = 1.5;
const B = 0.75;

const QUERY_STOPWORDS = new Set([
  "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
  "was", "were", "are", "been", "being", "what", "when", "where", "which",
  "who", "whom", "whose", "why", "how", "did", "does", "done", "doing", "can",
  "could", "would", "should", "will", "shall", "may", "might", "must", "about",
  "into", "over", "after", "before", "between", "under", "again", "further",
  "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
  "most", "other", "some", "such", "only", "own", "same", "than", "too", "very",
  "just", "also", "tell", "user", "assistant", "mention", "mentioned", "said",
  "say", "says", "asked", "ask", "please", "give", "report", "exact", "value",
]);

/**
 * Tokenize text preserving exact hyphenated/underscored codes (e.g. inc-7q42,
 * trc-0209, local_ring_buffer, 73.4%) alongside individual sub-tokens and CJK
 * unigrams/bigrams.
 */
export function tokenizeForIndex(text: string): string[] {
  const lower = text.toLowerCase();
  const tokens: string[] = [];

  // 1. Compound identifiers / numbers / codes (e.g. inc-7q42, 73.4%, v2.1, foo_bar)
  const compoundRe = /[a-z0-9]+(?:[-_.:/][a-z0-9]+)+%?|[a-z0-9]+%?/g;
  let m: RegExpExecArray | null;
  while ((m = compoundRe.exec(lower)) !== null) {
    const tok = m[0];
    if (tok.length >= 2 || /\d/.test(tok)) {
      tokens.push(tok);
      if (/[-_.:/%]/.test(tok)) {
        for (const part of tok.split(/[-_.:/%]+/)) {
          if (part.length >= 2 || /\d/.test(part)) tokens.push(part);
        }
      }
    }
  }

  // 2. CJK characters & bigrams
  const cjkRuns = lower.match(/[\u4e00-\u9fff]{1,}/g);
  if (cjkRuns) {
    for (const run of cjkRuns) {
      for (let i = 0; i < run.length; i++) {
        tokens.push(run[i]);
        if (i + 1 < run.length) tokens.push(run.slice(i, i + 2));
      }
    }
  }

  return tokens;
}

export class SearchIndex {
  private readonly docs = new Map<string, IndexedDoc>();
  private readonly df = new Map<string, number>();
  private totalWindows = 0;
  private totalWindowLen = 0;

  upsert(
    blockId: string,
    kind: string,
    description: string,
    content: string,
    baseLineOffset = 1,
  ): void {
    if (this.docs.has(blockId)) {
      this.remove(blockId);
    }
    const base = Math.max(1, Math.floor(baseLineOffset));
    const lines = content.split("\n");
    const windows: WindowEntry[] = [];

    for (let start = 0; start < lines.length; start += WINDOW_STRIDE) {
      const end = Math.min(lines.length, start + WINDOW_LINES);
      const slice = lines.slice(start, end).join("\n");
      const toks = tokenizeForIndex(`${description}\n${slice}`);
      const tf = new Map<string, number>();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      windows.push({
        startLine: base + start,
        endLine: base + Math.max(0, end - 1),
        tf,
        len: Math.max(1, toks.length),
        rawLower: slice.toLowerCase(),
      });
      for (const term of tf.keys()) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
      this.totalWindows += 1;
      this.totalWindowLen += Math.max(1, toks.length);
      if (end === lines.length) break;
    }

    this.docs.set(blockId, {
      blockId,
      kind,
      description,
      lines,
      baseLineOffset: base,
      totalChars: content.length,
      windows,
    });
  }

  remove(blockId: string): void {
    const prev = this.docs.get(blockId);
    if (!prev) return;
    for (const w of prev.windows) {
      for (const term of w.tf.keys()) {
        const c = (this.df.get(term) ?? 1) - 1;
        if (c <= 0) this.df.delete(term);
        else this.df.set(term, c);
      }
      this.totalWindows -= 1;
      this.totalWindowLen -= w.len;
    }
    this.docs.delete(blockId);
  }

  has(blockId: string): boolean {
    return this.docs.has(blockId);
  }

  getDoc(blockId: string): IndexedDoc | undefined {
    return this.docs.get(blockId);
  }

  sliceLines(blockId: string, startLine = 1, endLine = 80): SnippetHit | undefined {
    const doc = this.docs.get(blockId);
    if (!doc) return undefined;
    const base = doc.baseLineOffset;
    const maxFileLine = base + Math.max(0, doc.lines.length - 1);
    // Support both file-offset line numbers (e.g. L120..L160 when base=120)
    // and 1-based block-relative line numbers (e.g. 1..40 when base=120).
    let reqStart = startLine;
    let reqEnd = endLine;
    if (base > 1 && reqEnd < base) {
      reqStart = base + Math.max(0, reqStart - 1);
      reqEnd = base + Math.max(0, reqEnd - 1);
    }
    const sFile = Math.max(base, Math.min(maxFileLine, reqStart));
    const eFile = Math.max(sFile, Math.min(maxFileLine, reqEnd));
    const sIdx = sFile - base;
    const eIdx = eFile - base + 1;
    const formatted = doc.lines
      .slice(sIdx, eIdx)
      .map((line, idx) => `L${sFile + idx}: ${line}`)
      .join("\n");
    return {
      blockId,
      description: doc.description,
      startLine: sFile,
      endLine: eFile,
      score: 1,
      text: formatted,
    };
  }

  grep(
    pattern: string,
    opts: { blockIds?: string[]; contextLines?: number; maxHits?: number } = {},
  ): SnippetHit[] {
    const ctx = opts.contextLines ?? 2;
    const maxHits = opts.maxHits ?? 12;
    const allow = opts.blockIds?.length ? new Set(opts.blockIds) : undefined;

    let regex: RegExp;
    try {
      regex = new RegExp(pattern, "i");
    } catch {
      const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      regex = new RegExp(escaped, "i");
    }

    const hits: SnippetHit[] = [];
    for (const doc of this.docs.values()) {
      if (allow && !allow.has(doc.blockId)) continue;
      const base = doc.baseLineOffset;
      const matchedRanges: Array<[number, number]> = [];
      for (let i = 0; i < doc.lines.length; i++) {
        if (regex.test(doc.lines[i])) {
          const s = Math.max(0, i - ctx);
          const e = Math.min(doc.lines.length - 1, i + ctx);
          const prev = matchedRanges[matchedRanges.length - 1];
          // Merge adjacent matches only up to 14 lines so repetitive logs never
          // coalesce an entire 200-line file into one multi-thousand-token range!
          if (prev && s <= prev[1] + 1 && Math.max(prev[1], e) - prev[0] < 14) {
            prev[1] = Math.max(prev[1], e);
          } else {
            matchedRanges.push([s, e]);
          }
        }
      }
      // If a repetitive log produced many ranges, keep the first 3 and last 2
      // ranges of this block so both headers and footers survive concisely.
      const selectedRanges =
        matchedRanges.length > 5
          ? [...matchedRanges.slice(0, 3), ...matchedRanges.slice(-2)]
          : matchedRanges;
      for (const [sIdx, eIdx] of selectedRanges) {
        const sFile = base + sIdx;
        const eFile = base + eIdx;
        const text = doc.lines
          .slice(sIdx, eIdx + 1)
          .map((l, idx) => `L${sFile + idx}: ${l}`)
          .join("\n");
        hits.push({
          blockId: doc.blockId,
          description: doc.description,
          startLine: sFile,
          endLine: eFile,
          score: 10,
          text,
        });
        if (hits.length >= maxHits) return hits;
      }
    }
    return hits;
  }

  searchSnippets(
    query: string,
    opts: { blockIds?: string[]; topK?: number } = {},
  ): SnippetHit[] {
    const topK = opts.topK ?? 8;
    const allow = opts.blockIds?.length ? new Set(opts.blockIds) : undefined;
    const rawTokens = Array.from(new Set(tokenizeForIndex(query)));
    if (rawTokens.length === 0 || this.totalWindows === 0) return [];
    const filteredTokens = rawTokens.filter((t) => !QUERY_STOPWORDS.has(t));
    const qTokens = filteredTokens.length > 0 ? filteredTokens : rawTokens;

    const avgLen = this.totalWindowLen / Math.max(1, this.totalWindows);
    const rawPhrases = query
      .toLowerCase()
      .split(/\s+/)
      .filter((p) => p.length >= 3 && !QUERY_STOPWORDS.has(p));

    const scored: Array<{ doc: IndexedDoc; win: WindowEntry; score: number }> = [];

    for (const doc of this.docs.values()) {
      if (allow && !allow.has(doc.blockId)) continue;
      for (const win of doc.windows) {
        let score = 0;
        for (const term of qTokens) {
          const f = win.tf.get(term) ?? 0;
          if (f === 0) continue;
          const n = this.df.get(term) ?? 0;
          const idf = Math.log(1 + (this.totalWindows - n + 0.5) / (n + 0.5));
          const tfNorm = (f * (K1 + 1)) / (f + K1 * (1 - B + B * (win.len / avgLen)));
          // Boost rare compound identifiers / codes (e.g. trc-0209, inc-7q42)
          const boost = /[-_.:/]|\d/.test(term) ? 1.8 : 1.0;
          score += idf * tfNorm * boost;
        }
        // Exact substring bonus for multi-char query tokens inside the window
        for (const p of rawPhrases) {
          if (win.rawLower.includes(p)) score += 1.5;
        }
        if (score > 0) scored.push({ doc, win, score });
      }
    }

    scored.sort((a, b) => b.score - a.score);

    // Deduplicate overlapping windows within the same block, merging adjacent lines.
    const selected: SnippetHit[] = [];
    for (const cand of scored) {
      if (selected.length >= topK) break;
      const overlap = selected.find(
        (s) =>
          s.blockId === cand.doc.blockId &&
          cand.win.startLine <= s.endLine + 1 &&
          cand.win.endLine >= s.startLine - 1,
      );
      if (overlap) {
        overlap.startLine = Math.min(overlap.startLine, cand.win.startLine);
        overlap.endLine = Math.max(overlap.endLine, cand.win.endLine);
        overlap.score = Math.max(overlap.score, cand.score);
        const sIdx = overlap.startLine - cand.doc.baseLineOffset;
        const eIdx = overlap.endLine - cand.doc.baseLineOffset + 1;
        overlap.text = cand.doc.lines
          .slice(sIdx, eIdx)
          .map((l, idx) => `L${overlap.startLine + idx}: ${l}`)
          .join("\n");
        continue;
      }
      const { startLine, endLine } = cand.win;
      const sIdx = startLine - cand.doc.baseLineOffset;
      const eIdx = endLine - cand.doc.baseLineOffset + 1;
      selected.push({
        blockId: cand.doc.blockId,
        description: cand.doc.description,
        startLine,
        endLine,
        score: Number(cand.score.toFixed(3)),
        text: cand.doc.lines
          .slice(sIdx, eIdx)
          .map((l, idx) => `L${startLine + idx}: ${l}`)
          .join("\n"),
      });
    }

    return selected;
  }
}
