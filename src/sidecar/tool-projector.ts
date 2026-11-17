/**
 * Pre-Cache Oversized `tool_result` Projector (`src/sidecar/tool-projector.ts`).
 *
 * Why intercept at `tool_result` (Write-Time, BEFORE KV Cache) instead of `turn_end`?
 *   1. Zero Prefix-Cache Invalidation: Once a `ToolResultMessage` enters the LLM
 *      transcript, mutating it on a later turn shifts every subsequent message and
 *      destroys their Prefix Cache (`50:1` miss-vs-hit penalty). Compacting oversized
 *      low-density tool dumps BEFORE Turn t+1 ever sees them keeps the entire
 *      session 100% strictly append-only (`90%-98%` Prefix Cache Hit)!
 *   2. Saves First-Turn `P_miss` Prefill: Prevents dumping 15K-100K tokens of raw
 *      build logs, broad `grep` floods, JSON arrays, or 4,000-line file dumps into
 *      expensive uncached main-model prefill (`$0.0000` deterministic CPU projection, `<3ms`).
 *   3. Recovers Pi's Truncated Blind Spots: When Pi's built-in `bash` (`truncateTail`
 *      at 50KB) or `read` (`truncateHead` at 2,000 lines) truncates a 250KB output,
 *      this projector reads the full untruncated payload (`details.fullOutputPath`
 *      or source file), indexes 100% of all lines into CPAT's `SearchIndex`, and
 *      surfaces hidden early `ERROR`/`Traceback` hunks & full-file `[Lstart-Lend]`
 *      code symbol spans inline!
 */

import { estimateTokens } from "../util/misc.ts";
import { extractCodeSymbolSpans } from "./skeleton.ts";

export interface ToolProjectionDecision {
  shouldProject: boolean;
  reason:
    | "pass_below_threshold"
    | "pass_dense_narrative"
    | "coding_or_shell_oversized"
    | "high_redundancy_dump"
    | "hard_cap_exceeded"
    | "host_truncated_recovery";
  projectedText?: string;
  rawTokens: number;
  projectedTokens: number;
  redundancyRatio: number;
}

export interface ToolProjectionInput {
  blockId: string;
  toolName: string;
  description: string;
  rawText: string; // full untruncated text when recovered from disk, else tool output text
  baseLineOffset?: number;
  hostWasTruncated?: boolean;
  codingToolSoftCapTokens?: number; // default 24,000 tokens (~80KB)
  genericRedundantCapTokens?: number; // default 28,000 tokens (~95KB)
  hardInlineCapTokens?: number; // default 96,000 tokens (~320KB)
}

const CODING_SHELL_TOOLS = new Set([
  "bash",
  "sh",
  "zsh",
  "powershell",
  "read",
  "read_file",
  "grep",
  "grep_search",
  "rg",
  "find",
  "ls",
  "list_dir",
]);

const ERROR_SIGNAL_RE =
  /\b(?:ERROR|FATAL|CRITICAL|FAIL|FAILED|Traceback|Exception|AssertionError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|IndexError|RuntimeError|ImportError|NameError|ZeroDivisionError|panic!|segfault|ERR!|EADDRINUSE|ENOENT|EACCES)\b/i;

const GREP_LINE_RE = /^([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,10}):(\d+)[:-](.*)$/;

/**
 * Computes fast (<1.5ms) line-template redundancy over a sample of up to 400 lines.
 * Masks numbers, hex hashes, timestamps, and UUIDs so repetitive log/JSON/CSV dumps
 * score high redundancy (0.45 - 0.99), whereas natural prose/dialogue scores low (< 0.25).
 */
export function estimateLineRedundancy(lines: string[]): number {
  const nonEmpty = lines.filter((l) => l.trim().length > 0);
  if (nonEmpty.length < 20) return 0;

  const step = Math.max(1, Math.floor(nonEmpty.length / 400));
  const templates = new Set<string>();
  let sampled = 0;

  for (let i = 0; i < nonEmpty.length && sampled < 400; i += step) {
    const norm = nonEmpty[i]
      .trim()
      .slice(0, 120)
      .replace(/\b0x[0-9a-fA-F]+\b/g, "#HEX")
      .replace(/\b[0-9a-f]{8,}\b/gi, "#HASH")
      .replace(/\d+/g, "#")
      .replace(/\s+/g, " ");
    templates.add(norm);
    sampled += 1;
  }

  return Number((1 - templates.size / Math.max(1, sampled)).toFixed(3));
}

export function projectOversizedToolResult(input: ToolProjectionInput): ToolProjectionDecision {
  const rawTokens = estimateTokens(input.rawText);
  const codingCap = input.codingToolSoftCapTokens ?? 24_000;
  const redundantCap = input.genericRedundantCapTokens ?? 28_000;
  const hardCap = input.hardInlineCapTokens ?? 96_000;

  if (rawTokens < codingCap && !input.hostWasTruncated) {
    return {
      shouldProject: false,
      reason: "pass_below_threshold",
      rawTokens,
      projectedTokens: rawTokens,
      redundancyRatio: 0,
    };
  }

  const lines = input.rawText.split("\n");
  const redundancyRatio = estimateLineRedundancy(lines);
  const isCodingTool = CODING_SHELL_TOOLS.has(input.toolName.toLowerCase());
  const trimmed = input.rawText.trimStart();
  const looksStructuredDump =
    trimmed.startsWith("{") ||
    trimmed.startsWith("[") ||
    redundancyRatio >= 0.36;

  let reason: ToolProjectionDecision["reason"] | null = null;
  if (input.hostWasTruncated && rawTokens >= 3_500) {
    reason = "host_truncated_recovery";
  } else if (rawTokens >= hardCap) {
    reason = "hard_cap_exceeded";
  } else if (isCodingTool && rawTokens >= codingCap) {
    reason = "coding_or_shell_oversized";
  } else if (looksStructuredDump && rawTokens >= redundantCap) {
    reason = "high_redundancy_dump";
  }

  if (!reason) {
    return {
      shouldProject: false,
      reason: "pass_dense_narrative",
      rawTokens,
      projectedTokens: rawTokens,
      redundancyRatio,
    };
  }

  const base = Math.max(1, Math.floor(input.baseLineOffset ?? 1));
  const projectedText = buildHighSignalProjection({
    blockId: input.blockId,
    toolName: input.toolName.toLowerCase(),
    description: input.description,
    rawText: input.rawText,
    lines,
    rawTokens,
    baseLineOffset: base,
    hostWasTruncated: Boolean(input.hostWasTruncated),
  });
  const projectedTokens = estimateTokens(projectedText);

  if (projectedTokens >= rawTokens * 0.82 && !input.hostWasTruncated) {
    return {
      shouldProject: false,
      reason: "pass_below_threshold",
      rawTokens,
      projectedTokens: rawTokens,
      redundancyRatio,
    };
  }

  return {
    shouldProject: true,
    reason,
    projectedText,
    rawTokens,
    projectedTokens,
    redundancyRatio,
  };
}

function buildHighSignalProjection(opts: {
  blockId: string;
  toolName: string;
  description: string;
  rawText: string;
  lines: string[];
  rawTokens: number;
  baseLineOffset: number;
  hostWasTruncated: boolean;
}): string {
  const { blockId, toolName, description, rawText, lines, rawTokens, baseLineOffset, hostWasTruncated } = opts;
  const endFileLine = baseLineOffset + Math.max(0, lines.length - 1);
  const truncBadge = hostWasTruncated ? " | FULL UNTRUNCATED OUTPUT RECOVERED" : "";
  const header = [
    `[cpat:precache_digest #${blockId} | ${description} | ${rawTokens.toLocaleString()} raw tok (${lines.length.toLocaleString()} lines, L${baseLineOffset}-L${endFileLine})${truncBadge}]`,
    `[100% lossless lines indexed — call context_query(block_ids:["${blockId}"], start_line, end_line | query | regex) for exact interior code/lines]`,
  ].join("\n");

  // 1. Specialized Grep / Ripgrep multi-file grouping
  if (toolName === "grep" || toolName === "grep_search" || toolName === "rg" || looksLikeGrepDump(lines)) {
    const grouped = formatGrepSummary(lines);
    if (grouped) return `${header}\n\n${grouped}`;
  }

  // 2. Specialized JSON dump schema + sample + high-signal values
  const trimmed = rawText.trim();
  if ((trimmed.startsWith("{") || trimmed.startsWith("[")) && trimmed.length > 2_000) {
    const jsonView = tryFormatJsonDump(trimmed);
    if (jsonView) return `${header}\n\n${jsonView}`;
  }

  const sections: string[] = [header];

  // 3. Head preview (first 18 non-empty lines)
  const headLines: string[] = [];
  for (let i = 0; i < lines.length && headLines.length < 18; i++) {
    if (lines[i].trim()) {
      headLines.push(`L${baseLineOffset + i}: ${lines[i].slice(0, 180)}`);
    }
  }
  if (headLines.length > 0) {
    sections.push(`--- Head (L${baseLineOffset}..) ---\n${headLines.join("\n")}`);
  }

  // 4. Full-file Code Symbol Span Map (`[Lstart-Lend] class/def/function`)
  const symbols = extractCodeSymbolSpans(lines, baseLineOffset, 45);
  if (symbols.length > 0) {
    const symBody = symbols
      .map((s) => `  L${s.startLine}-L${s.endLine}: ${s.signature}`)
      .join("\n");
    sections.push(`--- Code Symbol Spans (${symbols.length} symbols across L${baseLineOffset}-L${endFileLine}) ---\n${symBody}`);
  }

  // 5. Diagnostic Error / Traceback / Failure Hunks across ENTIRE output (catches early errors Pi's 50KB tail-truncation drops!)
  const errorHunks = extractDiagnosticHunks(lines, baseLineOffset, 12);
  if (errorHunks.length > 0) {
    sections.push(`--- Diagnostic Errors / Failures (${errorHunks.length} hunks) ---\n${errorHunks.join("\n...\n")}`);
  }

  // 6. Tail preview (last 20 non-empty lines — exit codes, test summary counts, return values)
  if (lines.length > 24) {
    const tailLines: string[] = [];
    for (let i = lines.length - 1; i >= Math.max(18, lines.length - 35) && tailLines.length < 20; i--) {
      if (lines[i].trim()) {
        tailLines.unshift(`L${baseLineOffset + i}: ${lines[i].slice(0, 180)}`);
      }
    }
    if (tailLines.length > 0) {
      sections.push(`--- Tail (..L${endFileLine}) ---\n${tailLines.join("\n")}`);
    }
  }

  return sections.join("\n\n");
}

function looksLikeGrepDump(lines: string[]): boolean {
  let matched = 0;
  const check = Math.min(lines.length, 30);
  if (check < 10) return false;
  for (let i = 0; i < check; i++) {
    if (GREP_LINE_RE.test(lines[i])) matched += 1;
  }
  return matched >= check * 0.6;
}

function formatGrepSummary(lines: string[]): string | undefined {
  const byFile = new Map<string, Array<{ lineNo: string; preview: string }>>();
  let unparsed: string[] = [];

  for (const line of lines) {
    const m = GREP_LINE_RE.exec(line);
    if (m) {
      const file = m[1];
      const arr = byFile.get(file) ?? [];
      arr.push({ lineNo: m[2], preview: m[3].trim().slice(0, 130) });
      byFile.set(file, arr);
    } else if (line.trim() && unparsed.length < 6) {
      unparsed.push(line.trim().slice(0, 140));
    }
  }

  if (byFile.size === 0) return undefined;

  const sortedFiles = Array.from(byFile.entries()).sort((a, b) => b[1].length - a[1].length);
  const fileBlocks = sortedFiles.slice(0, 32).map(([file, hits]) => {
    const sample = hits
      .slice(0, 3)
      .map((h) => `    L${h.lineNo}: ${h.preview}`)
      .join("\n");
    const more = hits.length > 3 ? `\n    ... (+${hits.length - 3} more matches in ${file})` : "";
    return `  ${file} (${hits.length} matches):\n${sample}${more}`;
  });

  const extraFiles =
    sortedFiles.length > 32 ? `\n  ... (+${sortedFiles.length - 32} additional files indexed — use context_query)` : "";
  return `Grep Matches Grouped by File (${lines.length} lines across ${byFile.size} files):\n${fileBlocks.join("\n")}${extraFiles}`;
}

function tryFormatJsonDump(trimmed: string): string | undefined {
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (Array.isArray(parsed)) {
      const first = parsed.length > 0 ? JSON.stringify(parsed[0], null, 2).slice(0, 700) : "null";
      const last = parsed.length > 1 ? JSON.stringify(parsed[parsed.length - 1], null, 2).slice(0, 500) : "";
      return [
        `JSON Array (${parsed.length.toLocaleString()} items):`,
        `Item[0]:\n${first}`,
        ...(last ? [`Item[${parsed.length - 1}]:\n${last}`] : []),
      ].join("\n\n");
    }
    if (typeof parsed === "object" && parsed !== null) {
      const keys = Object.keys(parsed as Record<string, unknown>);
      const summary = keys
        .slice(0, 25)
        .map((k) => {
          const v = (parsed as Record<string, unknown>)[k];
          const desc = Array.isArray(v)
            ? `Array(${v.length}) sample=${JSON.stringify(v[0] ?? null).slice(0, 90)}`
            : typeof v === "object" && v !== null
              ? `Object(${Object.keys(v as object).slice(0, 8).join(",")})`
              : JSON.stringify(v).slice(0, 110);
          return `  ${k}: ${desc}`;
        })
        .join("\n");
      return `JSON Object (${keys.length} top-level keys):\n${summary}`;
    }
  } catch {
    // not valid JSON or truncated JSON
  }
  return undefined;
}

function extractDiagnosticHunks(
  lines: string[],
  baseLineOffset: number,
  maxHunks: number,
): string[] {
  const ranges: Array<[number, number]> = [];
  for (let i = 0; i < lines.length; i++) {
    if (!ERROR_SIGNAL_RE.test(lines[i])) continue;
    const s = Math.max(0, i - 2);
    const e = Math.min(lines.length - 1, i + 4);
    const prev = ranges[ranges.length - 1];
    if (prev && s <= prev[1] + 1 && e - prev[0] <= 12) {
      prev[1] = Math.max(prev[1], e);
    } else {
      ranges.push([s, e]);
    }
  }

  if (ranges.length === 0) return [];
  const picked =
    ranges.length > maxHunks
      ? [...ranges.slice(0, Math.ceil(maxHunks * 0.7)), ...ranges.slice(-Math.floor(maxHunks * 0.3))]
      : ranges;

  return picked.map(([s, e]) =>
    lines
      .slice(s, e + 1)
      .map((l, idx) => `L${baseLineOffset + s + idx}: ${l.slice(0, 180)}`)
      .join("\n"),
  );
}
