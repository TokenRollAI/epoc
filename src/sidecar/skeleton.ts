import { estimateTokens } from "../util/misc.ts";

/**
 * Deterministic (<2ms, zero-LLM) structural skeleton & verbatim signal extractor.
 *
 * Serves two critical roles in ContextHarness:
 *   1. Multi-resolution "skeleton" tier (inspired by Rivet): keeps structural
 *      signatures, headers, schemas, and high-entropy verbatim codes/numbers
 *      while shedding repetitive filler prose.
 *   2. Zero-latency safety fallback whenever the cheap sidecar model is not
 *      configured, still in-flight, or unreachable.
 */

const STRUCTURAL_LINE_RE =
  /^\s*(?:#{1,6}\s+|={3,}|-{3,}|\b(?:section|chapter|part|project|service|batch|module|summary|conclusion)\b[:\s]|\b(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|enum|def|fn|struct|impl)\s+[A-Za-z_])/i;

const HIGH_SIGNAL_LINE_RE =
  /(?:\b[A-Z]{2,8}-\d[\w-]*\b|\b0x[0-9a-fA-F]{4,}\b|\b\d+(?:\.\d+)?%|\b(?:ERROR|FATAL|FAIL|WARN|CRITICAL|TIMEOUT|SLA|lead|depends?\s+on|accuracy|incident|commit|root_cause|formula|capacity)\b)/i;

export function buildLandmark(
  blockId: string,
  description: string,
  rawTokens: number,
  lineCount: number,
): string {
  const cleanDesc = description.replace(/\s+/g, " ").trim().slice(0, 100);
  return `[context:landmark #${blockId} | ${cleanDesc} | ${rawTokens} tok (${lineCount} lines) indexed — call context_query to inspect]`;
}

const CODE_SYMBOL_RE =
  /^(\s*)(?:@\w[\w.]*(?:\([^)]*\))?\s*)*(?:export\s+(?:default\s+)?)?(?:public\s+|private\s+|protected\s+|static\s+|abstract\s+)?(?:async\s+)?(class|def|function|interface|enum|func|fn|struct|impl)\s+([A-Za-z_][\w]*)/;

export interface CodeSymbolSpan {
  kind: string;
  name: string;
  startLine: number;
  endLine: number;
  signature: string;
}

/**
 * Deterministic <1ms code symbol span extractor (Rivet-style structural anchors).
 * Computes exact 1-based [startLine, endLine] in the source file (honoring baseLineOffset).
 */
export function extractCodeSymbolSpans(
  lines: string[],
  baseLineOffset = 1,
  maxSymbols = 28,
): CodeSymbolSpan[] {
  const base = Math.max(1, Math.floor(baseLineOffset));
  const raw: Array<{
    idx: number;
    indent: number;
    kind: string;
    name: string;
    sig: string;
  }> = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const m = CODE_SYMBOL_RE.exec(line);
    if (!m) continue;
    const indent = m[1].replace(/\t/g, "    ").length;
    raw.push({
      idx: i,
      indent,
      kind: m[2],
      name: m[3],
      sig: line.trim().slice(0, 120),
    });
  }

  if (raw.length === 0) return [];

  const spans: CodeSymbolSpan[] = [];
  for (let k = 0; k < raw.length && spans.length < maxSymbols; k++) {
    const cur = raw[k];
    let endIdx = lines.length - 1;
    // Scan forward until a non-blank, non-comment line has indent <= cur.indent
    // (skipping immediate continuation lines of a multi-line signature).
    let bodyStarted = false;
    for (let j = cur.idx + 1; j < lines.length; j++) {
      const trimmed = lines[j].trim();
      if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("//")) continue;
      const lineIndent = (lines[j].match(/^\s*/)?.[0] ?? "").replace(/\t/g, "    ").length;
      if (!bodyStarted) {
        if (lineIndent > cur.indent || trimmed.startsWith(")") || trimmed.startsWith("->")) {
          bodyStarted = true;
          continue;
        }
      }
      // Closing brace at same indent belongs to this block
      if (lineIndent === cur.indent && (trimmed === "}" || trimmed === "};")) {
        endIdx = j;
        break;
      }
      if (lineIndent <= cur.indent) {
        endIdx = j - 1;
        break;
      }
    }
    while (endIdx > cur.idx && !lines[endIdx].trim()) {
      endIdx -= 1;
    }
    spans.push({
      kind: cur.kind,
      name: cur.name,
      startLine: base + cur.idx,
      endLine: base + Math.max(cur.idx, endIdx),
      signature: cur.sig,
    });
  }
  return spans;
}

export function extractSkeleton(
  blockId: string,
  description: string,
  content: string,
  maxChars = 1500,
  baseLineOffset = 1,
): string {
  const base = Math.max(1, Math.floor(baseLineOffset));
  const trimmed = content.trim();
  const lines = content.split("\n");
  const rawTokens = estimateTokens(content);

  // 1. JSON structural condensation
  if ((trimmed.startsWith("{") || trimmed.startsWith("[")) && trimmed.length > 200) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (Array.isArray(parsed)) {
        const sample = parsed.length > 0 ? JSON.stringify(parsed[0]).slice(0, 260) : "empty";
        return [
          `[context:skeleton #${blockId} | JSON Array (${parsed.length} items, ${rawTokens} tok) | ${description}]`,
          `Sample[0]: ${sample}`,
          ...extractVerbatimCodeBadges(content),
        ].join("\n");
      }
      if (typeof parsed === "object" && parsed !== null) {
        const keys = Object.keys(parsed as Record<string, unknown>);
        const preview = keys
          .slice(0, 12)
          .map((k) => {
            const v = (parsed as Record<string, unknown>)[k];
            const valStr = typeof v === "object" ? JSON.stringify(v).slice(0, 80) : String(v).slice(0, 80);
            return `  ${k}: ${valStr}`;
          })
          .join("\n");
        return [
          `[context:skeleton #${blockId} | JSON Object (${keys.length} keys, ${rawTokens} tok) | ${description}]`,
          preview,
          ...extractVerbatimCodeBadges(content),
        ].join("\n");
      }
    } catch {
      // fall through to line-based skeleton
    }
  }

  // 2. Line-based structural + symbol spans + high-signal extraction (code / dossiers / logs)
  const symbolSpans = extractCodeSymbolSpans(lines, base, 22);
  const picked = new Map<number, string>();
  const addLine = (idx: number) => {
    if (idx < 0 || idx >= lines.length) return;
    const text = lines[idx].trimEnd();
    if (!text.trim()) return;
    const fileLine = base + idx;
    picked.set(fileLine, `L${fileLine}: ${text.slice(0, 160)}`);
  };

  // Head context (first 4 non-empty lines)
  let headAdded = 0;
  for (let i = 0; i < lines.length && headAdded < 4; i++) {
    if (lines[i].trim()) {
      addLine(i);
      headAdded++;
    }
  }

  // Embed exact [Lstart-Lend] code symbol spans when present
  for (const sym of symbolSpans) {
    picked.set(sym.startLine, `L${sym.startLine}-L${sym.endLine}: ${sym.signature}`);
  }

  // Structural declarations & high-signal verbatim lines across the whole body
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fileLine = base + i;
    if (!picked.has(fileLine) && (STRUCTURAL_LINE_RE.test(line) || HIGH_SIGNAL_LINE_RE.test(line))) {
      addLine(i);
    }
    if (picked.size >= 26) break;
  }

  // Tail context (last 2 non-empty lines — catches buried conclusions / exit codes)
  let tailAdded = 0;
  for (let i = lines.length - 1; i >= 0 && tailAdded < 2; i--) {
    if (lines[i].trim() && !picked.has(base + i)) {
      addLine(i);
      tailAdded++;
    }
  }

  const ordered = Array.from(picked.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, line]) => line);

  const endFileLine = base + Math.max(0, lines.length - 1);
  const header = `[context:skeleton #${blockId} | ${description} | ${rawTokens} tok, L${base}-L${endFileLine} indexed — call context_query(block_ids:["${blockId}"], start_line, end_line) for full lines]`;
  let body = ordered.join("\n");
  if (body.length > maxChars) {
    body = body.slice(0, maxChars) + "\n…[skeleton capped; full lines indexed in SearchIndex]";
  }
  return `${header}\n${body}`;
}

function extractVerbatimCodeBadges(content: string): string[] {
  const codes = Array.from(
    new Set(content.match(/\b[A-Z]{2,8}-\d[\w-]*\b|\b\d+\.\d+%/g) ?? []),
  ).slice(0, 16);
  return codes.length > 0 ? [`Detected identifiers/metrics: ${codes.join(", ")}`] : [];
}
