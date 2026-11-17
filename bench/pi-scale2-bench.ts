/**
 * Scale-2 (120-Item) Stratified Full-Subtask 400K-Window Benchmark Runner (`npm run bench:pi-120`).
 *
 * Evaluates 4 real Pi CLI (`@earendil-works/pi-coding-agent` v1.5.2) arms at **400,000-token (400K)**
 * context window across 120 benchmark items (480 item-arm evaluations):
 *
 *   1. `LongMemEval-S` (40 Qs = 10 × 4-Arc Continuous ~450K Marathons across KU, MSR, TR, IE, ABS)
 *   2. `MemoryAgentBench` (40 Qs across all 4 official splits: CR=10, AR=10 @ 480K, LRU=10 @ 180K, TTL=10 @ 1M-1.48M!)
 *   3. `ContextBench Verified` (30 SWE issues = 10 × 3-Issue Continuous Multi-Repo Sessions across ~15 OSS repos)
 *   4. `Oversized-Tools & Pollution-Boundary` (10 SWE diagnostic & context-pollution items)
 *
 * Features:
 *   - Bounded parallel worker pool (`--concurrency 6`) with isolated per-run Pi directories
 *   - Incremental per-unit checkpointing (`--resume <outDir>`)
 *   - Full `third-party-1500` unified pricing ($0.15/M miss, $0.003/M hit, $0.60/M output) + Cache-Fork telemetry
 */

import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { CpatPiSessionMetrics } from "../extensions/pi-cpat.ts";
import { DeepSeekClient } from "../src/deepseek/client.ts";
import { loadDeepSeekEnv } from "../src/util/env.ts";
import { estimateTokens } from "../src/util/misc.ts";
import { normalizeAnswer, qaF1Score } from "./f1.ts";
import type { CbScaleItem, LmeScaleItem, MabScaleEpisode } from "./prepare-scale2-data.ts";

export type PiArm = "full" | "pi-native" | "billion-context" | "cpat";

const PRICE_MISS_PER_M = 0.15;
const PRICE_HIT_PER_M = 0.003;
const PRICE_OUT_PER_M = 0.6;

const DATA_ROOT = resolve("data/public");
const SCALE2_DIR = join(DATA_ROOT, "scale2_120");
const CPAT_EXT_PATH = resolve("extensions/pi-cpat.ts");
const FEEDER_EXT_PATH = resolve("bench/pi-stream-feeder.ts");
const TELEMETRY_EXT_PATH = resolve("bench/pi-telemetry-tool.ts");
const HARD_WIN_EXT_PATH = resolve("bench/pi-hard-window.ts");
const BC_EXT_PATH = resolve("data/public/bc-npm/node_modules/billion-context/dist/agent/billion-context.js");

interface PiRunMetrics {
  turns: number;
  promptTokens: number;
  cachedTokens: number;
  uncachedTokens: number;
  outputTokens: number;
  peakPromptTokens: number;
  compactions: number;
  cacheForkSummaries: number;
  cacheForkPromptTokens: number;
  cacheForkCachedTokens: number;
  boundaryFolds: number;
  preCacheProjections: number;
  sidecarCalls: number;
  sidecarPromptTokens: number;
  sidecarCachedTokens: number;
  sidecarOutputTokens: number;
  mainCostUsd: number;
  sidecarCostUsd: number;
  totalUnifiedCostUsd: number;
  toolCounts: Record<string, number>;
  overflowOrError: boolean;
  errorMessage?: string;
  finalAnswer: string;
  wallClockS: number;
}

function unifiedUsd(uncachedIn: number, cachedIn: number, out: number): number {
  return (
    (uncachedIn / 1_000_000) * PRICE_MISS_PER_M +
    (cachedIn / 1_000_000) * PRICE_HIT_PER_M +
    (out / 1_000_000) * PRICE_OUT_PER_M
  );
}

function bestSubstringMatch(prediction: string, groundTruths: string[]): number {
  const normPred = normalizeAnswer(prediction);
  if (!normPred) return 0;
  for (const gt of groundTruths) {
    const normGt = normalizeAnswer(gt);
    if (normGt && normPred.includes(normGt)) return 1;
  }
  return 0;
}

/** Token-level ROUGE-L F1 (longest common subsequence over normalized tokens, capped at 600 tokens for O(N*M) speed). */
function rougeLF1(prediction: string, references: string[]): number {
  const predTokens = normalizeAnswer(prediction).split(/\s+/).filter(Boolean).slice(0, 600);
  if (predTokens.length === 0) return 0;

  let best = 0;
  for (const ref of references) {
    const refTokens = normalizeAnswer(ref).split(/\s+/).filter(Boolean).slice(0, 600);
    if (refTokens.length === 0) continue;
    const m = predTokens.length;
    const n = refTokens.length;
    let prev = new Uint16Array(n + 1);
    let curr = new Uint16Array(n + 1);
    for (let i = 1; i <= m; i++) {
      const pt = predTokens[i - 1];
      for (let j = 1; j <= n; j++) {
        if (pt === refTokens[j - 1]) {
          curr[j] = prev[j - 1] + 1;
        } else {
          curr[j] = prev[j] >= curr[j - 1] ? prev[j] : curr[j - 1];
        }
      }
      const tmp = prev;
      prev = curr;
      curr = tmp;
      curr.fill(0);
    }
    const lcs = prev[n];
    if (lcs > 0) {
      const p = lcs / m;
      const r = lcs / n;
      const f1 = (2 * p * r) / (p + r);
      if (f1 > best) best = f1;
    }
  }
  return best;
}

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T, idx: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIdx = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const idx = nextIdx++;
      if (idx >= items.length) break;
      results[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(runners);
  return results;
}

function packIntoChunks(blocks: string[], targetChars: number): string[] {
  const chunks: string[] = [];
  let cur: string[] = [];
  let curLen = 0;
  for (const b of blocks) {
    if (cur.length > 0 && curLen + b.length > targetChars) {
      chunks.push(cur.join("\n\n"));
      cur = [];
      curLen = 0;
    }
    cur.push(b);
    curLen += b.length;
  }
  if (cur.length > 0) chunks.push(cur.join("\n\n"));
  return chunks;
}

// ---------------------------------------------------------------------------
// Isolated Pi CLI Runner
// ---------------------------------------------------------------------------

async function runPiCli(opts: {
  arm: PiArm;
  prompt: string;
  cwd: string;
  runWorkDir: string;
  sessionFile?: string;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  builtinTools?: string;
  streamChunksFile?: string;
  streamMaxBatchCount?: number;
  telemetryDumpFile?: string;
  timeoutMs?: number;
  tag: string;
  quietTurns?: boolean;
}): Promise<PiRunMetrics> {
  mkdirSync(opts.runWorkDir, { recursive: true });
  const piAgentDir = join(opts.runWorkDir, "pi-agent");
  const piHomeDir = join(opts.runWorkDir, "home");
  const sessionDir = join(opts.runWorkDir, "sessions");
  const xdgCfg = join(opts.runWorkDir, "xdg-config");
  const xdgData = join(opts.runWorkDir, "xdg-data");
  const xdgState = join(opts.runWorkDir, "xdg-state");
  mkdirSync(piAgentDir, { recursive: true });
  mkdirSync(piHomeDir, { recursive: true });
  mkdirSync(sessionDir, { recursive: true });

  const effectiveWin = opts.windowTokens;

  writeFileSync(
    join(piAgentDir, "models.json"),
    JSON.stringify(
      {
        providers: {
          deepseek: {
            baseUrl: opts.baseUrl,
            modelOverrides: {
              [opts.model]: {
                contextWindow: opts.arm === "full" ? Math.max(effectiveWin + 100_000, 1_000_000) : effectiveWin,
              },
            },
          },
        },
      },
      null,
      2,
    ),
    "utf8",
  );

  writeFileSync(
    join(piAgentDir, "settings.json"),
    JSON.stringify({ reserveTokens: 16_384, keepRecentTokens: 20_000 }, null, 2),
    "utf8",
  );

  if (opts.arm === "billion-context") {
    const bcCfgDir = join(xdgCfg, "billion-context");
    mkdirSync(bcCfgDir, { recursive: true });
    mkdirSync(xdgData, { recursive: true });
    mkdirSync(xdgState, { recursive: true });
    const cfgFile = join(bcCfgDir, "billion-context.json");
    if (!existsSync(cfgFile)) {
      writeFileSync(
        cfgFile,
        JSON.stringify(
          {
            pi: { inline: true, subagents: false, commands: false },
            guardrails: { maxOutputBytes: 0 },
            compress: {
              modelContextLimit: effectiveWin,
              maxContextLimit: "75%",
              emergencyThresholdPercent: "95%",
              nudgeGrowthTokens: 50_000,
            },
          },
          null,
          2,
        ),
        "utf8",
      );
    }
  }

  const cpatMetricsFile = join(opts.runWorkDir, "cpat-metrics.json");
  let prevSidecarCalls = 0;
  let prevSidecarIn = 0;
  let prevSidecarCached = 0;
  let prevSidecarOut = 0;
  let prevEdits = 0;
  let prevPreCache = 0;
  let prevForkCount = 0;
  let prevForkPrompt = 0;
  let prevForkCached = 0;
  let prevFolds = 0;
  if (opts.arm === "cpat" && existsSync(cpatMetricsFile)) {
    try {
      const prev = JSON.parse(readFileSync(cpatMetricsFile, "utf8")) as CpatPiSessionMetrics;
      prevSidecarCalls = prev.sidecarCalls;
      prevSidecarIn = prev.sidecarInputTokens;
      prevSidecarCached = prev.sidecarCachedTokens;
      prevSidecarOut = prev.sidecarOutputTokens;
      prevEdits = prev.contextEditsEmitted;
      prevPreCache = prev.preCacheToolProjections ?? 0;
      prevForkCount = prev.cacheForkSummaries ?? 0;
      prevForkPrompt = prev.cacheForkPromptTokens ?? 0;
      prevForkCached = prev.cacheForkCachedTokens ?? 0;
      prevFolds = prev.boundaryFolds ?? 0;
    } catch {
      // ignore
    }
  }

  const args: string[] = [
    "--mode",
    "json",
    "-p",
    "--session-dir",
    sessionDir,
    ...(opts.sessionFile ? ["--session", opts.sessionFile] : []),
    "-ne",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--no-mcp",
    "--model",
    `deepseek/${opts.model}`,
    "--thinking",
    "low",
  ];

  if (opts.builtinTools) args.push("--tools", opts.builtinTools);
  else args.push("--no-builtin-tools");

  if (opts.streamChunksFile) args.push("-e", FEEDER_EXT_PATH);
  if (opts.telemetryDumpFile) args.push("-e", TELEMETRY_EXT_PATH);
  if (opts.arm === "full") args.push("-e", HARD_WIN_EXT_PATH);
  else if (opts.arm === "billion-context") args.push("-e", BC_EXT_PATH);
  else if (opts.arm === "cpat") args.push("-e", CPAT_EXT_PATH);

  args.push("--", opts.prompt);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    DEEPSEEK_API_KEY: opts.apiKey,
    PI_CODING_AGENT_DIR: piAgentDir,
    HOME: piHomeDir,
    PI_OFFLINE: "1",
    PI_HARD_WINDOW_TOKENS: String(effectiveWin),
    ...(opts.streamChunksFile ? { PI_STREAM_CHUNKS_FILE: opts.streamChunksFile } : {}),
    ...(opts.streamMaxBatchCount ? { PI_STREAM_MAX_BATCH_COUNT: String(opts.streamMaxBatchCount) } : {}),
    ...(opts.telemetryDumpFile ? { PI_TELEMETRY_DUMP_FILE: opts.telemetryDumpFile } : {}),
    ...(opts.arm === "billion-context"
      ? {
          BILI_PI_INLINE: "1",
          BILI_GUARDRAIL_MAX_OUTPUT_BYTES: "0",
          XDG_CONFIG_HOME: xdgCfg,
          XDG_DATA_HOME: xdgData,
          XDG_STATE_HOME: xdgState,
          BILI_ADVISORY_CHECK: "0",
          BILI_RELEASE_NOTES_CHECK: "0",
          BILI_UPDATE_CHECK_INTERVAL_MS: "0",
        }
      : {
          BILLION_CONTEXT_PLUGIN: "0",
          BILI_NATIVE_PI: "0",
        }),
    ...(opts.arm === "cpat"
      ? {
          CPAT_MAX_CONTEXT: String(effectiveWin),
          CPAT_SIDECAR_MODEL: opts.model,
          CPAT_METRICS_FILE: cpatMetricsFile,
          CPAT_STORE_DIR: join(opts.runWorkDir, "cpat-store"),
        }
      : {}),
  };

  const t0 = Date.now();
  let turns = 0;
  let uncachedTokens = 0;
  let cachedTokens = 0;
  let outputTokens = 0;
  let peakPromptTokens = 0;
  let compactions = 0;
  let sidecarCalls = 0;
  let sidecarPromptTokens = 0;
  let sidecarCachedTokens = 0;
  let sidecarOutputTokens = 0;
  let transientTurnErrors = 0;
  const toolCounts: Record<string, number> = {};
  let overflowOrError = false;
  let errorMessage: string | undefined;
  let finalAnswer = "";

  await new Promise<void>((resolveDone) => {
    const child = spawn("pi", args, {
      cwd: opts.cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdoutBuf = "";
    let stderrBuf = "";
    let settled = false;

    const timeout = setTimeout(() => {
      if (!settled) {
        overflowOrError = true;
        errorMessage = `Timeout after ${Math.round((opts.timeoutMs ?? 600_000) / 1000)}s`;
        child.kill("SIGKILL");
      }
    }, opts.timeoutMs ?? 600_000);

    const handleRecord = (line: string) => {
      const trimmed = line.replace(/\r$/, "").trim();
      if (!trimmed.startsWith("{")) return;
      let ev: Record<string, any>;
      try {
        ev = JSON.parse(trimmed);
      } catch {
        return;
      }

      if (ev.type === "tool_execution_start" && typeof ev.toolName === "string") {
        toolCounts[ev.toolName] = (toolCounts[ev.toolName] ?? 0) + 1;
      } else if (ev.type === "extension_error") {
        const errStr = String(ev.error ?? "");
        if (errStr.includes("ContextWindowExceededError")) {
          overflowOrError = true;
          errorMessage = errStr;
        }
      } else if (ev.type === "compaction_end") {
        if (!ev.aborted && ev.result) {
          compactions += 1;
          const u = ev.result.usage;
          if (u && opts.arm !== "cpat") {
            sidecarCalls += 1;
            const cIn = Number(u.cacheRead ?? 0);
            const uIn = Number(u.input ?? 0);
            const out = Number(u.output ?? 0);
            sidecarPromptTokens += uIn + cIn;
            sidecarCachedTokens += cIn;
            sidecarOutputTokens += out;
          }
        }
      } else if (ev.type === "message_end" && ev.message?.role === "assistant") {
        const msg = ev.message;
        turns += 1;
        const u = msg.usage ?? {};
        const uIn = Number(u.input ?? 0);
        const cIn = Number(u.cacheRead ?? 0) + Number(u.cacheWrite ?? 0);
        const out = Number(u.output ?? 0);
        const turnPrompt = uIn + cIn;
        uncachedTokens += uIn;
        cachedTokens += cIn;
        outputTokens += out;
        if (turnPrompt > peakPromptTokens) peakPromptTokens = turnPrompt;

        const textParts = Array.isArray(msg.content)
          ? msg.content
              .filter((p: any) => p?.type === "text" && typeof p.text === "string")
              .map((p: any) => p.text)
              .join("\n")
          : "";
        const thinkingParts = Array.isArray(msg.content)
          ? msg.content
              .filter((p: any) => p?.type === "thinking" && typeof p.thinking === "string")
              .map((p: any) => p.thinking)
              .join("\n")
          : "";
        const hasToolCall =
          Array.isArray(msg.content) && msg.content.some((p: any) => p?.type === "toolCall");
        if (textParts.trim()) {
          finalAnswer = textParts.trim();
        } else if (!hasToolCall && thinkingParts.trim()) {
          // DeepSeek Flash occasionally emits its final JSON answer inside `thinking` with empty `text` on stop!
          finalAnswer = thinkingParts.trim();
        }
        if (msg.stopReason === "error" || msg.stopReason === "aborted") {
          transientTurnErrors += 1;
          errorMessage = String(msg.errorMessage ?? `stopReason=${msg.stopReason}`);
        }
      }
    };

    child.stdout.on("data", (chunk: Buffer) => {
      stdoutBuf += chunk.toString("utf8");
      let nl: number;
      while ((nl = stdoutBuf.indexOf("\n")) !== -1) {
        const line = stdoutBuf.slice(0, nl);
        stdoutBuf = stdoutBuf.slice(nl + 1);
        handleRecord(line);
      }
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderrBuf += chunk.toString("utf8");
    });

    child.on("close", (code) => {
      settled = true;
      clearTimeout(timeout);
      if (stdoutBuf.trim()) handleRecord(stdoutBuf);
      if (stderrBuf.includes("ContextWindowExceededError") || stderrBuf.includes("context_length_exceeded")) {
        overflowOrError = true;
        errorMessage = "Exceeded context window";
      } else if (code !== 0 && !finalAnswer) {
        overflowOrError = true;
        errorMessage = errorMessage ?? `pi exited ${code}: ${stderrBuf.slice(-180)}`;
      } else if (transientTurnErrors > 0 && !finalAnswer) {
        overflowOrError = true;
      }
      resolveDone();
    });
  });

  let cacheForkSummaries = 0;
  let cacheForkPromptTokens = 0;
  let cacheForkCachedTokens = 0;
  let boundaryFolds = 0;
  let preCacheProjections = 0;

  if (opts.arm === "cpat" && existsSync(cpatMetricsFile)) {
    try {
      const m = JSON.parse(readFileSync(cpatMetricsFile, "utf8")) as CpatPiSessionMetrics;
      sidecarCalls = Math.max(0, m.sidecarCalls - prevSidecarCalls);
      sidecarPromptTokens = Math.max(0, m.sidecarInputTokens - prevSidecarIn);
      sidecarCachedTokens = Math.max(0, m.sidecarCachedTokens - prevSidecarCached);
      sidecarOutputTokens = Math.max(0, m.sidecarOutputTokens - prevSidecarOut);
      preCacheProjections = Math.max(0, (m.preCacheToolProjections ?? 0) - prevPreCache);
      cacheForkSummaries = Math.max(0, (m.cacheForkSummaries ?? 0) - prevForkCount);
      cacheForkPromptTokens = Math.max(0, (m.cacheForkPromptTokens ?? 0) - prevForkPrompt);
      cacheForkCachedTokens = Math.max(0, (m.cacheForkCachedTokens ?? 0) - prevForkCached);
      boundaryFolds = Math.max(0, (m.boundaryFolds ?? 0) - prevFolds);
      compactions += Math.max(0, m.contextEditsEmitted - prevEdits) + preCacheProjections;
    } catch {
      // ignore
    }
  }

  const promptTokens = uncachedTokens + cachedTokens;
  const mainCostUsd = unifiedUsd(uncachedTokens, cachedTokens, outputTokens);
  const sidecarUncached = Math.max(0, sidecarPromptTokens - sidecarCachedTokens);
  const sidecarCostUsd = unifiedUsd(sidecarUncached, sidecarCachedTokens, sidecarOutputTokens);

  return {
    turns,
    promptTokens,
    cachedTokens,
    uncachedTokens,
    outputTokens,
    peakPromptTokens,
    compactions,
    cacheForkSummaries,
    cacheForkPromptTokens,
    cacheForkCachedTokens,
    boundaryFolds,
    preCacheProjections,
    sidecarCalls,
    sidecarPromptTokens,
    sidecarCachedTokens,
    sidecarOutputTokens,
    mainCostUsd,
    sidecarCostUsd,
    totalUnifiedCostUsd: mainCostUsd + sidecarCostUsd,
    toolCounts,
    overflowOrError,
    errorMessage,
    finalAnswer,
    wallClockS: Number(((Date.now() - t0) / 1000).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// Evaluators & Scorers
// ---------------------------------------------------------------------------

async function judgeLongMemEval(
  client: DeepSeekClient,
  model: string,
  item: LmeScaleItem,
  prediction: string,
): Promise<boolean> {
  if (!prediction.trim()) return false;
  if (item.category !== "ABS" && bestSubstringMatch(prediction, [item.answer]) === 1) {
    return true;
  }
  try {
    const sysPrompt =
      item.category === "ABS"
        ? 'You are an exact QA evaluator for LongMemEval Abstention questions. The ground-truth indicates the requested fact was NEVER mentioned or cannot be answered from the history. Return ONLY JSON: {"correct": true} if the prediction appropriately abstains / states the information is not mentioned / unavailable without hallucinating a false fact, otherwise {"correct": false}.'
        : 'You are an exact QA evaluator for LongMemEval. Compare the predicted answer against the ground-truth answer for the given question. Return ONLY JSON: {"correct": true} if the prediction accurately contains the ground-truth fact/value/entity/count (allowing equivalent phrasing), otherwise {"correct": false}.';

    const resp = await client.chat({
      model,
      messages: [
        { role: "system", content: sysPrompt },
        {
          role: "user",
          content: `Question: ${item.question}\nGround Truth: ${item.answer}\nPrediction: ${prediction.slice(0, 1500)}`,
        },
      ],
      temperature: 0,
      maxTokens: 200,
      reasoningEffort: "low",
      responseFormat: { type: "json_object" },
    });
    const obj = JSON.parse(resp.message.content ?? "{}") as { correct?: boolean };
    return Boolean(obj.correct);
  } catch {
    return false;
  }
}

function scoreContextBench(
  answer: string,
  gold: CbScaleItem["gold_context"],
): { lineF1: number; linePrecision: number; lineRecall: number; fileF1: number } {
  const predSpans: Array<{ file: string; start_line: number; end_line: number }> = [];
  const jsonMatch =
    answer.match(/```(?:json)?\s*(\{[\s\S]*?"spans"[\s\S]*?\})\s*```/i) ??
    answer.match(/(\{[\s\S]*?"spans"\s*:\s*\[[\s\S]*?\][\s\S]*?\})/);

  if (jsonMatch) {
    try {
      const obj = JSON.parse(jsonMatch[1]) as { spans?: Array<Record<string, unknown>> };
      for (const s of obj.spans ?? []) {
        if (typeof s.file === "string") {
          const cleanFile = s.file.replace(/^\.?\//, "").trim();
          const st = Math.max(1, Number(s.start_line ?? 1));
          const en = Math.max(st, Number(s.end_line ?? st + 30));
          predSpans.push({ file: cleanFile, start_line: st, end_line: en });
        }
      }
    } catch {
      // fallback below
    }
  }
  if (predSpans.length === 0) {
    for (const m of answer.matchAll(/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)(?::L?(\d+)(?:-L?(\d+))?)?/g)) {
      const file = m[1].replace(/^\.?\//, "");
      const st = m[2] ? Number(m[2]) : 1;
      const en = m[3] ? Number(m[3]) : st + 40;
      predSpans.push({ file, start_line: st, end_line: en });
    }
  }

  const goldFiles = new Set(gold.map((g) => g.file.replace(/^\.?\//, "")));
  const predFiles = new Set(predSpans.map((p) => p.file));
  let fileInter = 0;
  for (const f of predFiles) if (goldFiles.has(f)) fileInter += 1;
  const fileP = predFiles.size > 0 ? fileInter / predFiles.size : 0;
  const fileR = goldFiles.size > 0 ? fileInter / goldFiles.size : 0;
  const fileF1 = fileP + fileR > 0 ? (2 * fileP * fileR) / (fileP + fileR) : 0;

  const goldLines = new Set<string>();
  for (const g of gold) {
    const f = g.file.replace(/^\.?\//, "");
    for (let l = g.start_line; l <= g.end_line; l++) goldLines.add(`${f}:${l}`);
  }
  const predLines = new Set<string>();
  for (const p of predSpans) {
    const capEnd = Math.min(p.end_line, p.start_line + 250);
    for (let l = p.start_line; l <= capEnd; l++) predLines.add(`${p.file}:${l}`);
  }

  let lineInter = 0;
  for (const k of predLines) if (goldLines.has(k)) lineInter += 1;
  const lineP = predLines.size > 0 ? lineInter / predLines.size : 0;
  const lineR = goldLines.size > 0 ? lineInter / goldLines.size : 0;
  const lineF1 = lineP + lineR > 0 ? (2 * lineP * lineR) / (lineP + lineR) : 0;

  return { lineF1, linePrecision: lineP, lineRecall: lineR, fileF1 };
}

function parseJsonAnswersArray(text: string): string[] {
  const jm =
    text.match(/```(?:json)?\s*(\{[\s\S]*?"answers"[\s\S]*?\})\s*```/i) ??
    text.match(/(\{[\s\S]*?"answers"\s*:\s*\[[\s\S]*?\][\s\S]*?\})/);
  if (!jm) return [];
  try {
    const obj = JSON.parse(jm[1]) as { answers?: unknown[] };
    if (Array.isArray(obj.answers)) return obj.answers.map((x) => String(x ?? ""));
  } catch {
    // ignore
  }
  return [];
}

// ---------------------------------------------------------------------------
// Unit Task Builders (Checkpoint-Aware)
// ---------------------------------------------------------------------------

interface UnitCheckpoint {
  suite: "longmemeval" | "memoryagentbench" | "contextbench" | "oversized-pollution";
  unitId: string;
  arm: PiArm;
  subScores: Array<{
    subtask: string;
    primary: number;
    secondary: number;
  }>;
  overflowCount: number;
  metrics: {
    turns: number;
    promptTokens: number;
    cachedTokens: number;
    uncachedTokens: number;
    outputTokens: number;
    peakPromptTokens: number;
    compactions: number;
    cacheForkSummaries: number;
    cacheForkPromptTokens: number;
    cacheForkCachedTokens: number;
    boundaryFolds: number;
    preCacheProjections: number;
    recoveryQueries: number;
    sidecarCostUsd: number;
    totalUnifiedCostUsd: number;
    wallClockS: number;
  };
}

function summarizeRunMetrics(res: PiRunMetrics): UnitCheckpoint["metrics"] {
  const recov =
    (res.toolCounts["context_query"] ?? 0) +
    (res.toolCounts["search_context"] ?? 0) +
    (res.toolCounts["decompress"] ?? 0);
  return {
    turns: res.turns,
    promptTokens: res.promptTokens,
    cachedTokens: res.cachedTokens,
    uncachedTokens: res.uncachedTokens,
    outputTokens: res.outputTokens,
    peakPromptTokens: res.peakPromptTokens,
    compactions: res.compactions + (res.toolCounts["compress"] ?? 0),
    cacheForkSummaries: res.cacheForkSummaries,
    cacheForkPromptTokens: res.cacheForkPromptTokens,
    cacheForkCachedTokens: res.cacheForkCachedTokens,
    boundaryFolds: res.boundaryFolds,
    preCacheProjections: res.preCacheProjections,
    recoveryQueries: recov,
    sidecarCostUsd: res.sidecarCostUsd,
    totalUnifiedCostUsd: res.totalUnifiedCostUsd,
    wallClockS: res.wallClockS,
  };
}

function mergeMetrics(list: Array<UnitCheckpoint["metrics"]>): UnitCheckpoint["metrics"] {
  return {
    turns: list.reduce((s, m) => s + m.turns, 0),
    promptTokens: list.reduce((s, m) => s + m.promptTokens, 0),
    cachedTokens: list.reduce((s, m) => s + m.cachedTokens, 0),
    uncachedTokens: list.reduce((s, m) => s + m.uncachedTokens, 0),
    outputTokens: list.reduce((s, m) => s + m.outputTokens, 0),
    peakPromptTokens: list.reduce((s, m) => Math.max(s, m.peakPromptTokens), 0),
    compactions: list.reduce((s, m) => s + m.compactions, 0),
    cacheForkSummaries: list.reduce((s, m) => s + m.cacheForkSummaries, 0),
    cacheForkPromptTokens: list.reduce((s, m) => s + m.cacheForkPromptTokens, 0),
    cacheForkCachedTokens: list.reduce((s, m) => s + m.cacheForkCachedTokens, 0),
    boundaryFolds: list.reduce((s, m) => s + m.boundaryFolds, 0),
    preCacheProjections: list.reduce((s, m) => s + m.preCacheProjections, 0),
    recoveryQueries: list.reduce((s, m) => s + m.recoveryQueries, 0),
    sidecarCostUsd: list.reduce((s, m) => s + m.sidecarCostUsd, 0),
    totalUnifiedCostUsd: list.reduce((s, m) => s + m.totalUnifiedCostUsd, 0),
    wallClockS: Number(list.reduce((s, m) => s + m.wallClockS, 0).toFixed(1)),
  };
}

// 1. LongMemEval-S Marathon Group (4 questions / ~450K tokens > 400K window)
async function runLmeGroup(opts: {
  groupIdx: number;
  items: LmeScaleItem[];
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  judgeClient: DeepSeekClient;
  outDir: string;
}): Promise<UnitCheckpoint> {
  const unitId = `lme-grp${opts.groupIdx + 1}`;
  const workDir = join(opts.outDir, "work", `${unitId}-${opts.arm}`);
  rmSync(workDir, { recursive: true, force: true });
  mkdirSync(workDir, { recursive: true });

  const sessionBlocks: string[] = [];
  for (let arcIdx = 0; arcIdx < opts.items.length; arcIdx++) {
    const it = opts.items[arcIdx];
    for (let si = 0; si < it.haystack_sessions.length; si++) {
      const date = it.haystack_dates?.[si] ?? `Session-${si + 1}`;
      const turns = it.haystack_sessions[si].map((t) => `[${t.role}]: ${t.content}`).join("\n");
      sessionBlocks.push(
        `--- ARC ${arcIdx + 1} (UserProfile-${arcIdx + 1}) | SESSION ${si + 1} (Date: ${date}) ---\n${turns}`,
      );
    }
  }
  const batches = packIntoChunks(sessionBlocks, 90_000);
  const streamFile = join(workDir, "stream.json");
  writeFileSync(streamFile, JSON.stringify({ title: unitId, chunks: batches }), "utf8");

  const numberedQs = opts.items
    .map(
      (it, i) =>
        `Q${i + 1} [Arc ${i + 1} / UserProfile-${i + 1}, Current Date: ${it.question_date ?? "unknown"}]: ${it.question}`,
    )
    .join("\n");

  const prompt = [
    `You are evaluated on LongMemEval-S Continuous Marathon #${opts.groupIdx + 1} (${batches.length} sequential batches across ${opts.items.length} independent user arcs: UserProfile-1..${opts.items.length}).`,
    `Step 1: Call \`read_next_batch\` (with count=2) repeatedly until \`END_OF_STREAM\` is reached.`,
    `Step 2: Once \`END_OF_STREAM\` is reached, use your context query/search tool if needed to verify exact dates, updates, or whether a fact was ever mentioned in that specific UserProfile's arc, then answer ALL ${opts.items.length} questions concisely in JSON:`,
    ``,
    numberedQs,
    ``,
    `End your final response with a JSON block in this exact format:`,
    "```json",
    `{"answers": [${opts.items.map((_, i) => `"answer to Q${i + 1}"`).join(", ")}]}`,
    "```",
  ].join("\n");

  const res = await runPiCli({
    arm: opts.arm,
    prompt,
    cwd: workDir,
    runWorkDir: workDir,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    streamChunksFile: streamFile,
    streamMaxBatchCount: 2,
    timeoutMs: 540_000,
    tag: `${unitId}/${opts.arm}`,
  });

  const parsed = parseJsonAnswersArray(res.finalAnswer);
  const subScores: UnitCheckpoint["subScores"] = [];
  for (let i = 0; i < opts.items.length; i++) {
    const it = opts.items[i];
    const pred = res.overflowOrError && !parsed[i] ? "" : (parsed[i] ?? res.finalAnswer);
    const ok = pred ? await judgeLongMemEval(opts.judgeClient, opts.model, it, pred) : false;
    const f1 = pred ? qaF1Score(pred, [it.answer]) : 0;
    subScores.push({
      subtask: it.category,
      primary: ok ? 1 : 0,
      secondary: f1,
    });
  }

  return {
    suite: "longmemeval",
    unitId,
    arm: opts.arm,
    subScores,
    overflowCount: res.overflowOrError ? 1 : 0,
    metrics: summarizeRunMetrics(res),
  };
}

// 2. MemoryAgentBench Episode (`CR`, `AR`, `LRU`, `TTL`)
async function runMabEpisodeUnit(opts: {
  ep: MabScaleEpisode;
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<UnitCheckpoint> {
  const { ep } = opts;
  const unitId = ep.id;
  const workDir = join(opts.outDir, "work", `${unitId}-${opts.arm}`);
  rmSync(workDir, { recursive: true, force: true });
  mkdirSync(workDir, { recursive: true });
  const streamFile = join(workDir, "stream.json");
  writeFileSync(streamFile, JSON.stringify({ title: ep.id, chunks: ep.chunks }), "utf8");

  const numberedQs = ep.questions.map((q, i) => `Q${i + 1}: ${q}`).join("\n");
  const splitGuidance =
    ep.shortSplit === "LRU"
      ? "Provide comprehensive, faithful plot/character summaries (around 600-1000 words) covering key events, relationships, and outcomes."
      : ep.shortSplit === "TTL"
        ? ep.source.toLowerCase().includes("redial")
          ? "For each user-system movie dialogue question, output a semicolon-separated Top-10 list of recommended movie titles (Recall@10) learned from the historical dialogues."
          : "For each classification query, output ONLY the exact integer `label` ID learned from the labeled examples in the stream (e.g. `\"28\"`)."
        : ep.shortSplit === "CR"
          ? "Counterfactual Conflict Resolution rule: rely ONLY on the provided numbered facts (ignoring real-world pretraining knowledge), and whenever multiple facts conflict for the same entity+relation, the LATER (higher-numbered) fact supersedes earlier ones!"
          : "Use your context query/search tool after END_OF_STREAM to verify exact verbatim names, dates, and numbers.";

  const batchCountHint = ep.totalTokensEst > 700_000 ? 3 : 2;
  const prompt = [
    `You are evaluated on MemoryAgentBench (${ep.split}, ~${Math.round(ep.totalTokensEst / 1000)}K tokens across ${ep.chunks.length} sequential batches).`,
    `Step 1: Call \`read_next_batch\` (with count=${batchCountHint}) repeatedly until \`END_OF_STREAM\` is reached.`,
    `Step 2: ${splitGuidance} Answer ALL ${ep.questions.length} questions in JSON:`,
    ``,
    numberedQs,
    ``,
    `End your final response with a JSON block in this exact format:`,
    "```json",
    `{"answers": [${ep.questions.map((_, i) => `"answer to Q${i + 1}"`).join(", ")}]}`,
    "```",
  ].join("\n");

  const res = await runPiCli({
    arm: opts.arm,
    prompt,
    cwd: workDir,
    runWorkDir: workDir,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    streamChunksFile: streamFile,
    streamMaxBatchCount: batchCountHint,
    timeoutMs: ep.totalTokensEst > 700_000 ? 780_000 : 540_000,
    tag: `${unitId}/${opts.arm}`,
  });

  const parsed = parseJsonAnswersArray(res.finalAnswer);
  const subScores: UnitCheckpoint["subScores"] = [];

  for (let qi = 0; qi < ep.questions.length; qi++) {
    const pred = res.overflowOrError && !parsed[qi] ? "" : (parsed[qi] ?? res.finalAnswer);
    const refs = ep.answers[qi];
    let primary = 0;
    let secondary = 0;
    if (ep.shortSplit === "LRU") {
      // BookSum / Long-Range Understanding: ROUGE-L F1 primary, QA Token F1 secondary
      primary = pred ? rougeLF1(pred, refs) : 0;
      secondary = pred ? qaF1Score(pred, refs) : 0;
    } else if (ep.shortSplit === "CR") {
      // Conflict Resolution: Max(Substring EM, Token F1) primary
      const em = bestSubstringMatch(pred, refs);
      const f1 = qaF1Score(pred, refs);
      primary = Math.max(em, f1);
      secondary = em;
    } else if (ep.shortSplit === "TTL" && !ep.source.toLowerCase().includes("redial")) {
      // Exact numeric label match (`label: 28` or `"28"` — avoid substring false-positive like "128" matching "28"!)
      const nums = (pred.match(/\b\d+\b/g) ?? []);
      const exactLabel = refs.some((r) => nums.includes(r.trim())) ? 1 : 0;
      primary = exactLabel;
      secondary = exactLabel;
    } else {
      // AR (Exact Match) & TTL ReDial (Recall@10 movie title match in predicted Top-10 list!)
      const em = bestSubstringMatch(pred, refs);
      const f1 = Math.max(em, qaF1Score(pred, refs));
      primary = em;
      secondary = f1;
    }
    subScores.push({ subtask: ep.shortSplit, primary, secondary });
  }

  return {
    suite: "memoryagentbench",
    unitId,
    arm: opts.arm,
    subScores,
    overflowCount: res.overflowOrError ? 1 : 0,
    metrics: summarizeRunMetrics(res),
  };
}

// 3. ContextBench 3-Issue Continuous Multi-Repo SWE Session (10 sessions × 3 issues = 30 issues)
async function runCbSessionUnit(opts: {
  sessionIdx: number;
  items: CbScaleItem[];
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<UnitCheckpoint> {
  const unitId = `cb-sess${opts.sessionIdx + 1}`;
  const sharedWorkDir = join(opts.outDir, "work", `${unitId}-${opts.arm}`);
  rmSync(sharedWorkDir, { recursive: true, force: true });
  mkdirSync(sharedWorkDir, { recursive: true });
  const sharedSessionFile = join(sharedWorkDir, "cb-session.jsonl");

  const subScores: UnitCheckpoint["subScores"] = [];
  const runMetricsList: Array<UnitCheckpoint["metrics"]> = [];
  let overflows = 0;

  for (let i = 0; i < opts.items.length; i++) {
    const item = opts.items[i];
    const slug = `${item.repo.replace(/\//g, "__")}-${item.base_commit.slice(0, 8)}`;
    const repoDir = join(DATA_ROOT, "repos", slug);
    const boundaryTag = i > 0 ? "[TASK_BOUNDARY] " : "";

    const prompt = [
      `${boundaryTag}[Task ${i + 1} of ${opts.items.length} in Continuous SWE Session]`,
      `Repository working directory: \`${repoDir}\` (\`${item.repo}\`).`,
      `Explore \`${repoDir}\` using read-only tools (\`grep\`, \`find\`, \`read\`, \`ls\`, \`bash\`) to locate ALL exact source files (relative to \`${repoDir}\`) and 1-indexed line ranges (\`start_line\`..\`end_line\`) needed to resolve this issue.`,
      `Do NOT modify any files. Finish within <= 12 tool turns.`,
      ``,
      `### Issue Description`,
      item.problem_statement,
      ``,
      `### Required Final Output Format`,
      `End your final response with a JSON block (paths relative to \`${repoDir}\`):`,
      "```json",
      `{"spans": [{"file": "relative/path/to/file.py", "start_line": 100, "end_line": 145}]}`,
      "```",
    ].join("\n");

    const res = await runPiCli({
      arm: opts.arm,
      prompt,
      cwd: repoDir,
      runWorkDir: sharedWorkDir,
      sessionFile: sharedSessionFile,
      windowTokens: opts.windowTokens,
      model: opts.model,
      apiKey: opts.apiKey,
      baseUrl: opts.baseUrl,
      builtinTools: "read,grep,find,ls,bash",
      timeoutMs: 270_000,
      tag: `${unitId}/${opts.arm}/${item.repo.split("/")[1]}`,
    });

    const normalizedAns = res.finalAnswer.split(`${repoDir}/`).join("");
    const sc = scoreContextBench(normalizedAns, item.gold_context);
    subScores.push({
      subtask: item.repo,
      primary: sc.lineF1,
      secondary: sc.fileF1,
    });
    if (res.overflowOrError) overflows += 1;
    runMetricsList.push(summarizeRunMetrics(res));
  }

  return {
    suite: "contextbench",
    unitId,
    arm: opts.arm,
    subScores,
    overflowCount: overflows,
    metrics: mergeMetrics(runMetricsList),
  };
}

// 4. Oversized-Tools (5 items) & Pollution-Boundary (5 items) = 10 items
async function runOversizedAndPollutionUnits(opts: {
  kind: "oversized" | "pollution";
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<UnitCheckpoint> {
  const unitId = opts.kind === "oversized" ? "spec-oversized" : "spec-pollution";
  const runDir = join(opts.outDir, "work", `${unitId}-${opts.arm}`);
  rmSync(runDir, { recursive: true, force: true });
  const wsDir = join(runDir, "workspace");
  mkdirSync(wsDir, { recursive: true });

  if (opts.kind === "oversized") {
    mkdirSync(join(wsDir, "services"), { recursive: true });
    const ciLines: string[] = ["=== CI Matrix Runner v4.8 ==="];
    for (let i = 1; i <= 3900; i++) {
      if (i === 46) {
        ciLines.push(
          'FAIL tests/auth/test_kdf_handshake.py::test_primary_salt_derivation - AssertionError: CHECKSUM_SALT_MISMATCH expected="v4-argon-9917" got="v1-md5-0042"',
        );
      } else {
        ciLines.push(`[2026-11-17T04:12:00Z] INFO shard-${(i % 7) + 2} :: test_routine_batch_${i} PASSED (12ms)`);
      }
    }
    const ciLogPath = join(wsDir, ".ci_raw.dat");
    writeFileSync(ciLogPath, ciLines.join("\n") + "\n", "utf8");
    const ciScript = join(wsDir, "run_ci_suite.sh");
    writeFileSync(ciScript, `#!/bin/sh\ncat "${ciLogPath}"\nexit 1\n`, "utf8");
    chmodSync(ciScript, 0o755);

    const pyLines: string[] = ['"""Settlement Engine."""', ""];
    for (let fnIdx = 1; fnIdx <= 420; fnIdx++) {
      if (pyLines.length >= 2858 && pyLines.length <= 2865) {
        while (pyLines.length < 2859) pyLines.push(`# pad ${pyLines.length + 1}`);
        pyLines.push("def reconcile_epoch_fee(gross_units: int) -> dict:");
        pyLines.push('    SETTLEMENT_CODE = "LEDGER-ZETA-841"');
        pyLines.push("    MAX_DRIFT_BPS = 37");
        pyLines.push('    return {"code": SETTLEMENT_CODE, "bps": MAX_DRIFT_BPS}');
        pyLines.push("");
      } else {
        pyLines.push(`def account_rule_${fnIdx}(b: int) -> int:\n    return (b + ${fnIdx * 17}) ^ ${(fnIdx * 31) % 255}\n    # rule ${fnIdx}\n    # audit ${10000 + fnIdx}\n`);
      }
    }
    writeFileSync(join(wsDir, "services", "ledger_engine.py"), pyLines.join("\n") + "\n", "utf8");

    const dumpLines: string[] = [];
    for (let r = 1; r <= 2600; r++) {
      if (r === 1142) {
        dumpLines.push(
          JSON.stringify({
            seq: r,
            level: "FATAL",
            event: "AssertionError_OOM_Kill",
            culprit_shard: "shard-eu-west-07",
            oom_peak_mb: 65408,
          }),
        );
      } else {
        dumpLines.push(
          JSON.stringify({
            seq: r,
            level: "INFO",
            event: "heartbeat_ok",
            shard: `shard-us-east-${(r % 16) + 1}`,
            rss_mb: 2048,
          }),
        );
      }
    }
    const telemetryDumpFile = join(runDir, "telemetry-dump.jsonl");
    writeFileSync(telemetryDumpFile, dumpLines.join("\n"), "utf8");

    const res = await runPiCli({
      arm: opts.arm,
      prompt: [
        `Investigate the release incident in \`${wsDir}\`:`,
        `1. Run \`./run_ci_suite.sh\` via \`bash\` to find \`expected_kdf_salt\`.`,
        `2. Inspect \`services/ledger_engine.py\` (starting with \`read\`) to find \`reconcile_epoch_fee\`'s \`SETTLEMENT_CODE\` and \`MAX_DRIFT_BPS\`.`,
        `3. Call \`fetch_telemetry_dump\` (\`cluster: "prod-eu-west"\`) to find the OOM \`culprit_shard\` and \`oom_peak_mb\`.`,
        `Return JSON: \`\`\`json\n{"expected_kdf_salt":"...","settlement_code":"...","max_drift_bps":"...","culprit_shard":"...","oom_peak_mb":"..."}\n\`\`\``,
      ].join("\n"),
      cwd: wsDir,
      runWorkDir: runDir,
      windowTokens: opts.windowTokens,
      model: opts.model,
      apiKey: opts.apiKey,
      baseUrl: opts.baseUrl,
      builtinTools: "read,grep,find,ls,bash",
      telemetryDumpFile,
      timeoutMs: 300_000,
      tag: `${unitId}/${opts.arm}`,
    });

    let obj: Record<string, unknown> = {};
    const jm = res.finalAnswer.match(/(\{[\s\S]*?"expected_kdf_salt"[\s\S]*?\})/);
    if (jm) {
      try {
        obj = JSON.parse(jm[1]);
      } catch {
        // ignore
      }
    }
    const facts = [
      { k: "expected_kdf_salt", r: ["v4-argon-9917"] },
      { k: "settlement_code", r: ["LEDGER-ZETA-841"] },
      { k: "max_drift_bps", r: ["37"] },
      { k: "culprit_shard", r: ["shard-eu-west-07"] },
      { k: "oom_peak_mb", r: ["65408"] },
    ];
    return {
      suite: "oversized-pollution",
      unitId,
      arm: opts.arm,
      subScores: facts.map((f) => {
        const v = obj[f.k] !== undefined ? String(obj[f.k]) : res.finalAnswer;
        const em = bestSubstringMatch(v, f.r);
        return { subtask: "OVERSIZED", primary: em, secondary: Math.max(em, qaF1Score(v, f.r)) };
      }),
      overflowCount: res.overflowOrError ? 1 : 0,
      metrics: summarizeRunMetrics(res),
    };
  }

  // Pollution-boundary (5 items)
  mkdirSync(join(wsDir, "docs"), { recursive: true });
  mkdirSync(join(wsDir, "inventory"), { recursive: true });
  const makeLegacy = (title: string) =>
    Array.from(
      { length: 115 },
      (_, s) =>
        `## ${title} #${s + 1}\nAll production nodes MUST bind \`gateway_port = 7071\`, \`aead_cipher = "chacha20-legacy-v1"\`, \`quorum_voters = 3\`, \`auth_header = "X-Cluster-Token-V1"\`.`,
    ).join("\n\n");
  writeFileSync(join(wsDir, "docs", "v1_gateway_rfc.md"), makeLegacy("Gateway RFC"), "utf8");
  writeFileSync(join(wsDir, "docs", "v1_crypto_rfc.md"), makeLegacy("Crypto RFC"), "utf8");
  writeFileSync(join(wsDir, "docs", "v1_quorum_rfc.md"), makeLegacy("Quorum RFC"), "utf8");
  const invLines = Array.from({ length: 620 }, (_, r) =>
    r === 317 ? 'PRIMARY_HSM_SERIAL="HSM-NITRO-5519"' : `RACK_${r}="blade-${r}.dc1"`,
  );
  writeFileSync(join(wsDir, "inventory", "datacenter_assets.env"), invLines.join("\n"), "utf8");

  const sharedSess = join(runDir, "pollution.jsonl");
  const r1 = await runPiCli({
    arm: opts.arm,
    prompt:
      "Read `docs/v1_gateway_rfc.md`, `docs/v1_crypto_rfc.md`, `docs/v1_quorum_rfc.md`, and `inventory/datacenter_assets.env` with `read`, then summarize the v1 deployment parameters and HSM serial.",
    cwd: wsDir,
    runWorkDir: runDir,
    sessionFile: sharedSess,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    builtinTools: "read,grep,find,ls,bash",
    timeoutMs: 240_000,
    tag: `${unitId}-p1/${opts.arm}`,
  });

  mkdirSync(join(wsDir, "specs"), { recursive: true });
  writeFileSync(
    join(wsDir, "specs", "v3_cutover_contract.md"),
    "# Sovereign v3\n- `gateway_port = 19443`\n- `aead_cipher = \"aes-256-gcm-siv-v3\"`\n- Check `v3_attestation_policy.md` and `v3_consensus_tiers.md`.",
    "utf8",
  );
  writeFileSync(
    join(wsDir, "specs", "v3_attestation_policy.md"),
    '# Attestation\n- `auth_header = "X-Sovereign-V3-Attest"`',
    "utf8",
  );
  writeFileSync(
    join(wsDir, "specs", "v3_consensus_tiers.md"),
    "# Consensus\n- Active Sovereign v3 Production Contract: `quorum_voters = 9`.",
    "utf8",
  );

  const r2 = await runPiCli({
    arm: opts.arm,
    prompt:
      'Ignore the previous hypothesis and discard all earlier legacy v1 conclusions — our initial v1 RFC direction was deprecated and wrong! Read `specs/` for the Sovereign v3 settings and include `PRIMARY_HSM_SERIAL` from the hardware inventory. Output JSON: ```json\n{"gateway_port":"...","aead_cipher":"...","quorum_voters":"...","auth_header":"...","primary_hsm_serial":"..."}\n```',
    cwd: wsDir,
    runWorkDir: runDir,
    sessionFile: sharedSess,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    builtinTools: "read,grep,find,ls,bash",
    timeoutMs: 240_000,
    tag: `${unitId}-p2/${opts.arm}`,
  });

  let obj: Record<string, unknown> = {};
  const jm = r2.finalAnswer.match(/(\{[\s\S]*?"gateway_port"[\s\S]*?\})/);
  if (jm) {
    try {
      obj = JSON.parse(jm[1]);
    } catch {
      // ignore
    }
  }
  const p2Saved =
    r1.peakPromptTokens > 0 ? (r1.peakPromptTokens - r2.peakPromptTokens) / r1.peakPromptTokens : 0;
  const facts = [
    { k: "gateway_port", r: ["19443"] },
    { k: "aead_cipher", r: ["aes-256-gcm-siv-v3"] },
    { k: "quorum_voters", r: ["9"] },
    { k: "auth_header", r: ["X-Sovereign-V3-Attest"] },
    { k: "primary_hsm_serial", r: ["HSM-NITRO-5519"] },
  ];
  return {
    suite: "oversized-pollution",
    unitId,
    arm: opts.arm,
    subScores: facts.map((f) => {
      const v = obj[f.k] !== undefined ? String(obj[f.k]) : r2.finalAnswer;
      return { subtask: "POLLUTION", primary: bestSubstringMatch(v, f.r), secondary: p2Saved };
    }),
    overflowCount: (r1.overflowOrError ? 1 : 0) + (r2.overflowOrError ? 1 : 0),
    metrics: mergeMetrics([summarizeRunMetrics(r1), summarizeRunMetrics(r2)]),
  };
}

// ---------------------------------------------------------------------------
// Main Orchestrator & Report Generator
// ---------------------------------------------------------------------------

function parseArg(flag: string, fallback: string): string {
  const idx = process.argv.indexOf(flag);
  if (idx !== -1 && idx + 1 < process.argv.length) return process.argv[idx + 1];
  return fallback;
}

async function main() {
  const armsArg = parseArg("--arms", "full,pi-native,billion-context,cpat");
  const suitesArg = parseArg("--suites", "all");
  const windowTokens = Math.max(32_000, Number(parseArg("--window", "400000")));
  const concurrency = Math.max(1, Number(parseArg("--concurrency", "6")));
  const model = parseArg("--model", "deepseek-flash");
  const resumeDir = parseArg("--resume", "");

  const arms = armsArg.split(",").map((s) => s.trim()) as PiArm[];
  const enabledSuites = new Set(
    suitesArg === "all"
      ? ["longmemeval", "memoryagentbench", "contextbench", "oversized-pollution"]
      : suitesArg.split(",").map((s) => s.trim()),
  );

  const env = loadDeepSeekEnv();
  const judgeClient = new DeepSeekClient(env);

  const outDir = resumeDir
    ? resolve(resumeDir)
    : resolve("runs", `pi-scale2-120-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  const ckptDir = join(outDir, "checkpoints");
  mkdirSync(ckptDir, { recursive: true });

  const lmeItems = JSON.parse(readFileSync(join(SCALE2_DIR, "lme_40.json"), "utf8")) as LmeScaleItem[];
  const mabEpisodes = JSON.parse(readFileSync(join(SCALE2_DIR, "mab_40.json"), "utf8")) as MabScaleEpisode[];
  const cbItems = JSON.parse(readFileSync(join(SCALE2_DIR, "cb_30.json"), "utf8")) as CbScaleItem[];

  type Job = { key: string; run: () => Promise<UnitCheckpoint> };
  const jobs: Job[] = [];

  // Interleave arms and suites so progress is balanced across all 4 arms!
  if (enabledSuites.has("oversized-pollution")) {
    for (const kind of ["oversized", "pollution"] as const) {
      for (const arm of arms) {
        const key = `spec-${kind}__${arm}`;
        jobs.push({
          key,
          run: () =>
            runOversizedAndPollutionUnits({
              kind,
              arm,
              windowTokens,
              model,
              apiKey: env.apiKey,
              baseUrl: env.baseUrl,
              outDir,
            }),
        });
      }
    }
  }

  if (enabledSuites.has("memoryagentbench")) {
    for (const ep of mabEpisodes) {
      for (const arm of arms) {
        const key = `${ep.id}__${arm}`;
        jobs.push({
          key,
          run: () =>
            runMabEpisodeUnit({
              ep,
              arm,
              windowTokens,
              model,
              apiKey: env.apiKey,
              baseUrl: env.baseUrl,
              outDir,
            }),
        });
      }
    }
  }

  if (enabledSuites.has("longmemeval")) {
    for (let g = 0; g < Math.ceil(lmeItems.length / 4); g++) {
      const slice = lmeItems.slice(g * 4, g * 4 + 4);
      for (const arm of arms) {
        const key = `lme-grp${g + 1}__${arm}`;
        jobs.push({
          key,
          run: () =>
            runLmeGroup({
              groupIdx: g,
              items: slice,
              arm,
              windowTokens,
              model,
              apiKey: env.apiKey,
              baseUrl: env.baseUrl,
              judgeClient,
              outDir,
            }),
        });
      }
    }
  }

  if (enabledSuites.has("contextbench")) {
    for (let s = 0; s < Math.ceil(cbItems.length / 3); s++) {
      const slice = cbItems.slice(s * 3, s * 3 + 3);
      for (const arm of arms) {
        const key = `cb-sess${s + 1}__${arm}`;
        jobs.push({
          key,
          run: () =>
            runCbSessionUnit({
              sessionIdx: s,
              items: slice,
              arm,
              windowTokens,
              model,
              apiKey: env.apiKey,
              baseUrl: env.baseUrl,
              outDir,
            }),
        });
      }
    }
  }

  console.log(`\n=== Scale-2 (120-Item × 4-Arm = 480 Evals) 400K Benchmark Suite ===`);
  console.log(`Jobs: ${jobs.length} unit-arm runs | Concurrency: ${concurrency} | Window: ${windowTokens.toLocaleString()} | Output: ${outDir}\n`);

  let completed = 0;
  const allUnits = await mapWithConcurrency(jobs, concurrency, async (job) => {
    const ckptFile = join(ckptDir, `${job.key}.json`);
    if (existsSync(ckptFile)) {
      const cached = JSON.parse(readFileSync(ckptFile, "utf8")) as UnitCheckpoint;
      completed += 1;
      console.log(`[${completed}/${jobs.length}] (cached) ${job.key}`);
      return cached;
    }
    const res = await job.run();
    if (res.metrics.promptTokens === 0 && res.overflowCount > 0) {
      throw new Error(
        `Fatal API / Quota error on ${job.key} (0 prompt tokens — e.g. HTTP 402 Insufficient Balance). Refusing to write corrupted checkpoint; top up API key and re-run with --resume ${outDir}`,
      );
    }
    writeFileSync(ckptFile, JSON.stringify(res, null, 2), "utf8");
    completed += 1;
    const avgPrim =
      res.subScores.length > 0
        ? ((res.subScores.reduce((s, x) => s + x.primary, 0) / res.subScores.length) * 100).toFixed(1)
        : "0.0";
    const hitPct =
      res.metrics.promptTokens > 0
        ? ((res.metrics.cachedTokens / res.metrics.promptTokens) * 100).toFixed(1)
        : "0.0";
    console.log(
      `[${completed}/${jobs.length}] DONE ${job.key.padEnd(28)} | score=${avgPrim}% | fail=${res.overflowCount} | prompt=${res.metrics.promptTokens.toLocaleString()} (hit ${hitPct}%) | fork=${res.metrics.cacheForkSummaries} | fold=${res.metrics.boundaryFolds} | $${res.metrics.totalUnifiedCostUsd.toFixed(4)} | ${res.metrics.wallClockS}s`,
    );
    return res;
  });

  // Aggregate per-suite & per-subtask summaries
  const suitesOrder: Array<UnitCheckpoint["suite"]> = [
    "longmemeval",
    "memoryagentbench",
    "contextbench",
    "oversized-pollution",
  ];

  const reportRows: any[] = [];
  for (const suite of suitesOrder) {
    if (!enabledSuites.has(suite)) continue;
    for (const arm of arms) {
      const units = allUnits.filter((u) => u.suite === suite && u.arm === arm);
      if (units.length === 0) continue;
      const flatSubs = units.flatMap((u) => u.subScores);
      const m = mergeMetrics(units.map((u) => u.metrics));
      const overflows = units.reduce((s, u) => s + u.overflowCount, 0);

      const bySub: Record<string, { n: number; pSum: number; sSum: number }> = {};
      for (const sc of flatSubs) {
        const cur = bySub[sc.subtask] ?? { n: 0, pSum: 0, sSum: 0 };
        cur.n += 1;
        cur.pSum += sc.primary;
        cur.sSum += sc.secondary;
        bySub[sc.subtask] = cur;
      }
      const subtaskScoresPct: Record<string, number> = {};
      for (const [k, v] of Object.entries(bySub)) {
        subtaskScoresPct[k] = Number(((v.pSum / Math.max(1, v.n)) * 100).toFixed(1));
      }

      // For MemoryAgentBench, primary score = 4-split macro average (matching official BC report!)
      let primaryPct =
        flatSubs.length > 0
          ? Number(((flatSubs.reduce((s, x) => s + x.primary, 0) / flatSubs.length) * 100).toFixed(1))
          : 0;
      if (suite === "memoryagentbench") {
        const splitKeys = ["CR", "AR", "LRU", "TTL"].filter((k) => subtaskScoresPct[k] !== undefined);
        if (splitKeys.length > 0) {
          primaryPct = Number(
            (splitKeys.reduce((s, k) => s + subtaskScoresPct[k], 0) / splitKeys.length).toFixed(1),
          );
        }
      }
      const secondaryPct =
        flatSubs.length > 0
          ? Number(((flatSubs.reduce((s, x) => s + x.secondary, 0) / flatSubs.length) * 100).toFixed(1))
          : 0;

      reportRows.push({
        suite,
        arm,
        items: flatSubs.length,
        primary_pct: primaryPct,
        secondary_pct: secondaryPct,
        subtasks: subtaskScoresPct,
        overflows,
        turns: m.turns,
        prompt_tokens: m.promptTokens,
        cache_hit_pct: m.promptTokens > 0 ? Number(((m.cachedTokens / m.promptTokens) * 100).toFixed(1)) : 0,
        peak_window: m.peakPromptTokens,
        cache_fork_summaries: m.cacheForkSummaries,
        cache_fork_hit_pct:
          m.cacheForkPromptTokens > 0
            ? Number(((m.cacheForkCachedTokens / m.cacheForkPromptTokens) * 100).toFixed(1))
            : 0,
        boundary_folds: m.boundaryFolds,
        pre_cache_projections: m.preCacheProjections,
        recovery_queries: m.recoveryQueries,
        sidecar_usd: Number(m.sidecarCostUsd.toFixed(4)),
        total_unified_usd: Number(m.totalUnifiedCostUsd.toFixed(4)),
        wall_clock_s: m.wallClockS,
      });
    }
  }

  writeFileSync(join(outDir, "summary.json"), JSON.stringify(reportRows, null, 2), "utf8");

  const md = [
    `# Scale-2 (120-Item Stratified) 400K-Window Benchmark Summary`,
    ``,
    `| Suite | Arm | Items | Primary Score | Secondary | Subtask Breakdown | Fail | Prompt Tok | Cache Hit | Peak Win | Fork Sum (Hit%) | Folds | PreCache | Sidecar $ | Total $ |`,
    `|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|`,
    ...reportRows.map((r) => {
      const subStr =
        r.suite === "contextbench"
          ? `${Object.keys(r.subtasks).length} repos`
          : Object.entries(r.subtasks)
              .map(([k, v]) => `${k}:${v}%`)
              .join(" ");
      return `| ${r.suite} | \`${r.arm}\` | ${r.items} | **${r.primary_pct}%** | ${r.secondary_pct}% | ${subStr} | ${r.overflows} | ${r.prompt_tokens.toLocaleString()} | ${r.cache_hit_pct}% | ${r.peak_window.toLocaleString()} | ${r.cache_fork_summaries} (${r.cache_fork_hit_pct}%) | ${r.boundary_folds} | ${r.pre_cache_projections} | $${r.sidecar_usd.toFixed(4)} | **$${r.total_unified_usd.toFixed(4)}** |`;
    }),
  ].join("\n");

  writeFileSync(join(outDir, "summary.md"), md + "\n", "utf8");
  console.log(`\n${md}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
