/**
 * Pi CLI × Billion-Context 400K-Window Benchmark Suite (`npm run bench:pi`).
 *
 * Evaluates 4 real Pi CLI (`@earendil-works/pi-coding-agent` v1.5.2) arms
 * side-by-side on `deepseek/deepseek-flash` with a realistic **400,000-token (400K)**
 * context window across the three benchmarks in `billion-context/benchmarks/third-party-1500`:
 *
 *   Arms (all configured at 400,000-token context window):
 *     1. `full`            — Uncompacted 400K hard-window baseline (`pi-hard-window.ts`)
 *     2. `pi-native`       — Stock Pi CLI 400K built-in threshold auto-compaction (`reserveTokens=16384`)
 *     3. `billion-context` — Stock `billion-context` (`v0.1.216`, `modelContextLimit=400000`, `nudgeGrowthTokens=50000`)
 *     4. `cpat`            — CPAT ContextHarness Pi plugin (`extensions/pi-cpat.ts`, `CPAT_MAX_CONTEXT=400000`)
 *
 *   400K-Scale Benchmarks:
 *     1. `memoryagentbench` — ~480K-token (>400K!) RULER Accurate Needle Retrieval (`AR`) stream (8 Qs)
 *     2. `longmemeval`      — ~510K-token (>400K!) Continuous Multi-Session Companion Marathon (205 sessions, 4 Qs)
 *     3. `contextbench`     — Continuous Multi-Repo SWE Coding Session (`astropy` -> `django` -> `sympy`) in Pi CLI
 *
 *   Full-Cost Attribution (`arXiv:2609.22114` / `third-party-1500` unified pricing):
 *     - Uncached Input: $0.15 / 1M tokens | Prefix Cache Hit: $0.003 / 1M tokens | Output: $0.60 / 1M tokens
 */

import { execFileSync, spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { CpatPiSessionMetrics } from "../extensions/pi-cpat.ts";
import { DeepSeekClient } from "../src/deepseek/client.ts";
import { loadDeepSeekEnv } from "../src/util/env.ts";
import { estimateTokens } from "../src/util/misc.ts";
import { normalizeAnswer, qaF1Score } from "./f1.ts";

export type PiArm = "full" | "pi-native" | "billion-context" | "cpat";
export type BenchSuite =
  | "oversized-tools"
  | "pollution-boundary"
  | "memoryagentbench"
  | "longmemeval"
  | "contextbench";

const DEFAULT_WINDOW_TOKENS = 400_000;

const PRICE_MISS_PER_M = 0.15;
const PRICE_HIT_PER_M = 0.003;
const PRICE_OUT_PER_M = 0.6;

const DATA_CACHE_DIR = resolve("data/public");
const CPAT_EXT_PATH = resolve("extensions/pi-cpat.ts");
const FEEDER_EXT_PATH = resolve("bench/pi-stream-feeder.ts");
const TELEMETRY_EXT_PATH = resolve("bench/pi-telemetry-tool.ts");
const HARD_WIN_EXT_PATH = resolve("bench/pi-hard-window.ts");
const BC_EXT_PATH = resolve("data/public/bc-npm/node_modules/billion-context/dist/agent/billion-context.js");

interface ContextBenchItem {
  instance_id: string;
  original_inst_id: string;
  repo: string;
  repo_url: string;
  base_commit: string;
  problem_statement: string;
  gold_context: Array<{ file: string; start_line: number; end_line: number }>;
}

interface Mab400kEpisode {
  id: string;
  totalTokensEst: number;
  chunks: string[];
  questions: string[];
  answers: string[][];
}

interface LongMemEvalItem {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  question_date?: string;
  haystack_dates?: string[];
  haystack_sessions: Array<Array<{ role: string; content: string }>>;
}

interface PiInvocationMetrics {
  turns: number;
  promptTokens: number;
  cachedTokens: number;
  uncachedTokens: number;
  outputTokens: number;
  peakPromptTokens: number;
  compactions: number;
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

interface SuiteArmSummary {
  benchmark: BenchSuite;
  arm: PiArm;
  window_tokens: number;
  corpus_tokens_est: number;
  items: number;
  primary_metric: string;
  primary_score_pct: number;
  secondary_metric: string;
  secondary_score_pct: number;
  overflow_or_fail_count: number;
  avg_turns: number;
  total_prompt_tokens: number;
  cache_hit_pct: number;
  avg_peak_window: number;
  compress_or_edit_ops: number;
  recovery_queries: number;
  sidecar_usd: number;
  total_unified_usd: number;
  wall_clock_s: number;
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

// ---------------------------------------------------------------------------
// Core Pi CLI Process Runner (`pi --mode json -p ...`)
// ---------------------------------------------------------------------------

async function runPiCli(opts: {
  arm: PiArm;
  prompt: string;
  cwd: string;
  runWorkDir: string;
  sessionFile?: string; // optional shared session file for multi-issue marathons
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  builtinTools?: string;
  streamChunksFile?: string;
  telemetryDumpFile?: string;
  timeoutMs?: number;
  tag: string;
}): Promise<PiInvocationMetrics> {
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

  const effectiveWin = opts.windowTokens; // 400,000 across ALL arms!

  // Stock Pi defaults (reserveTokens: 16384, keepRecentTokens: 20000) at 400K!
  writeFileSync(
    join(piAgentDir, "models.json"),
    JSON.stringify(
      {
        providers: {
          deepseek: {
            baseUrl: opts.baseUrl,
            modelOverrides: {
              [opts.model]: {
                // For `full`, let provider catalog allow up to effectiveWin + guard via pi-hard-window.ts
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
    JSON.stringify(
      {
        reserveTokens: 16_384,
        keepRecentTokens: 20_000,
      },
      null,
      2,
    ),
    "utf8",
  );

  // Stock Billion-Context config at 400K window (`modelContextLimit: 400000`, stock `maxContextLimit: "75%"`, stock `nudgeGrowthTokens: 50000`)
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
            pi: {
              inline: true,
              subagents: false,
              commands: false,
            },
            guardrails: {
              maxOutputBytes: 0,
            },
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
  // Capture pre-invocation sidecar totals when sharing `runWorkDir` across multi-step marathon prompts
  let prevSidecarCalls = 0;
  let prevSidecarIn = 0;
  let prevSidecarCached = 0;
  let prevSidecarOut = 0;
  let prevEdits = 0;
  let prevPreCacheProj = 0;
  if (opts.arm === "cpat" && existsSync(cpatMetricsFile)) {
    try {
      const prev = JSON.parse(readFileSync(cpatMetricsFile, "utf8")) as CpatPiSessionMetrics;
      prevSidecarCalls = prev.sidecarCalls;
      prevSidecarIn = prev.sidecarInputTokens;
      prevSidecarCached = prev.sidecarCachedTokens;
      prevSidecarOut = prev.sidecarOutputTokens;
      prevEdits = prev.contextEditsEmitted;
      prevPreCacheProj = prev.preCacheToolProjections ?? 0;
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

  if (opts.builtinTools) {
    args.push("--tools", opts.builtinTools);
  } else {
    args.push("--no-builtin-tools");
  }

  if (opts.streamChunksFile) {
    args.push("-e", FEEDER_EXT_PATH);
  }
  if (opts.telemetryDumpFile) {
    args.push("-e", TELEMETRY_EXT_PATH);
  }
  if (opts.arm === "full") {
    args.push("-e", HARD_WIN_EXT_PATH);
  } else if (opts.arm === "billion-context") {
    args.push("-e", BC_EXT_PATH);
  } else if (opts.arm === "cpat") {
    args.push("-e", CPAT_EXT_PATH);
  }

  args.push("--", opts.prompt);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    DEEPSEEK_API_KEY: opts.apiKey,
    PI_CODING_AGENT_DIR: piAgentDir,
    HOME: piHomeDir,
    PI_OFFLINE: "1",
    PI_HARD_WINDOW_TOKENS: String(effectiveWin),
    ...(opts.streamChunksFile ? { PI_STREAM_CHUNKS_FILE: opts.streamChunksFile } : {}),
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
  const toolCounts: Record<string, number> = {};
  let overflowOrError = false;
  let errorMessage: string | undefined;
  let finalAnswer = "";
  const rawJsonlLines: string[] = [];

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
        errorMessage = `Timeout after ${Math.round((opts.timeoutMs ?? 420_000) / 1000)}s`;
        child.kill("SIGKILL");
      }
    }, opts.timeoutMs ?? 420_000);

    const handleRecord = (line: string) => {
      const trimmed = line.replace(/\r$/, "").trim();
      if (!trimmed.startsWith("{")) return;
      rawJsonlLines.push(trimmed);
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
        const toolCallsInTurn = Array.isArray(msg.content)
          ? msg.content
              .filter((p: any) => p?.type === "toolCall" && typeof p.name === "string")
              .map((p: any) => p.name)
          : [];

        if (textParts.trim()) {
          finalAnswer = textParts.trim();
        }
        if (msg.stopReason === "error" || msg.stopReason === "aborted") {
          overflowOrError = true;
          errorMessage = String(msg.errorMessage ?? `stopReason=${msg.stopReason}`);
        }

        const hitPct = turnPrompt > 0 ? Math.round((cIn / turnPrompt) * 100) : 0;
        console.log(
          `  [${opts.tag}] t${String(turns).padStart(2)}: ${turnPrompt.toLocaleString()}in(${hitPct}%hit)/${out}out -> ${toolCallsInTurn.join(",") || (msg.stopReason === "error" ? `ERR:${errorMessage?.slice(0, 50)}` : "ANS")}`,
        );
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
      writeFileSync(
        join(opts.runWorkDir, `pi-events-${Date.now()}.jsonl`),
        rawJsonlLines.join("\n") + "\n",
        "utf8",
      );
      if (stderrBuf.trim()) {
        writeFileSync(join(opts.runWorkDir, "pi-stderr.log"), stderrBuf, "utf8");
        if (stderrBuf.includes("ContextWindowExceededError")) {
          overflowOrError = true;
          errorMessage = "Exceeded 400K context window";
          console.log(`  [${opts.tag}] OVERFLOW: Exceeded ${Math.round(effectiveWin / 1000)}K hard context window!`);
        }
      }
      if (code !== 0 && !finalAnswer) {
        overflowOrError = true;
        errorMessage = errorMessage ?? `pi exited with code ${code}: ${stderrBuf.slice(-200)}`;
      }
      resolveDone();
    });
  });

  if (opts.arm === "cpat" && existsSync(cpatMetricsFile)) {
    try {
      const m = JSON.parse(readFileSync(cpatMetricsFile, "utf8")) as CpatPiSessionMetrics;
      sidecarCalls = Math.max(0, m.sidecarCalls - prevSidecarCalls);
      sidecarPromptTokens = Math.max(0, m.sidecarInputTokens - prevSidecarIn);
      sidecarCachedTokens = Math.max(0, m.sidecarCachedTokens - prevSidecarCached);
      sidecarOutputTokens = Math.max(0, m.sidecarOutputTokens - prevSidecarOut);
      compactions +=
        Math.max(0, m.contextEditsEmitted - prevEdits) +
        Math.max(0, (m.preCacheToolProjections ?? 0) - prevPreCacheProj);
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
// 1. MemoryAgentBench 400K+ (`AR` ~480K-token RULER Needle-in-a-Haystack)
// ---------------------------------------------------------------------------

function packIntoChunks(docs: string[], targetChars: number): string[] {
  const chunks: string[] = [];
  let cur: string[] = [];
  let curLen = 0;
  for (const d of docs) {
    if (cur.length > 0 && curLen + d.length > targetChars) {
      chunks.push(cur.join("\n\n"));
      cur = [];
      curLen = 0;
    }
    cur.push(d);
    curLen += d.length;
  }
  if (cur.length > 0) chunks.push(cur.join("\n\n"));
  return chunks;
}

async function loadMab400kEpisode(): Promise<Mab400kEpisode> {
  mkdirSync(DATA_CACHE_DIR, { recursive: true });
  const cachePath = join(DATA_CACHE_DIR, "mab_ar_480k_episode.json");
  if (existsSync(cachePath)) {
    return JSON.parse(readFileSync(cachePath, "utf8")) as Mab400kEpisode;
  }

  console.log("  [fetch] Building ~480K-token MemoryAgentBench AR episode from Hugging Face...");
  const allDocs: string[] = [];
  const pickedQs: string[] = [];
  const pickedAns: string[][] = [];
  let totalChars = 0;

  for (const offset of [0, 1, 2]) {
    if (totalChars >= 1_620_000) break; // ~485K tokens
    const url = `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Accurate_Retrieval&offset=${offset}&length=1`;
    const res = await fetch(url);
    if (!res.ok) continue;
    const data = (await res.json()) as { rows?: Array<{ row: Record<string, unknown> }> };
    const row = data.rows?.[0]?.row;
    if (!row) continue;

    const fullContext = String(row.context ?? "");
    const qs = (row.questions as string[]) ?? [];
    const ans = (row.answers as Array<string[] | string>) ?? [];

    const docs = fullContext
      .split(/(?=\nDocument \d+:)/g)
      .map((s) => s.trim())
      .filter(Boolean);

    // Pick 2-3 verified questions from this row whose gold evidence sits in `docs`
    let rowPicked = 0;
    for (let qi = 0; qi < qs.length && rowPicked < 3 && pickedQs.length < 8; qi++) {
      const refs = Array.isArray(ans[qi]) ? ans[qi].map(String) : [String(ans[qi] ?? "")];
      const ref0 = refs[0]?.trim();
      if (!ref0 || ref0.length < 3) continue;
      if (fullContext.toLowerCase().includes(ref0.toLowerCase())) {
        pickedQs.push(qs[qi]);
        pickedAns.push(refs);
        rowPicked += 1;
      }
    }

    for (const d of docs) {
      allDocs.push(`[Row ${offset + 1}] ${d}`);
      totalChars += d.length;
      if (totalChars >= 1_650_000) break;
    }
  }

  // Pack ~1.62M chars (~485K tokens) into 18 sequential batches (~27K tokens per batch)
  const chunks = packIntoChunks(allDocs, 92_000);
  const totalTokensEst = chunks.reduce((s, c) => s + estimateTokens(c), 0);

  const ep: Mab400kEpisode = {
    id: "MAB_AR_480K",
    totalTokensEst,
    chunks,
    questions: pickedQs,
    answers: pickedAns,
  };
  writeFileSync(cachePath, JSON.stringify(ep, null, 2), "utf8");
  console.log(
    `  [fetch] Cached ${ep.id}: ${chunks.length} batches, ~${totalTokensEst.toLocaleString()} tokens, ${pickedQs.length} gold needle Qs.`,
  );
  return ep;
}

async function runMab400kSuite(opts: {
  episode: Mab400kEpisode;
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<SuiteArmSummary> {
  const ep = opts.episode;
  const epWorkDir = join(opts.outDir, `mab400k-${opts.arm}`);
  mkdirSync(epWorkDir, { recursive: true });
  const streamFile = join(epWorkDir, "stream.json");
  writeFileSync(streamFile, JSON.stringify({ title: ep.id, chunks: ep.chunks }), "utf8");

  const numberedQs = ep.questions.map((q, i) => `Q${i + 1}: ${q}`).join("\n");
  const prompt = [
    `You are evaluated on MemoryAgentBench Accurate_Retrieval (~${Math.round(ep.totalTokensEst / 1000)}K tokens across ${ep.chunks.length} sequential batches, inside a ${Math.round(opts.windowTokens / 1000)}K context window).`,
    `Step 1: Call \`read_next_batch\` (with count=2) repeatedly until \`END_OF_STREAM\` is reached (${ep.chunks.length} total batches). Batches are one-pass and cannot be re-pulled from the stream once read.`,
    `Step 2: Once \`END_OF_STREAM\` is reached, use your context search/query/decompress tool if needed to verify exact verbatim names/dates/numbers from earlier batches, then answer ALL ${ep.questions.length} questions concisely in JSON:`,
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
    cwd: epWorkDir,
    runWorkDir: epWorkDir,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    streamChunksFile: streamFile,
    timeoutMs: 540_000,
    tag: `MAB-480K/${opts.arm}`,
  });

  let parsedAnswers: string[] = [];
  const jm =
    res.finalAnswer.match(/```(?:json)?\s*(\{[\s\S]*?"answers"[\s\S]*?\})\s*```/i) ??
    res.finalAnswer.match(/(\{[\s\S]*?"answers"\s*:\s*\[[\s\S]*?\][\s\S]*?\})/);
  if (jm) {
    try {
      const obj = JSON.parse(jm[1]) as { answers?: unknown[] };
      if (Array.isArray(obj.answers)) parsedAnswers = obj.answers.map((x) => String(x ?? ""));
    } catch {
      // fallback below
    }
  }

  let emSum = 0;
  let f1Sum = 0;
  const qResults = ep.questions.map((q, qi) => {
    const pred = res.overflowOrError && !parsedAnswers[qi] ? "" : (parsedAnswers[qi] ?? res.finalAnswer);
    const refs = ep.answers[qi];
    const em = bestSubstringMatch(pred, refs);
    const f1 = Math.max(em, qaF1Score(pred, refs));
    emSum += em;
    f1Sum += f1;
    return { question: q, refs, pred, em, f1 };
  });

  writeFileSync(
    join(opts.outDir, `mab400k-${opts.arm}.json`),
    JSON.stringify({ episode: ep.id, corpus_tokens_est: ep.totalTokensEst, qResults, metrics: res }, null, 2),
  );

  const qDenom = Math.max(1, ep.questions.length);
  const ops = res.compactions + (res.toolCounts["compress"] ?? 0);
  const recov =
    (res.toolCounts["context_query"] ?? 0) +
    (res.toolCounts["search_context"] ?? 0) +
    (res.toolCounts["decompress"] ?? 0);

  return {
    benchmark: "memoryagentbench",
    arm: opts.arm,
    window_tokens: opts.windowTokens,
    corpus_tokens_est: ep.totalTokensEst,
    items: ep.questions.length,
    primary_metric: "AR Needle EM",
    primary_score_pct: Number(((emSum / qDenom) * 100).toFixed(1)),
    secondary_metric: "Max(EM,F1)",
    secondary_score_pct: Number(((f1Sum / qDenom) * 100).toFixed(1)),
    overflow_or_fail_count: res.overflowOrError ? 1 : 0,
    avg_turns: res.turns,
    total_prompt_tokens: res.promptTokens,
    cache_hit_pct: res.promptTokens > 0 ? Number(((res.cachedTokens / res.promptTokens) * 100).toFixed(1)) : 0,
    avg_peak_window: res.peakPromptTokens,
    compress_or_edit_ops: ops,
    recovery_queries: recov,
    sidecar_usd: Number(res.sidecarCostUsd.toFixed(4)),
    total_unified_usd: Number(res.totalUnifiedCostUsd.toFixed(4)),
    wall_clock_s: res.wallClockS,
  };
}

// ---------------------------------------------------------------------------
// 2. LongMemEval-S 400K+ (~510K-token Continuous Multi-Session Marathon)
// ---------------------------------------------------------------------------

async function judgeLongMemEval(
  client: DeepSeekClient,
  model: string,
  question: string,
  gold: string,
  prediction: string,
): Promise<boolean> {
  if (!prediction.trim()) return false;
  if (bestSubstringMatch(prediction, [gold]) === 1) return true;
  try {
    const resp = await client.chat({
      model,
      messages: [
        {
          role: "system",
          content:
            'You are an exact QA evaluator for LongMemEval. Compare the predicted answer against the ground-truth answer for the given question. Return ONLY JSON: {"correct": true} if the prediction accurately contains the ground-truth fact/value/entity (allowing equivalent phrasing), otherwise {"correct": false}.',
        },
        {
          role: "user",
          content: `Question: ${question}\nGround Truth: ${gold}\nPrediction: ${prediction}`,
        },
      ],
      temperature: 0,
      maxTokens: 400,
      reasoningEffort: "low",
      responseFormat: { type: "json_object" },
    });
    const obj = JSON.parse(resp.message.content ?? "{}") as { correct?: boolean };
    return Boolean(obj.correct);
  } catch {
    return false;
  }
}

async function runLongMem400kSuite(opts: {
  items: LongMemEvalItem[];
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  judgeClient: DeepSeekClient;
  outDir: string;
}): Promise<SuiteArmSummary> {
  // Combine 4 LongMemEval-S histories (~205 chronological sessions ≈ 510K tokens > 400K window!)
  const selected = opts.items.slice(0, 4);
  const sessionBlocks: string[] = [];
  for (let arcIdx = 0; arcIdx < selected.length; arcIdx++) {
    const item = selected[arcIdx];
    for (let si = 0; si < item.haystack_sessions.length; si++) {
      const date = item.haystack_dates?.[si] ?? `Session-${si + 1}`;
      const turns = item.haystack_sessions[si]
        .map((t) => `[${t.role}]: ${t.content}`)
        .join("\n");
      sessionBlocks.push(
        `--- ARC ${arcIdx + 1} (UserProfile-${arcIdx + 1}) | SESSION ${si + 1} (Date: ${date}) ---\n${turns}`,
      );
    }
  }

  const batches = packIntoChunks(sessionBlocks, 95_000); // ~18 batches of ~28K tokens ≈ 510K tokens
  const corpusTokensEst = batches.reduce((s, b) => s + estimateTokens(b), 0);

  const workDir = join(opts.outDir, `lme400k-${opts.arm}`);
  mkdirSync(workDir, { recursive: true });
  const streamFile = join(workDir, "stream.json");
  writeFileSync(streamFile, JSON.stringify({ title: "LME_510K_MARATHON", chunks: batches }), "utf8");

  const numberedQs = selected
    .map(
      (it, i) =>
        `Q${i + 1} [Arc ${i + 1} / UserProfile-${i + 1}, Current Date: ${it.question_date ?? "unknown"}]: ${it.question}`,
    )
    .join("\n");

  const prompt = [
    `You are evaluated on a LongMemEval-S Continuous Marathon (~${Math.round(corpusTokensEst / 1000)}K tokens across 4 chronological user arcs / 205 chat sessions, packed into ${batches.length} sequential batches inside a ${Math.round(opts.windowTokens / 1000)}K context window).`,
    `Step 1: Call \`read_next_batch\` (with count=2) repeatedly until \`END_OF_STREAM\` is reached so every arc is ingested.`,
    `Step 2: Once \`END_OF_STREAM\` is reached, use your context search/query/decompress tool if needed to verify exact dates and details from early arcs (Arc 1..4), then answer ALL ${selected.length} questions concisely in JSON:`,
    ``,
    numberedQs,
    ``,
    `End your final response with a JSON block in this exact format:`,
    "```json",
    `{"answers": ["answer to Q1", "answer to Q2", "answer to Q3", "answer to Q4"]}`,
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
    timeoutMs: 540_000,
    tag: `LME-510K/${opts.arm}`,
  });

  let parsedAnswers: string[] = [];
  const jm =
    res.finalAnswer.match(/```(?:json)?\s*(\{[\s\S]*?"answers"[\s\S]*?\})\s*```/i) ??
    res.finalAnswer.match(/(\{[\s\S]*?"answers"\s*:\s*\[[\s\S]*?\][\s\S]*?\})/);
  if (jm) {
    try {
      const obj = JSON.parse(jm[1]) as { answers?: unknown[] };
      if (Array.isArray(obj.answers)) parsedAnswers = obj.answers.map((x) => String(x ?? ""));
    } catch {
      // fallback below
    }
  }

  let judgeHits = 0;
  let f1Sum = 0;
  const qDetails = [];
  for (let i = 0; i < selected.length; i++) {
    const it = selected[i];
    const pred = res.overflowOrError && !parsedAnswers[i] ? "" : (parsedAnswers[i] ?? res.finalAnswer);
    const ok = pred ? await judgeLongMemEval(opts.judgeClient, opts.model, it.question, it.answer, pred) : false;
    const f1 = pred ? qaF1Score(pred, [it.answer]) : 0;
    if (ok) judgeHits += 1;
    f1Sum += f1;
    qDetails.push({
      arc: i + 1,
      question_id: it.question_id,
      question: it.question,
      gold: it.answer,
      pred,
      judge_correct: ok,
      f1,
    });
  }

  writeFileSync(
    join(opts.outDir, `longmemeval400k-${opts.arm}.json`),
    JSON.stringify({ corpus_tokens_est: corpusTokensEst, qDetails, metrics: res }, null, 2),
  );

  const denom = Math.max(1, selected.length);
  const ops = res.compactions + (res.toolCounts["compress"] ?? 0);
  const recov =
    (res.toolCounts["context_query"] ?? 0) +
    (res.toolCounts["search_context"] ?? 0) +
    (res.toolCounts["decompress"] ?? 0);

  return {
    benchmark: "longmemeval",
    arm: opts.arm,
    window_tokens: opts.windowTokens,
    corpus_tokens_est: corpusTokensEst,
    items: selected.length,
    primary_metric: "Judge Acc",
    primary_score_pct: Number(((judgeHits / denom) * 100).toFixed(1)),
    secondary_metric: "Token F1",
    secondary_score_pct: Number(((f1Sum / denom) * 100).toFixed(1)),
    overflow_or_fail_count: res.overflowOrError ? 1 : 0,
    avg_turns: res.turns,
    total_prompt_tokens: res.promptTokens,
    cache_hit_pct: res.promptTokens > 0 ? Number(((res.cachedTokens / res.promptTokens) * 100).toFixed(1)) : 0,
    avg_peak_window: res.peakPromptTokens,
    compress_or_edit_ops: ops,
    recovery_queries: recov,
    sidecar_usd: Number(res.sidecarCostUsd.toFixed(4)),
    total_unified_usd: Number(res.totalUnifiedCostUsd.toFixed(4)),
    wall_clock_s: res.wallClockS,
  };
}

// ---------------------------------------------------------------------------
// 3. ContextBench 400K (Continuous Multi-Repo SWE Coding Session in Pi CLI)
// ---------------------------------------------------------------------------

function ensureRepoCheckout(item: ContextBenchItem): string {
  const repoCacheRoot = join(DATA_CACHE_DIR, "repos");
  mkdirSync(repoCacheRoot, { recursive: true });
  const slug = `${item.repo.replace(/\//g, "__")}-${item.base_commit.slice(0, 8)}`;
  const targetDir = join(repoCacheRoot, slug);
  if (existsSync(join(targetDir, ".git"))) return targetDir;
  console.log(`  [git] Shallow-fetching ${item.repo} @ ${item.base_commit.slice(0, 8)}...`);
  mkdirSync(targetDir, { recursive: true });
  execFileSync("git", ["init", "-q"], { cwd: targetDir });
  execFileSync("git", ["remote", "add", "origin", item.repo_url], { cwd: targetDir });
  execFileSync("git", ["fetch", "-q", "--depth", "1", "origin", item.base_commit], {
    cwd: targetDir,
    timeout: 120_000,
  });
  execFileSync("git", ["checkout", "-q", "--detach", "FETCH_HEAD"], { cwd: targetDir });
  return targetDir;
}

function scoreContextBench(
  answer: string,
  gold: ContextBenchItem["gold_context"],
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

async function runContextBench400kMarathon(opts: {
  items: ContextBenchItem[];
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<SuiteArmSummary> {
  // Run a continuous multi-issue SWE session sharing ONE `session.jsonl` in Pi CLI!
  const sharedWorkDir = join(opts.outDir, `cb400k-${opts.arm}`);
  mkdirSync(sharedWorkDir, { recursive: true });
  const sharedSessionFile = join(sharedWorkDir, "marathon-session.jsonl");

  let lineF1Sum = 0;
  let fileF1Sum = 0;
  let failCount = 0;
  let totalTurns = 0;
  let totalPrompt = 0;
  let totalCached = 0;
  let maxPeak = 0;
  let opsTotal = 0;
  let recovTotal = 0;
  let sidecarUsd = 0;
  let totalUsd = 0;
  const t0 = Date.now();
  const details: unknown[] = [];

  for (let idx = 0; idx < opts.items.length; idx++) {
    const item = opts.items[idx];
    const repoDir = ensureRepoCheckout(item);
    const prompt = [
      `[Task ${idx + 1} of ${opts.items.length} in Continuous SWE Session]`,
      `Repository working directory: \`${repoDir}\` (\`${item.repo}\`).`,
      `Explore \`${repoDir}\` using read-only tools (\`grep\`, \`find\`, \`read\`, \`ls\`, \`bash\`) to locate ALL exact source files (relative to \`${repoDir}\`) and 1-indexed line ranges (\`start_line\`..\`end_line\`) needed to resolve this issue.`,
      `Do NOT modify any files. Finish within <= 14 tool turns.`,
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
      timeoutMs: 300_000,
      tag: `CB-400K/${opts.arm}/#${idx + 1}-${item.repo.split("/")[1]}`,
    });

    // Normalize any accidentally prefixed repoDir paths in finalAnswer before scoring
    const normalizedAns = res.finalAnswer.split(`${repoDir}/`).join("");
    const sc = scoreContextBench(normalizedAns, item.gold_context);
    lineF1Sum += sc.lineF1;
    fileF1Sum += sc.fileF1;
    if (res.overflowOrError) failCount += 1;
    totalTurns += res.turns;
    totalPrompt += res.promptTokens;
    totalCached += res.cachedTokens;
    if (res.peakPromptTokens > maxPeak) maxPeak = res.peakPromptTokens;
    opsTotal += res.compactions + (res.toolCounts["compress"] ?? 0);
    recovTotal +=
      (res.toolCounts["context_query"] ?? 0) +
      (res.toolCounts["search_context"] ?? 0) +
      (res.toolCounts["decompress"] ?? 0);
    sidecarUsd += res.sidecarCostUsd;
    totalUsd += res.totalUnifiedCostUsd;

    details.push({
      task_index: idx + 1,
      instance_id: item.instance_id,
      repo: item.repo,
      scores: sc,
      metrics: res,
    });
  }

  writeFileSync(join(opts.outDir, `contextbench400k-${opts.arm}.json`), JSON.stringify(details, null, 2));
  const n = Math.max(1, opts.items.length);
  return {
    benchmark: "contextbench",
    arm: opts.arm,
    window_tokens: opts.windowTokens,
    corpus_tokens_est: maxPeak,
    items: opts.items.length,
    primary_metric: "Utilized Line F1",
    primary_score_pct: Number(((lineF1Sum / n) * 100).toFixed(1)),
    secondary_metric: "File F1",
    secondary_score_pct: Number(((fileF1Sum / n) * 100).toFixed(1)),
    overflow_or_fail_count: failCount,
    avg_turns: Number((totalTurns / n).toFixed(1)),
    total_prompt_tokens: totalPrompt,
    cache_hit_pct: totalPrompt > 0 ? Number(((totalCached / totalPrompt) * 100).toFixed(1)) : 0,
    avg_peak_window: maxPeak,
    compress_or_edit_ops: opsTotal,
    recovery_queries: recovTotal,
    sidecar_usd: Number(sidecarUsd.toFixed(4)),
    total_unified_usd: Number(totalUsd.toFixed(4)),
    wall_clock_s: Number(((Date.now() - t0) / 1000).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// 4. Oversized Tool-Result & Host-Truncation Recovery SWE Suite (~230K raw tool bytes)
// ---------------------------------------------------------------------------

function buildOversizedToolsWorkspace(wsDir: string): {
  telemetryDumpFile: string;
  corpusTokensEst: number;
  goldFacts: Array<{ key: string; refs: string[] }>;
} {
  mkdirSync(join(wsDir, "services"), { recursive: true });

  // 1. `./run_ci_suite.sh`: outputs ~240KB (3,900 lines). Shard-1 root-cause AssertionError
  //    is at line 48, followed by 3,850 lines of verbose worker teardown logs (>50KB Pi tail limit!).
  const ciLines: string[] = [
    "=== CI Matrix Runner v4.8 (8 Shards) ===",
    "[shard-0] Bootstrapping test containers...",
  ];
  for (let i = 1; i <= 3900; i++) {
    if (i === 46) {
      ciLines.push(
        'FAIL tests/auth/test_kdf_handshake.py::test_primary_salt_derivation - AssertionError: CHECKSUM_SALT_MISMATCH expected="v4-argon-9917" got="v1-md5-0042"',
      );
      ciLines.push("  at verify_gateway_salt (services/kdf_verifier.py:118)");
    } else {
      ciLines.push(
        `[2026-11-17T04:12:${String(i % 60).padStart(2, "0")}Z] INFO shard-${(i % 7) + 2} :: test_routine_batch_${i} PASSED (duration_ms=${11 + (i % 19)}, alloc_kb=4096)`,
      );
    }
  }
  ciLines.push("=== CI SUMMARY: 3899 passed, 1 failed (exit_code=1) ===");
  const ciLogPath = join(wsDir, ".ci_raw_stream.dat");
  writeFileSync(ciLogPath, ciLines.join("\n") + "\n", "utf8");
  const ciScriptPath = join(wsDir, "run_ci_suite.sh");
  writeFileSync(ciScriptPath, `#!/bin/sh\ncat "${ciLogPath}"\nexit 1\n`, "utf8");
  chmodSync(ciScriptPath, 0o755);

  // 2. `services/ledger_engine.py`: 3,400-line Python module (>2,000-line Pi `read` head limit!).
  //    Target function `reconcile_epoch_fee` sits at lines 2860..2875.
  const pyLines: string[] = [
    '"""Core Settlement Ledger Engine."""',
    "",
  ];
  for (let fnIdx = 1; fnIdx <= 420; fnIdx++) {
    if (pyLines.length >= 2858 && pyLines.length <= 2865) {
      while (pyLines.length < 2859) pyLines.push(`# padding line ${pyLines.length + 1}`);
      pyLines.push("def reconcile_epoch_fee(gross_units: int, tier: str) -> dict:");
      pyLines.push('    """Reconciles cross-shard settlement fee at epoch close."""');
      pyLines.push('    SETTLEMENT_CODE = "LEDGER-ZETA-841"');
      pyLines.push("    MAX_DRIFT_BPS = 37");
      pyLines.push("    net = gross_units - (gross_units * MAX_DRIFT_BPS) // 10_000");
      pyLines.push('    return {"code": SETTLEMENT_CODE, "max_drift_bps": MAX_DRIFT_BPS, "net": net}');
      pyLines.push("");
    } else {
      pyLines.push(`def account_rule_${fnIdx}(balance: int) -> int:`);
      pyLines.push(`    """Standard ledger rule #${fnIdx}."""`);
      pyLines.push(`    step_a = balance + ${fnIdx * 17}`);
      pyLines.push(`    step_b = step_a ^ ${(fnIdx * 31) % 255}`);
      pyLines.push(`    return step_b`);
      pyLines.push(`    # verification invariant filler for rule ${fnIdx}`);
      pyLines.push(`    # audit trail marker ${10000 + fnIdx}`);
      pyLines.push("");
    }
  }
  const pyContent = pyLines.join("\n") + "\n";
  writeFileSync(join(wsDir, "services", "ledger_engine.py"), pyContent, "utf8");

  // 3. `telemetry-dump.jsonl`: ~102K tokens (2,600 JSON-lines trace records, ~340KB) returned
  //    UNTRUNCATED by `fetch_telemetry_dump` (simulating an MCP / observability export tool!).
  const dumpLines: string[] = [];
  for (let r = 1; r <= 2600; r++) {
    if (r === 1142) {
      dumpLines.push(
        JSON.stringify({
          seq: r,
          ts: "2026-11-17T04:15:42Z",
          level: "FATAL",
          event: "AssertionError_OOM_Kill",
          culprit_shard: "shard-eu-west-07",
          oom_peak_mb: 65408,
          detail: "CRITICAL OOM in shard-eu-west-07 at oom_peak_mb=65408 during epoch compaction",
        }),
      );
    } else {
      dumpLines.push(
        JSON.stringify({
          seq: r,
          ts: "2026-11-17T04:15:00Z",
          level: "INFO",
          event: "heartbeat_ok",
          shard: `shard-us-east-${String((r % 16) + 1).padStart(2, "0")}`,
          rss_mb: 2048 + (r % 64),
          detail: "routine cluster health telemetry frame normal",
        }),
      );
    }
  }
  const dumpRaw = dumpLines.join("\n");
  const telemetryDumpFile = join(wsDir, "..", "telemetry-dump.jsonl");
  writeFileSync(telemetryDumpFile, dumpRaw, "utf8");

  const corpusTokensEst =
    estimateTokens(ciLines.join("\n")) + estimateTokens(pyContent) + estimateTokens(dumpRaw);

  return {
    telemetryDumpFile,
    corpusTokensEst,
    goldFacts: [
      { key: "expected_kdf_salt", refs: ["v4-argon-9917"] },
      { key: "settlement_code", refs: ["LEDGER-ZETA-841"] },
      { key: "max_drift_bps", refs: ["37"] },
      { key: "culprit_shard", refs: ["shard-eu-west-07"] },
      { key: "oom_peak_mb", refs: ["65408"] },
    ],
  };
}

async function runOversizedTools400kSuite(opts: {
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<SuiteArmSummary> {
  const runDir = join(opts.outDir, `oversized-${opts.arm}`);
  const wsDir = join(runDir, "workspace");
  mkdirSync(wsDir, { recursive: true });

  const spec = buildOversizedToolsWorkspace(wsDir);

  const prompt = [
    `You are investigating a production release incident in \`${wsDir}\` (inside a ${Math.round(opts.windowTokens / 1000)}K context window).`,
    `Perform ALL THREE diagnostic checks (do NOT modify any files):`,
    `1. Execute \`./run_ci_suite.sh\` via \`bash\` to find the exact \`expected\` KDF salt string from the failing test.`,
    `2. Inspect \`services/ledger_engine.py\` (starting with \`read\`) to locate \`reconcile_epoch_fee\` and extract its exact \`SETTLEMENT_CODE\` and \`MAX_DRIFT_BPS\`.`,
    `3. Call \`fetch_telemetry_dump\` (with \`cluster: "prod-eu-west"\`) to inspect the cluster telemetry export and identify the fatal OOM \`culprit_shard\` and \`oom_peak_mb\`.`,
    ``,
    `End your final response with a JSON block in this exact format:`,
    "```json",
    `{"expected_kdf_salt": "...", "settlement_code": "...", "max_drift_bps": "...", "culprit_shard": "...", "oom_peak_mb": "..."}`,
    "```",
  ].join("\n");

  const res = await runPiCli({
    arm: opts.arm,
    prompt,
    cwd: wsDir,
    runWorkDir: runDir,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    builtinTools: "read,grep,find,ls,bash",
    telemetryDumpFile: spec.telemetryDumpFile,
    timeoutMs: 360_000,
    tag: `OVERSIZED-TOOLS/${opts.arm}`,
  });

  let parsedObj: Record<string, unknown> = {};
  const jm =
    res.finalAnswer.match(/```(?:json)?\s*(\{[\s\S]*?"expected_kdf_salt"[\s\S]*?\})\s*```/i) ??
    res.finalAnswer.match(/(\{[\s\S]*?"expected_kdf_salt"[\s\S]*?\})/);
  if (jm) {
    try {
      parsedObj = JSON.parse(jm[1]) as Record<string, unknown>;
    } catch {
      // fallback to raw answer matching
    }
  }

  let emHits = 0;
  let f1Sum = 0;
  const factDetails = spec.goldFacts.map((gf) => {
    const val = parsedObj[gf.key] !== undefined ? String(parsedObj[gf.key]) : res.finalAnswer;
    const pred = res.overflowOrError && !val ? "" : val;
    const em = bestSubstringMatch(pred, gf.refs);
    const f1 = Math.max(em, qaF1Score(pred, gf.refs));
    emHits += em;
    f1Sum += f1;
    return { key: gf.key, refs: gf.refs, pred, em, f1 };
  });

  writeFileSync(
    join(opts.outDir, `oversized400k-${opts.arm}.json`),
    JSON.stringify({ corpus_tokens_est: spec.corpusTokensEst, factDetails, metrics: res }, null, 2),
  );

  const n = spec.goldFacts.length;
  const ops = res.compactions + (res.toolCounts["compress"] ?? 0);
  const recov =
    (res.toolCounts["context_query"] ?? 0) +
    (res.toolCounts["search_context"] ?? 0) +
    (res.toolCounts["decompress"] ?? 0);

  return {
    benchmark: "oversized-tools",
    arm: opts.arm,
    window_tokens: opts.windowTokens,
    corpus_tokens_est: spec.corpusTokensEst,
    items: n,
    primary_metric: "Exact Fact Recall",
    primary_score_pct: Number(((emHits / n) * 100).toFixed(1)),
    secondary_metric: "Max(EM,F1)",
    secondary_score_pct: Number(((f1Sum / n) * 100).toFixed(1)),
    overflow_or_fail_count: res.overflowOrError ? 1 : 0,
    avg_turns: res.turns,
    total_prompt_tokens: res.promptTokens,
    cache_hit_pct: res.promptTokens > 0 ? Number(((res.cachedTokens / res.promptTokens) * 100).toFixed(1)) : 0,
    avg_peak_window: res.peakPromptTokens,
    compress_or_edit_ops: ops,
    recovery_queries: recov,
    sidecar_usd: Number(res.sidecarCostUsd.toFixed(4)),
    total_unified_usd: Number(res.totalUnifiedCostUsd.toFixed(4)),
    wall_clock_s: res.wallClockS,
  };
}

// ---------------------------------------------------------------------------
// 5. Sub-400K Context Pollution & Explicit Task-Pivot Suite (~85K polluted session)
// ---------------------------------------------------------------------------

function buildPollutionPhase1Workspace(wsDir: string): number {
  mkdirSync(join(wsDir, "docs"), { recursive: true });
  mkdirSync(join(wsDir, "inventory"), { recursive: true });

  // 3 legacy v1 RFC files (~42KB / ~13K tok each => <50KB Pi read limit & <24K coding cap,
  // so ALL 4 arms load ~55K+ tokens of polluted v1 assumptions directly into active context!).
  const makeLegacyDoc = (title: string, domain: string) => {
    const blocks: string[] = [`# ${title} (Legacy Architecture Draft v1.9)`];
    for (let s = 1; s <= 115; s++) {
      blocks.push(
        `## Section ${s}: ${domain} Enforcement Policy #${s}\n` +
          `All production nodes in ${domain} MUST bind \`gateway_port = 7071\`, negotiate \`aead_cipher = "chacha20-legacy-v1"\`, enforce \`quorum_voters = 3\`, and require header \`auth_header = "X-Cluster-Token-V1"\`. Any attempt to deviate from port 7071 or quorum 3 under v1 is prohibited during legacy freeze window ${2024 + (s % 3)}.`,
      );
    }
    return blocks.join("\n\n") + "\n";
  };

  const d1 = makeLegacyDoc("Gateway Transport RFC-101", "Ingress-Gateway");
  const d2 = makeLegacyDoc("AEAD Handshake RFC-102", "Crypto-Envelope");
  const d3 = makeLegacyDoc("Raft Quorum RFC-103", "Consensus-Ring");

  writeFileSync(join(wsDir, "docs", "v1_gateway_rfc.md"), d1, "utf8");
  writeFileSync(join(wsDir, "docs", "v1_crypto_rfc.md"), d2, "utf8");
  writeFileSync(join(wsDir, "docs", "v1_quorum_rfc.md"), d3, "utf8");

  // Authoritative hardware inventory file read during Phase 1 (contains ground-truth HSM serial!)
  const invLines: string[] = ["# Datacenter Hardware Inventory (Authoritative Asset Registry)"];
  for (let r = 1; r <= 620; r++) {
    if (r === 318) {
      invLines.push('PRIMARY_HSM_SERIAL="HSM-NITRO-5519" # active root hardware security module');
    } else {
      invLines.push(`RACK_NODE_${String(r).padStart(4, "0")}="srv-blade-${r}.dc1.internal" # status=online`);
    }
  }
  const invText = invLines.join("\n") + "\n";
  writeFileSync(join(wsDir, "inventory", "datacenter_assets.env"), invText, "utf8");

  return (
    estimateTokens(d1) +
    estimateTokens(d2) +
    estimateTokens(d3) +
    estimateTokens(invText)
  );
}

function addPollutionPhase2Specs(wsDir: string): number {
  mkdirSync(join(wsDir, "specs"), { recursive: true });
  const s1 = [
    "# Sovereign Production Cutover Contract (v3.0 — AUTHORITATIVE)",
    "",
    "Effective immediately, all legacy v1 RFCs (`docs/v1_*.md`) are revoked and superseded.",
    "- `gateway_port`: `19443` (replaces deprecated v1 port 7071)",
    '- `aead_cipher`: `"aes-256-gcm-siv-v3"` (replaces chacha20-legacy-v1)',
    "- See `specs/v3_attestation_policy.md` for `auth_header` and `specs/v3_consensus_tiers.md` for `quorum_voters`.",
  ].join("\n");

  const s2 = [
    "# Sovereign v3 Attestation Policy",
    "",
    '- Mandatory production header: `auth_header = "X-Sovereign-V3-Attest"`',
    '- Legacy header `"X-Cluster-Token-V1"` is strictly rejected at ingress.',
  ].join("\n");

  const s3 = [
    "# Sovereign v3 Consensus Tiers",
    "",
    "- Historical note (revoked): v1 used 3 voters; staging-preview v2 tested 5 voters.",
    "- Active Sovereign v3 Production Contract: `quorum_voters = 9` (`ACTIVE_PROD_TIER = sovereign-v3-nine`).",
  ].join("\n");

  writeFileSync(join(wsDir, "specs", "v3_cutover_contract.md"), s1, "utf8");
  writeFileSync(join(wsDir, "specs", "v3_attestation_policy.md"), s2, "utf8");
  writeFileSync(join(wsDir, "specs", "v3_consensus_tiers.md"), s3, "utf8");

  return estimateTokens(s1) + estimateTokens(s2) + estimateTokens(s3);
}

async function runPollutionBoundary400kSuite(opts: {
  arm: PiArm;
  windowTokens: number;
  model: string;
  apiKey: string;
  baseUrl: string;
  outDir: string;
}): Promise<SuiteArmSummary> {
  const runDir = join(opts.outDir, `pollution-${opts.arm}`);
  const wsDir = join(runDir, "workspace");
  mkdirSync(wsDir, { recursive: true });
  const sharedSessionFile = join(runDir, "pollution-session.jsonl");

  const p1Corpus = buildPollutionPhase1Workspace(wsDir);

  const phase1Prompt = [
    `Read ALL FOUR files in \`${wsDir}\` using \`read\`:`,
    `1. \`docs/v1_gateway_rfc.md\``,
    `2. \`docs/v1_crypto_rfc.md\``,
    `3. \`docs/v1_quorum_rfc.md\``,
    `4. \`inventory/datacenter_assets.env\``,
    `Then write a detailed deployment readiness summary committing to the v1 gateway port, cipher, quorum, auth header, and hardware HSM serial.`,
  ].join("\n");

  const res1 = await runPiCli({
    arm: opts.arm,
    prompt: phase1Prompt,
    cwd: wsDir,
    runWorkDir: runDir,
    sessionFile: sharedSessionFile,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    builtinTools: "read,grep,find,ls,bash",
    timeoutMs: 300_000,
    tag: `POLLUTION-P1/${opts.arm}`,
  });

  const p2Corpus = addPollutionPhase2Specs(wsDir);

  const phase2Prompt = [
    `Ignore the previous hypothesis and discard all earlier legacy v1 conclusions — our initial v1 RFC direction was deprecated and wrong!`,
    `We have just placed the authoritative v3 cutover specifications in \`specs/\`.`,
    `Read the files under \`specs/\` step by step to determine the active Sovereign v3 production settings, and also include the exact \`PRIMARY_HSM_SERIAL\` from the hardware inventory (use \`context_query\` if indexed, or verify from disk).`,
    `Do NOT inherit any revoked v1 values (7071, chacha20-legacy-v1, 3, X-Cluster-Token-V1).`,
    ``,
    `End your final response with a JSON block in this exact format:`,
    "```json",
    `{"gateway_port": "...", "aead_cipher": "...", "quorum_voters": "...", "auth_header": "...", "primary_hsm_serial": "..."}`,
    "```",
  ].join("\n");

  const res2 = await runPiCli({
    arm: opts.arm,
    prompt: phase2Prompt,
    cwd: wsDir,
    runWorkDir: runDir,
    sessionFile: sharedSessionFile,
    windowTokens: opts.windowTokens,
    model: opts.model,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl,
    builtinTools: "read,grep,find,ls,bash",
    timeoutMs: 300_000,
    tag: `POLLUTION-P2/${opts.arm}`,
  });

  const goldFacts = [
    { key: "gateway_port", refs: ["19443"], poison: "7071" },
    { key: "aead_cipher", refs: ["aes-256-gcm-siv-v3"], poison: "chacha20-legacy-v1" },
    { key: "quorum_voters", refs: ["9"], poison: "3" },
    { key: "auth_header", refs: ["X-Sovereign-V3-Attest"], poison: "X-Cluster-Token-V1" },
    { key: "primary_hsm_serial", refs: ["HSM-NITRO-5519"], poison: "" },
  ];

  let parsedObj: Record<string, unknown> = {};
  const jm =
    res2.finalAnswer.match(/```(?:json)?\s*(\{[\s\S]*?"gateway_port"[\s\S]*?\})\s*```/i) ??
    res2.finalAnswer.match(/(\{[\s\S]*?"gateway_port"[\s\S]*?\})/);
  if (jm) {
    try {
      parsedObj = JSON.parse(jm[1]) as Record<string, unknown>;
    } catch {
      // fallback
    }
  }

  let cleanHits = 0;
  let f1Sum = 0;
  const details = goldFacts.map((gf) => {
    const val = parsedObj[gf.key] !== undefined ? String(parsedObj[gf.key]) : res2.finalAnswer;
    const pred = res2.overflowOrError && !val ? "" : val;
    const hasGold = bestSubstringMatch(pred, gf.refs) === 1;
    const leakedPoison = gf.poison ? bestSubstringMatch(pred, [gf.poison]) === 1 && !hasGold : false;
    const clean = hasGold && !leakedPoison ? 1 : 0;
    const f1 = Math.max(clean, qaF1Score(pred, gf.refs));
    cleanHits += clean;
    f1Sum += f1;
    return { key: gf.key, refs: gf.refs, pred, clean, leakedPoison, f1 };
  });

  writeFileSync(
    join(opts.outDir, `pollution400k-${opts.arm}.json`),
    JSON.stringify(
      {
        corpus_tokens_est: p1Corpus + p2Corpus,
        phase1_metrics: res1,
        phase2_metrics: res2,
        details,
      },
      null,
      2,
    ),
  );

  // Report Phase-2 post-pivot metrics AND combined session totals clearly
  const totalPrompt = res1.promptTokens + res2.promptTokens;
  const totalCached = res1.cachedTokens + res2.cachedTokens;
  const totalTurns = res1.turns + res2.turns;
  const peakWin = Math.max(res1.peakPromptTokens, res2.peakPromptTokens);
  const ops =
    res1.compactions +
    res2.compactions +
    (res1.toolCounts["compress"] ?? 0) +
    (res2.toolCounts["compress"] ?? 0);
  const recov =
    (res1.toolCounts["context_query"] ?? 0) +
    (res2.toolCounts["context_query"] ?? 0) +
    (res1.toolCounts["search_context"] ?? 0) +
    (res2.toolCounts["search_context"] ?? 0);
  const sidecarUsd = res1.sidecarCostUsd + res2.sidecarCostUsd;
  const totalUsd = res1.totalUnifiedCostUsd + res2.totalUnifiedCostUsd;

  return {
    benchmark: "pollution-boundary",
    arm: opts.arm,
    window_tokens: opts.windowTokens,
    corpus_tokens_est: p1Corpus + p2Corpus,
    items: goldFacts.length,
    primary_metric: "Unpolluted Acc",
    primary_score_pct: Number(((cleanHits / goldFacts.length) * 100).toFixed(1)),
    secondary_metric: "P2 Prompt Saved%",
    // Measure Phase-2 active window reduction vs Phase-1 peak!
    secondary_score_pct:
      res1.peakPromptTokens > 0
        ? Number((((res1.peakPromptTokens - res2.peakPromptTokens) / res1.peakPromptTokens) * 100).toFixed(1))
        : 0,
    overflow_or_fail_count: (res1.overflowOrError ? 1 : 0) + (res2.overflowOrError ? 1 : 0),
    avg_turns: totalTurns,
    total_prompt_tokens: totalPrompt,
    cache_hit_pct: totalPrompt > 0 ? Number(((totalCached / totalPrompt) * 100).toFixed(1)) : 0,
    avg_peak_window: res2.peakPromptTokens, // Phase-2 post-pivot peak window!
    compress_or_edit_ops: ops,
    recovery_queries: recov,
    sidecar_usd: Number(sidecarUsd.toFixed(4)),
    total_unified_usd: Number(totalUsd.toFixed(4)),
    wall_clock_s: Number((res1.wallClockS + res2.wallClockS).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// CLI Entrypoint
// ---------------------------------------------------------------------------

function parseArg(flag: string, fallback: string): string {
  const idx = process.argv.indexOf(flag);
  if (idx !== -1 && idx + 1 < process.argv.length) return process.argv[idx + 1];
  return fallback;
}

async function main() {
  const benchArg = parseArg("--bench", "all");
  const armsArg = parseArg("--arms", "full,pi-native,billion-context,cpat");
  const windowTokens = Math.max(32_000, Number(parseArg("--window", String(DEFAULT_WINDOW_TOKENS))));
  const cbItemsCount = Math.max(1, Number(parseArg("--cb-items", "3")));
  const model = parseArg("--model", "deepseek-flash");

  const suites: BenchSuite[] =
    benchArg === "all"
      ? [
          "oversized-tools",
          "pollution-boundary",
          "memoryagentbench",
          "longmemeval",
          "contextbench",
        ]
      : (benchArg.split(",").map((s) => s.trim()) as BenchSuite[]);
  const arms = armsArg.split(",").map((s) => s.trim()) as PiArm[];

  const env = loadDeepSeekEnv();
  const judgeClient = new DeepSeekClient(env);

  const stamp = `${new Date().toISOString().replace(/[:.]/g, "-")}-p${process.pid}`;
  const outDir = resolve("runs", `pi-400k-bench-${stamp}`);
  mkdirSync(outDir, { recursive: true });

  console.log(`\n=== Pi CLI × Billion-Context 400K-Window Benchmark Suite ===`);
  console.log(
    `Window: ${windowTokens.toLocaleString()} tok | Suites: ${suites.join(", ")} | Arms: ${arms.join(", ")} | Model: ${model}`,
  );
  console.log(`Output: ${outDir}\n`);

  const summaries: SuiteArmSummary[] = [];

  for (const suite of suites) {
    if (suite === "oversized-tools") {
      for (const arm of arms) {
        console.log(
          `\n--- Running Oversized Tool-Result & Truncation Recovery SWE Suite (${Math.round(windowTokens / 1000)}K window) [${arm}] ---`,
        );
        const s = await runOversizedTools400kSuite({
          arm,
          windowTokens,
          model,
          apiKey: env.apiKey,
          baseUrl: env.baseUrl,
          outDir,
        });
        summaries.push(s);
        console.log(
          `  => [OVERSIZED/${arm}] ${s.primary_metric}=${s.primary_score_pct}% | ${s.secondary_metric}=${s.secondary_score_pct}% | fail=${s.overflow_or_fail_count} | prompt=${s.total_prompt_tokens.toLocaleString()} (hit ${s.cache_hit_pct}%) | peak=${s.avg_peak_window.toLocaleString()} | cost=$${s.total_unified_usd.toFixed(4)} | ${s.wall_clock_s}s`,
        );
      }
    } else if (suite === "pollution-boundary") {
      for (const arm of arms) {
        console.log(
          `\n--- Running Sub-400K Context Pollution & Explicit Task-Pivot Suite (${Math.round(windowTokens / 1000)}K window) [${arm}] ---`,
        );
        const s = await runPollutionBoundary400kSuite({
          arm,
          windowTokens,
          model,
          apiKey: env.apiKey,
          baseUrl: env.baseUrl,
          outDir,
        });
        summaries.push(s);
        console.log(
          `  => [POLLUTION/${arm}] ${s.primary_metric}=${s.primary_score_pct}% | ${s.secondary_metric}=${s.secondary_score_pct}% | fail=${s.overflow_or_fail_count} | prompt=${s.total_prompt_tokens.toLocaleString()} (hit ${s.cache_hit_pct}%) | P2_peak=${s.avg_peak_window.toLocaleString()} | cost=$${s.total_unified_usd.toFixed(4)} | ${s.wall_clock_s}s`,
        );
      }
    } else if (suite === "memoryagentbench") {
      const ep = await loadMab400kEpisode();
      for (const arm of arms) {
        console.log(
          `\n--- Running MemoryAgentBench AR (~${Math.round(ep.totalTokensEst / 1000)}K stream vs ${Math.round(windowTokens / 1000)}K window) [${arm}] ---`,
        );
        const s = await runMab400kSuite({
          episode: ep,
          arm,
          windowTokens,
          model,
          apiKey: env.apiKey,
          baseUrl: env.baseUrl,
          outDir,
        });
        summaries.push(s);
        console.log(
          `  => [MAB-480K/${arm}] ${s.primary_metric}=${s.primary_score_pct}% | ${s.secondary_metric}=${s.secondary_score_pct}% | fail=${s.overflow_or_fail_count} | prompt=${s.total_prompt_tokens.toLocaleString()} (hit ${s.cache_hit_pct}%) | peak=${s.avg_peak_window.toLocaleString()} | cost=$${s.total_unified_usd.toFixed(4)} | ${s.wall_clock_s}s`,
        );
      }
    } else if (suite === "longmemeval") {
      const lmeAll = JSON.parse(
        readFileSync(join(DATA_CACHE_DIR, "longmemeval_stratified.json"), "utf8"),
      ) as LongMemEvalItem[];
      for (const arm of arms) {
        console.log(`\n--- Running LongMemEval-S (~510K marathon vs ${Math.round(windowTokens / 1000)}K window) [${arm}] ---`);
        const s = await runLongMem400kSuite({
          items: lmeAll,
          arm,
          windowTokens,
          model,
          apiKey: env.apiKey,
          baseUrl: env.baseUrl,
          judgeClient,
          outDir,
        });
        summaries.push(s);
        console.log(
          `  => [LME-510K/${arm}] ${s.primary_metric}=${s.primary_score_pct}% | ${s.secondary_metric}=${s.secondary_score_pct}% | fail=${s.overflow_or_fail_count} | prompt=${s.total_prompt_tokens.toLocaleString()} (hit ${s.cache_hit_pct}%) | peak=${s.avg_peak_window.toLocaleString()} | cost=$${s.total_unified_usd.toFixed(4)} | ${s.wall_clock_s}s`,
        );
      }
    } else if (suite === "contextbench") {
      const cbAll = JSON.parse(
        readFileSync(join(DATA_CACHE_DIR, "contextbench_verified_sample.json"), "utf8"),
      ) as ContextBenchItem[];
      // Pick astropy, django, sympy (indices 0, 1, 3 — fast shallow checkouts)
      const picked = [cbAll[0], cbAll[1], cbAll[3]].filter(Boolean).slice(0, cbItemsCount);
      for (const arm of arms) {
        console.log(
          `\n--- Running ContextBench Continuous SWE Marathon (${picked.length} repos in 1 ${Math.round(windowTokens / 1000)}K session) [${arm}] ---`,
        );
        const s = await runContextBench400kMarathon({
          items: picked,
          arm,
          windowTokens,
          model,
          apiKey: env.apiKey,
          baseUrl: env.baseUrl,
          outDir,
        });
        summaries.push(s);
        console.log(
          `  => [CB-400K/${arm}] ${s.primary_metric}=${s.primary_score_pct}% | ${s.secondary_metric}=${s.secondary_score_pct}% | fail=${s.overflow_or_fail_count} | prompt=${s.total_prompt_tokens.toLocaleString()} (hit ${s.cache_hit_pct}%) | peak=${s.avg_peak_window.toLocaleString()} | cost=$${s.total_unified_usd.toFixed(4)} | ${s.wall_clock_s}s`,
        );
      }
    }
  }

  writeFileSync(join(outDir, "summary.json"), JSON.stringify(summaries, null, 2), "utf8");

  const mdLines = [
    `# Pi CLI × Billion-Context 400K-Window Benchmark Summary`,
    ``,
    `| Benchmark | Arm | Window | Corpus | Score 1 | Score 2 | Fail | Turns | Prompt Tok | Cache Hit | Peak Win | Ops | Queries | Sidecar $ | Total $ (BC口径) | Time |`,
    `|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|`,
    ...summaries.map(
      (s) =>
        `| ${s.benchmark} | \`${s.arm}\` | ${Math.round(s.window_tokens / 1000)}K | ~${Math.round(s.corpus_tokens_est / 1000)}K | **${s.primary_score_pct}%** (${s.primary_metric}) | ${s.secondary_score_pct}% (${s.secondary_metric}) | ${s.overflow_or_fail_count} | ${s.avg_turns} | ${s.total_prompt_tokens.toLocaleString()} | ${s.cache_hit_pct}% | ${s.avg_peak_window.toLocaleString()} | ${s.compress_or_edit_ops} | ${s.recovery_queries} | $${s.sidecar_usd.toFixed(4)} | **$${s.total_unified_usd.toFixed(4)}** | ${s.wall_clock_s}s |`,
    ),
  ];
  const md = mdLines.join("\n") + "\n";
  writeFileSync(join(outDir, "summary.md"), md, "utf8");
  console.log(`\n${md}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
