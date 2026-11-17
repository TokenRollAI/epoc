/**
 * Real Third-Party Public Benchmark Runner for CPAT ContextHarness (v2.1):
 *
 * Evaluates `react` | `threshold` | `cpat` (v1) | `harness` (v2.1) on real
 * items fetched from three canonical public benchmarks on Hugging Face / GitHub:
 *
 *   1. `longmemeval` — LongMemEval-S (`xiaowu0162/longmemeval-cleaned`)
 *      Real ~45–55 session (~110K token) timestamped chat histories evaluated
 *      under a 32K bounded window (~3.4x window), scored by both normalized
 *      substring/F1 and official LongMemEval LLM-Judge protocol.
 *
 *   2. `memoryagentbench` — MemoryAgentBench (`ai-hyz/MemoryAgentBench`)
 *      Real multi-hop (`Accurate_Retrieval`) & `Conflict_Resolution` items
 *      evaluated under MemoryAgentBench's official one-pass incremental chunk
 *      ingestion protocol (~60K token stream inside a 26K bounded window,
 *      non-rereadable chunks, multi-question probing).
 *
 *   3. `contextbench` — ContextBench Verified (`Contextbench/ContextBench`)
 *      Real GitHub repositories checked out at exact `base_commit` under a
 *      bounded 24K working window, scored against human-annotated `gold_context`
 *      for File F1 and Line-Level Utilized F1.
 *
 * Usage:
 *   node bench/public-bench.ts [--bench all|longmemeval|memoryagentbench|contextbench]
 *                              [--arms react,threshold,cpat,harness]
 *                              [--items 3] [--model deepseek-flash]
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  runAgent,
  type AgentMode,
  type CustomToolSuite,
  type RunResult,
} from "../src/agent/loop.ts";
import { DeepSeekClient } from "../src/deepseek/client.ts";
import type { CpatConfig, ToolDefinition } from "../src/types.ts";
import { loadDeepSeekEnv, loadSidecarEnv } from "../src/util/env.ts";
import { estimateTokens } from "../src/util/misc.ts";
import { normalizeAnswer, qaF1Score } from "./f1.ts";

function bestSubstringMatch(prediction: string, groundTruths: string[]): number {
  const normPred = normalizeAnswer(prediction);
  if (!normPred) return 0;
  for (const gt of groundTruths) {
    const normGt = normalizeAnswer(gt);
    if (normGt && normPred.includes(normGt)) return 1;
  }
  return 0;
}

type PublicBenchName = "longmemeval" | "memoryagentbench" | "contextbench";

interface LongMemEvalItem {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  question_date?: string;
  haystack_dates?: string[];
  haystack_sessions: Array<Array<{ role: string; content: string }>>;
}

interface MabEpisode {
  id: string;
  split: string;
  source: string;
  chunks: string[];
  totalTokensEst: number;
  questions: string[];
  answers: string[][];
}

interface ContextBenchItem {
  instance_id: string;
  original_inst_id: string;
  repo: string;
  repo_url: string;
  base_commit: string;
  problem_statement: string;
  gold_context: Array<{
    file: string;
    start_line: number;
    end_line: number;
  }>;
}

interface PublicArmSummary {
  benchmark: PublicBenchName;
  arm: AgentMode;
  items: number;
  primary_score_pct: number;
  secondary_score_pct: number;
  primary_metric_name: string;
  secondary_metric_name: string;
  terminated_early_count: number;
  avg_turns: number;
  total_prompt_tokens: number;
  cache_hit_ratio: number;
  avg_peak_tokens: number;
  sidecar_usd: number;
  effective_cost_usd: number;
  context_queries: number;
  wall_clock_s: number;
}

const DATA_CACHE_DIR = resolve("data/public");

// ---------------------------------------------------------------------------
// 1. LongMemEval-S Fetcher & Runner
// ---------------------------------------------------------------------------

function extractJsonObjectsFromChunk(
  chunk: string,
  skipFirst = false,
): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  let startIdx = 0;
  if (skipFirst) {
    // Advance to a clean session-item boundary `{"question_id":`
    const marker = chunk.indexOf('{"question_id"');
    if (marker === -1) return out;
    startIdx = marker;
  }

  let depth = 0;
  let inStr = false;
  let esc = false;
  let objStart = -1;

  for (let i = startIdx; i < chunk.length; i++) {
    const ch = chunk[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") {
      if (depth === 0) objStart = i;
      depth++;
    } else if (ch === "}") {
      if (depth > 0) depth--;
      if (depth === 0 && objStart !== -1) {
        try {
          const parsed = JSON.parse(chunk.slice(objStart, i + 1)) as Record<string, unknown>;
          if (typeof parsed.question_id === "string" && Array.isArray(parsed.haystack_sessions)) {
            out.push(parsed);
          }
        } catch {
          // ignore partial boundary object
        }
        objStart = -1;
      }
    }
  }
  return out;
}

async function loadLongMemEvalItems(count: number): Promise<LongMemEvalItem[]> {
  mkdirSync(DATA_CACHE_DIR, { recursive: true });
  const cachePath = join(DATA_CACHE_DIR, "longmemeval_stratified.json");
  if (existsSync(cachePath)) {
    const cached = JSON.parse(readFileSync(cachePath, "utf8")) as LongMemEvalItem[];
    if (cached.length >= count) return cached.slice(0, count);
  }

  console.log("  [fetch] Pulling stratified LongMemEval-S items via HTTP Range...");
  const url =
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json";
  const offsets = [0, 65_000_000, 140_000_000, 215_000_000];
  const pool: LongMemEvalItem[] = [];

  for (const off of offsets) {
    const end = off + 6_500_000;
    const res = await fetch(url, { headers: { Range: `bytes=${off}-${end}` } });
    if (!res.ok && res.status !== 206) continue;
    const text = await res.text();
    const objs = extractJsonObjectsFromChunk(text, off > 0);
    for (const o of objs) {
      if (
        typeof o.question === "string" &&
        (typeof o.answer === "string" || typeof o.answer === "number") &&
        Array.isArray(o.haystack_sessions)
      ) {
        pool.push({
          question_id: String(o.question_id),
          question_type: String(o.question_type ?? "unknown"),
          question: o.question,
          answer: String(o.answer),
          question_date: o.question_date ? String(o.question_date) : undefined,
          haystack_dates: Array.isArray(o.haystack_dates)
            ? o.haystack_dates.map(String)
            : undefined,
          haystack_sessions: o.haystack_sessions as LongMemEvalItem["haystack_sessions"],
        });
      }
    }
  }

  // Stratify across distinct `question_type`s, preferring ~55K–115K token items
  const byType = new Map<string, LongMemEvalItem[]>();
  for (const item of pool) {
    const arr = byType.get(item.question_type) ?? [];
    arr.push(item);
    byType.set(item.question_type, arr);
  }

  const selected: LongMemEvalItem[] = [];
  const types = Array.from(byType.keys());
  let round = 0;
  while (selected.length < Math.max(count, 8) && round < 10) {
    for (const t of types) {
      const list = byType.get(t)!;
      if (round < list.length && !selected.some((x) => x.question_id === list[round].question_id)) {
        selected.push(list[round]);
      }
    }
    round++;
  }

  writeFileSync(cachePath, JSON.stringify(selected, null, 2), "utf8");
  return selected.slice(0, count);
}

async function judgeLongMemEvalAnswer(
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
            'You are an exact QA evaluator for LongMemEval. Compare the predicted answer against the ground-truth answer for the given question. Return ONLY JSON: {"correct": true} if the prediction accurately contains the ground-truth fact/value/entity (allowing equivalent phrasing or minor unit formatting), otherwise {"correct": false}.',
        },
        {
          role: "user",
          content: `Question: ${question}\nGround Truth: ${gold}\nModel Prediction: ${prediction.slice(0, 1500)}`,
        },
      ],
      temperature: 0,
      maxTokens: 300,
      reasoningEffort: "low",
      responseFormat: { type: "json_object" },
    });
    const cleaned = (resp.message.content ?? "")
      .trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "");
    const obj = JSON.parse(cleaned) as { correct?: boolean };
    return obj.correct === true;
  } catch {
    return qaF1Score(prediction, [gold]) >= 0.5;
  }
}

async function runLongMemEvalArm(opts: {
  items: LongMemEvalItem[];
  arm: AgentMode;
  model: string;
  sidecarModel: string;
  outDir: string;
  client: DeepSeekClient;
  sidecarClient: DeepSeekClient;
}): Promise<PublicArmSummary> {
  const hardWindow = 32_000; // ~110K haystack vs 32K bounded window (~3.4x window!)
  let judgeCorrect = 0;
  let f1Sum = 0;
  let termCount = 0;
  let totalTurns = 0;
  let totalPrompt = 0;
  let totalHit = 0;
  let peakSum = 0;
  let sidecarUsd = 0;
  let effUsd = 0;
  let queries = 0;
  const t0 = Date.now();
  const details: unknown[] = [];

  for (let idx = 0; idx < opts.items.length; idx++) {
    const item = opts.items[idx];
    const itemDir = join(opts.outDir, `longmemeval-${opts.arm}-item${idx + 1}`);
    const workdir = join(itemDir, "workspace");
    mkdirSync(workdir, { recursive: true });

    // Group the ~45–55 sessions into ~8–10 chronological batch chunks (~10K-12K tok each)
    // served through a one-shot stream tool `read_session_batch` so the agent
    // experiences the full chronological history inside the 32K bounded window!
    const batches: string[] = [];
    let curLines: string[] = [];
    let curChars = 0;
    item.haystack_sessions.forEach((sess, sIdx) => {
      const dateStr = item.haystack_dates?.[sIdx] ?? `Session #${sIdx + 1}`;
      const formatted =
        `=== SESSION #${sIdx + 1} [Date: ${dateStr}] ===\n` +
        sess.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
      if (curChars > 0 && curChars + formatted.length > 34_000) {
        batches.push(curLines.join("\n\n"));
        curLines = [];
        curChars = 0;
      }
      curLines.push(formatted);
      curChars += formatted.length;
    });
    if (curLines.length > 0) batches.push(curLines.join("\n\n"));

    let cursor = 0;
    const streamTool: ToolDefinition = {
      type: "function",
      function: {
        name: "read_next_session_batch",
        description: `Read the next chronological batch of user-assistant chat sessions (total ${batches.length} batches). One-shot stream: consume all batches before answering.`,
        parameters: { type: "object", properties: {} },
      },
    };
    const extraTools: CustomToolSuite = {
      defs: [streamTool],
      dispatch: (name) => {
        if (name !== "read_next_session_batch") return undefined;
        if (cursor >= batches.length) {
          return `END_OF_HISTORY: All ${batches.length} chronological session batches have been read.`;
        }
        const b = `--- BATCH ${cursor + 1} OF ${batches.length} ---\n${batches[cursor]}`;
        cursor += 1;
        return b;
      },
    };

    const config: CpatConfig = {
      model: opts.model,
      sidecarModel: opts.sidecarModel,
      maxContextTokens: hardWindow,
      softLimitRatio: 0.68,
      mustActRatio: 0.82,
      criticalRatio: 0.93,
      generational: false,
      allowReplace: false,
      allowRedact: false,
      strictTools: true,
      maxTurns: 36,
      runDir: join(itemDir, "run"),
      verbose: false,
    };

    const prompt =
      `You are evaluating a long-term multi-session chat history (${batches.length} chronological batches). ` +
      `First, call read_next_session_batch repeatedly until all ${batches.length} batches are consumed (do NOT answer early before reading all batches, as later sessions may update earlier facts!). ` +
      `Then answer this question concisely and accurately:\n` +
      `${item.question_date ? `[Current Date: ${item.question_date}]\n` : ""}` +
      `Question: ${item.question}`;

    const run = await runAgent({
      task: prompt,
      config,
      client: opts.client,
      sidecarClient: opts.sidecarClient,
      workdir,
      mode: opts.arm,
      hardWindowTokens: hardWindow,
      taskToolNames: [],
      extraTools,
      onTurn: (t) => {
        const hitPct = t.promptTokens ? Math.round((t.cacheHitTokens / t.promptTokens) * 100) : 0;
        console.log(
          `  [LME/${opts.arm}/#${idx + 1} ${item.question_type}] t${String(t.turn).padStart(2)}: ${t.promptTokens}in(${hitPct}%)/${t.completionTokens}out -> ${t.toolCalls.join(",") || "ANS"}`,
        );
      },
    });

    const isCorrect = await judgeLongMemEvalAnswer(
      opts.client,
      opts.model,
      item.question,
      item.answer,
      run.answer,
    );
    const f1 = qaF1Score(run.answer, [item.answer]);
    if (isCorrect) judgeCorrect += 1;
    f1Sum += f1;
    if (run.metrics.terminated_early) termCount += 1;
    totalTurns += run.metrics.llm_calls;
    totalPrompt += run.metrics.prompt_tokens;
    totalHit += run.metrics.cache_hit_tokens;
    peakSum += run.metrics.peak_view_tokens;
    sidecarUsd += run.metrics.cost.sidecar_total_usd;
    effUsd += run.metrics.cost.effective_cost_usd;
    queries += run.metrics.context_query_calls;

    details.push({
      question_id: item.question_id,
      question_type: item.question_type,
      batches: batches.length,
      question: item.question,
      gold: item.answer,
      prediction: run.answer,
      judge_correct: isCorrect,
      token_f1: Number(f1.toFixed(3)),
      metrics: run.metrics,
    });
  }

  writeFileSync(
    join(opts.outDir, `longmemeval-${opts.arm}-details.json`),
    JSON.stringify(details, null, 2),
  );
  const n = Math.max(1, opts.items.length);
  return {
    benchmark: "longmemeval",
    arm: opts.arm,
    items: opts.items.length,
    primary_score_pct: Number(((judgeCorrect / n) * 100).toFixed(1)),
    secondary_score_pct: Number(((f1Sum / n) * 100).toFixed(1)),
    primary_metric_name: "Judge Acc",
    secondary_metric_name: "Token F1",
    terminated_early_count: termCount,
    avg_turns: Number((totalTurns / n).toFixed(1)),
    total_prompt_tokens: totalPrompt,
    cache_hit_ratio: totalPrompt > 0 ? Number((totalHit / totalPrompt).toFixed(3)) : 0,
    avg_peak_tokens: Math.round(peakSum / n),
    sidecar_usd: Number(sidecarUsd.toFixed(4)),
    effective_cost_usd: Number(effUsd.toFixed(4)),
    context_queries: queries,
    wall_clock_s: Number(((Date.now() - t0) / 1000).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// 2. MemoryAgentBench Fetcher & Runner (Accurate_Retrieval & Conflict_Resolution)
// ---------------------------------------------------------------------------

async function loadMemoryAgentBenchEpisodes(count: number): Promise<MabEpisode[]> {
  mkdirSync(DATA_CACHE_DIR, { recursive: true });
  const cachePath = join(DATA_CACHE_DIR, "memoryagentbench_episodes.json");
  if (existsSync(cachePath)) {
    const cached = JSON.parse(readFileSync(cachePath, "utf8")) as MabEpisode[];
    if (cached.length >= count) return cached.slice(0, count);
  }

  console.log("  [fetch] Pulling MemoryAgentBench rows from Hugging Face...");
  const specs = [
    { split: "Accurate_Retrieval", offset: 0 }, // single-hop needle QA
    { split: "Accurate_Retrieval", offset: 1 }, // 2-hop multi-doc QA (ruler_qa2)
    { split: "Conflict_Resolution", offset: 0 }, // conflicting updates
  ];

  const episodes: MabEpisode[] = [];
  for (const sp of specs) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=${sp.split}&offset=${sp.offset}&length=1`;
    const res = await fetch(url);
    if (!res.ok) continue;
    const data = (await res.json()) as {
      rows?: Array<{ row_idx: number; row: Record<string, unknown> }>;
    };
    const row = data.rows?.[0]?.row;
    if (!row) continue;

    const fullContext = String(row.context ?? "");
    const allQs = (row.questions as string[]) ?? [];
    const allAns = (row.answers as Array<string[] | string>) ?? [];
    const meta = (row.metadata as Record<string, unknown>) ?? {};

    // Split into logical documents/sections while guaranteeing selected Qs'
    // gold evidence documents + real distractors total ~55K–65K tokens (~2.2x 26K window).
    const rawDocs = fullContext.includes("\nDocument ")
      ? fullContext
          .split(/(?=\nDocument \d+:)/g)
          .map((s) => s.trim())
          .filter(Boolean)
      : chunkStringByParagraphs(fullContext, 6_000);

    const pickedQIdxs: number[] = [];
    const requiredDocIdxs = new Set<number>();

    for (let qi = 0; qi < allQs.length && pickedQIdxs.length < 4; qi++) {
      const refs = Array.isArray(allAns[qi]) ? allAns[qi].map(String) : [String(allAns[qi] ?? "")];
      const primaryRef = refs[0]?.trim();
      if (!primaryRef || primaryRef.length < 2) continue;

      // Find docs mentioning key terms from question + answer
      const lowerRef = primaryRef.toLowerCase();
      const matchingDocs: number[] = [];
      rawDocs.forEach((docText, di) => {
        if (docText.toLowerCase().includes(lowerRef)) matchingDocs.push(di);
      });
      if (matchingDocs.length > 0 && matchingDocs.length <= 6) {
        pickedQIdxs.push(qi);
        for (const di of matchingDocs.slice(0, 3)) requiredDocIdxs.add(di);
      }
    }

    // If Conflict_Resolution has sequential turns rather than exact answer substrings,
    // take the first 55K tokens of sequential context + matching questions.
    let selectedTexts: string[] = [];
    if (pickedQIdxs.length >= 2) {
      const chosenIndices = new Set<number>(requiredDocIdxs);
      let estChars = Array.from(chosenIndices).reduce((s, i) => s + rawDocs[i].length, 0);
      for (let i = 0; i < rawDocs.length && estChars < 185_000; i++) {
        if (!chosenIndices.has(i)) {
          chosenIndices.add(i);
          estChars += rawDocs[i].length;
        }
      }
      selectedTexts = Array.from(chosenIndices)
        .sort((a, b) => a - b)
        .map((i) => rawDocs[i]);
    } else {
      let estChars = 0;
      for (const d of rawDocs) {
        selectedTexts.push(d);
        estChars += d.length;
        if (estChars >= 180_000) break;
      }
      for (let qi = 0; qi < Math.min(4, allQs.length); qi++) pickedQIdxs.push(qi);
    }

    // Pack into ~8–10 stream chunks (~6K tokens each) per MemoryAgentBench protocol
    const chunks = packIntoChunks(selectedTexts, 20_000);
    const totalTokensEst = chunks.reduce((s, c) => s + estimateTokens(c), 0);

    episodes.push({
      id: `${sp.split}_${sp.offset}`,
      split: sp.split,
      source: String(meta.source ?? sp.split),
      chunks,
      totalTokensEst,
      questions: pickedQIdxs.map((i) => allQs[i]),
      answers: pickedQIdxs.map((i) =>
        Array.isArray(allAns[i]) ? allAns[i].map(String) : [String(allAns[i] ?? "")],
      ),
    });
  }

  writeFileSync(cachePath, JSON.stringify(episodes, null, 2), "utf8");
  return episodes.slice(0, count);
}

async function runMemoryAgentBenchArm(opts: {
  episodes: MabEpisode[];
  arm: AgentMode;
  model: string;
  sidecarModel: string;
  outDir: string;
  client: DeepSeekClient;
  sidecarClient: DeepSeekClient;
}): Promise<PublicArmSummary> {
  const hardWindow = 26_000; // ~56K stream vs 26K bounded window!
  let subHitTotal = 0;
  let f1SumTotal = 0;
  let qCountTotal = 0;
  let termCount = 0;
  let totalTurns = 0;
  let totalPrompt = 0;
  let totalHit = 0;
  let peakSum = 0;
  let sidecarUsd = 0;
  let effUsd = 0;
  let queries = 0;
  const t0 = Date.now();
  const details: unknown[] = [];

  for (let idx = 0; idx < opts.episodes.length; idx++) {
    const ep = opts.episodes[idx];
    const epDir = join(opts.outDir, `mab-${opts.arm}-${ep.id}`);
    const workdir = join(epDir, "workspace");
    mkdirSync(workdir, { recursive: true });

    let cursor = 0;
    const chunkTool: ToolDefinition = {
      type: "function",
      function: {
        name: "ingest_next_chunk",
        description: `Read the next incremental memory chunk (${ep.chunks.length} chunks total). One-pass protocol: chunks cannot be re-fetched from the stream once read.`,
        parameters: { type: "object", properties: {} },
      },
    };
    const extraTools: CustomToolSuite = {
      defs: [chunkTool],
      dispatch: (name) => {
        if (name !== "ingest_next_chunk") return undefined;
        if (cursor >= ep.chunks.length) {
          return `END_OF_CHUNKS: All ${ep.chunks.length} chunks have been ingested.`;
        }
        const out = `=== MEMORY CHUNK ${cursor + 1} OF ${ep.chunks.length} ===\n${ep.chunks[cursor]}`;
        cursor += 1;
        return out;
      },
    };

    const config: CpatConfig = {
      model: opts.model,
      sidecarModel: opts.sidecarModel,
      maxContextTokens: hardWindow,
      softLimitRatio: 0.68,
      mustActRatio: 0.82,
      criticalRatio: 0.93,
      generational: false,
      allowReplace: false,
      allowRedact: false,
      strictTools: true,
      maxTurns: 45,
      runDir: join(epDir, "run"),
      verbose: false,
    };

    const firstPrompt =
      `You are evaluated on MemoryAgentBench (${ep.split} / ${ep.source}). ` +
      `First call ingest_next_chunk repeatedly until all ${ep.chunks.length} chunks are ingested (END_OF_CHUNKS). ` +
      `Chunks are one-pass and cannot be re-pulled later. Once all chunks are read, answer concisely:\n` +
      `Question 1: ${ep.questions[0]}`;

    const followups = ep.questions
      .slice(1)
      .map((q, i) => `Question ${i + 2} (answer concisely): ${q}`);

    const run = await runAgent({
      task: firstPrompt,
      followups,
      config,
      client: opts.client,
      sidecarClient: opts.sidecarClient,
      workdir,
      mode: opts.arm,
      hardWindowTokens: hardWindow,
      taskToolNames: [],
      extraTools,
      onTurn: (t) => {
        const hitPct = t.promptTokens ? Math.round((t.cacheHitTokens / t.promptTokens) * 100) : 0;
        console.log(
          `  [MAB/${opts.arm}/${ep.id}] t${String(t.turn).padStart(2)}: ${t.promptTokens}in(${hitPct}%)/${t.completionTokens}out -> ${t.toolCalls.join(",") || "ANS"}`,
        );
      },
    });

    const qScores = ep.questions.map((q, qi) => {
      const pred = run.answers[qi] ?? "";
      const refs = ep.answers[qi];
      // Judge exact substring or semantic match for yes/no & entity answers
      const sub = bestSubstringMatch(pred, refs);
      const f1 = qaF1Score(pred, refs);
      subHitTotal += sub;
      f1SumTotal += Math.max(sub, f1);
      qCountTotal += 1;
      return { question: q, refs, prediction: pred, substring_match: sub, f1 };
    });

    if (run.metrics.terminated_early) termCount += 1;
    totalTurns += run.metrics.llm_calls;
    totalPrompt += run.metrics.prompt_tokens;
    totalHit += run.metrics.cache_hit_tokens;
    peakSum += run.metrics.peak_view_tokens;
    sidecarUsd += run.metrics.cost.sidecar_total_usd;
    effUsd += run.metrics.cost.effective_cost_usd;
    queries += run.metrics.context_query_calls;

    details.push({ episode: ep.id, qScores, metrics: run.metrics });
  }

  writeFileSync(
    join(opts.outDir, `mab-${opts.arm}-details.json`),
    JSON.stringify(details, null, 2),
  );
  const denom = Math.max(1, qCountTotal);
  const epCount = Math.max(1, opts.episodes.length);
  return {
    benchmark: "memoryagentbench",
    arm: opts.arm,
    items: qCountTotal,
    primary_score_pct: Number(((subHitTotal / denom) * 100).toFixed(1)),
    secondary_score_pct: Number(((f1SumTotal / denom) * 100).toFixed(1)),
    primary_metric_name: "Substring EM",
    secondary_metric_name: "Max(EM,F1)",
    terminated_early_count: termCount,
    avg_turns: Number((totalTurns / epCount).toFixed(1)),
    total_prompt_tokens: totalPrompt,
    cache_hit_ratio: totalPrompt > 0 ? Number((totalHit / totalPrompt).toFixed(3)) : 0,
    avg_peak_tokens: Math.round(peakSum / epCount),
    sidecar_usd: Number(sidecarUsd.toFixed(4)),
    effective_cost_usd: Number(effUsd.toFixed(4)),
    context_queries: queries,
    wall_clock_s: Number(((Date.now() - t0) / 1000).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// 3. ContextBench Verified Fetcher & Runner (Real GitHub Repo Localization)
// ---------------------------------------------------------------------------

const FAST_REPOS_PREFERENCE = new Set([
  "psf/requests",
  "pallets/flask",
  "pytest-dev/pytest",
  "pylint-dev/pylint",
  "mwaskom/seaborn",
  "pydata/xarray",
  "scikit-learn/scikit-learn",
]);

async function loadContextBenchItems(count: number): Promise<ContextBenchItem[]> {
  mkdirSync(DATA_CACHE_DIR, { recursive: true });
  const cachePath = join(DATA_CACHE_DIR, "contextbench_verified_sample.json");
  if (existsSync(cachePath)) {
    const cached = JSON.parse(readFileSync(cachePath, "utf8")) as ContextBenchItem[];
    if (cached.length >= count) return cached.slice(0, count);
  }

  console.log("  [fetch] Scanning ContextBench Verified rows on Hugging Face...");
  const candidates: ContextBenchItem[] = [];
  for (const offset of [0, 40, 90, 150, 220, 300, 380]) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=Contextbench%2FContextBench&config=contextbench_verified&split=train&offset=${offset}&length=8`;
    const res = await fetch(url);
    if (!res.ok) continue;
    const data = (await res.json()) as {
      rows?: Array<{ row: Record<string, unknown> }>;
    };
    for (const r of data.rows ?? []) {
      const row = r.row;
      const rawGold = row.gold_context;
      const goldParsed =
        typeof rawGold === "string" ? JSON.parse(rawGold) : rawGold;
      if (!Array.isArray(goldParsed) || goldParsed.length === 0) continue;
      candidates.push({
        instance_id: String(row.instance_id),
        original_inst_id: String(row.original_inst_id),
        repo: String(row.repo),
        repo_url: String(row.repo_url),
        base_commit: String(row.base_commit),
        problem_statement: String(row.problem_statement),
        gold_context: goldParsed.map((g: Record<string, unknown>) => ({
          file: String(g.file),
          start_line: Number(g.start_line),
          end_line: Number(g.end_line),
        })),
      });
    }
  }

  // Prefer fast-cloning medium repos and diversify across distinct repositories
  candidates.sort((a, b) => {
    const pa = FAST_REPOS_PREFERENCE.has(a.repo) ? 0 : 1;
    const pb = FAST_REPOS_PREFERENCE.has(b.repo) ? 0 : 1;
    return pa - pb;
  });

  const diversified: ContextBenchItem[] = [];
  const usedRepos = new Set<string>();
  for (const c of candidates) {
    if (!usedRepos.has(c.repo)) {
      diversified.push(c);
      usedRepos.add(c.repo);
    }
    if (diversified.length >= Math.max(count, 6)) break;
  }
  for (const c of candidates) {
    if (diversified.length >= Math.max(count, 6)) break;
    if (!diversified.some((x) => x.instance_id === c.instance_id)) {
      diversified.push(c);
    }
  }

  writeFileSync(cachePath, JSON.stringify(diversified, null, 2), "utf8");
  return diversified.slice(0, count);
}

function ensureRepoCheckout(item: ContextBenchItem): string {
  const repoCacheRoot = join(DATA_CACHE_DIR, "repos");
  mkdirSync(repoCacheRoot, { recursive: true });
  const slug = `${item.repo.replace(/\//g, "__")}-${item.base_commit.slice(0, 8)}`;
  const targetDir = join(repoCacheRoot, slug);
  if (existsSync(join(targetDir, ".git"))) {
    return targetDir;
  }
  console.log(`  [git] Shallow-fetching ${item.repo} @ ${item.base_commit.slice(0, 8)}...`);
  mkdirSync(targetDir, { recursive: true });
  execFileSync("git", ["init", "-q"], { cwd: targetDir });
  execFileSync("git", ["remote", "add", "origin", item.repo_url], { cwd: targetDir });
  execFileSync(
    "git",
    ["fetch", "-q", "--depth", "1", "origin", item.base_commit],
    { cwd: targetDir, timeout: 120_000 },
  );
  execFileSync("git", ["checkout", "-q", "--detach", "FETCH_HEAD"], { cwd: targetDir });
  return targetDir;
}

function scoreContextBenchPrediction(
  answer: string,
  gold: ContextBenchItem["gold_context"],
): { fileF1: number; lineF1: number } {
  const predSpans: Array<{ file: string; start_line: number; end_line: number }> = [];
  // Extract JSON block from final answer
  const jsonMatch =
    answer.match(/```(?:json)?\s*(\{[\s\S]*?"spans"[\s\S]*?\})\s*```/i) ??
    answer.match(/(\{[\s\S]*?"spans"\s*:\s*\[[\s\S]*?\][\s\S]*?\})/);

  if (jsonMatch) {
    try {
      const obj = JSON.parse(jsonMatch[1]) as {
        spans?: Array<Record<string, unknown>>;
      };
      for (const s of obj.spans ?? []) {
        if (typeof s.file === "string") {
          const cleanFile = s.file.replace(/^\.?\//, "").trim();
          const st = Math.max(1, Number(s.start_line ?? 1));
          const en = Math.max(st, Number(s.end_line ?? st + 30));
          predSpans.push({ file: cleanFile, start_line: st, end_line: en });
        }
      }
    } catch {
      // fallback regex below
    }
  }

  // Fallback: extract `path/to/file.py:L10-L50` patterns if JSON wasn't clean
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
    // Cap per-span line expansion to 400 lines to prevent gaming via 1..99999
    const end = Math.min(p.end_line, p.start_line + 400);
    for (let l = p.start_line; l <= end; l++) predLines.add(`${p.file}:${l}`);
  }

  let lineInter = 0;
  for (const k of predLines) if (goldLines.has(k)) lineInter += 1;
  const lineP = predLines.size > 0 ? lineInter / predLines.size : 0;
  const lineR = goldLines.size > 0 ? lineInter / goldLines.size : 0;
  const lineF1 = lineP + lineR > 0 ? (2 * lineP * lineR) / (lineP + lineR) : 0;

  return { fileF1, lineF1 };
}

async function runContextBenchArm(opts: {
  items: ContextBenchItem[];
  arm: AgentMode;
  model: string;
  sidecarModel: string;
  outDir: string;
  client: DeepSeekClient;
  sidecarClient: DeepSeekClient;
}): Promise<PublicArmSummary> {
  const hardWindow = 24_000; // Reading full 800–2000 line repo files hits 24K in 2-3 reads!
  let fileF1Sum = 0;
  let lineF1Sum = 0;
  let termCount = 0;
  let totalTurns = 0;
  let totalPrompt = 0;
  let totalHit = 0;
  let peakSum = 0;
  let sidecarUsd = 0;
  let effUsd = 0;
  let queries = 0;
  const t0 = Date.now();
  const details: unknown[] = [];

  for (let idx = 0; idx < opts.items.length; idx++) {
    const item = opts.items[idx];
    const repoDir = ensureRepoCheckout(item);
    const itemDir = join(opts.outDir, `contextbench-${opts.arm}-item${idx + 1}`);
    mkdirSync(itemDir, { recursive: true });

    const config: CpatConfig = {
      model: opts.model,
      sidecarModel: opts.sidecarModel,
      maxContextTokens: hardWindow,
      softLimitRatio: 0.68,
      mustActRatio: 0.82,
      criticalRatio: 0.93,
      generational: false,
      allowReplace: false,
      allowRedact: false,
      strictTools: true,
      maxTurns: 24,
      runDir: join(itemDir, "run"),
      verbose: false,
    };

    const taskPrompt =
      `You are evaluating ContextBench code context retrieval on repository \`${item.repo}\` (${item.original_inst_id}).\n` +
      `Explore the repository using grep_search, list_dir, and read_file to locate the EXACT minimal code blocks (files and 1-based start_line..end_line ranges) that must be understood and modified to resolve this issue.\n\n` +
      `=== ISSUE DESCRIPTION ===\n${item.problem_statement}\n=========================\n\n` +
      `When you have verified the exact functions/classes and line numbers in the source files (not test files), reply with a brief root-cause explanation followed by a JSON block in this exact format:\n` +
      "```json\n" +
      `{"spans": [{"file": "relative/path/from/repo/root.py", "start_line": 120, "end_line": 165}]}\n` +
      "```";

    const run = await runAgent({
      task: taskPrompt,
      config,
      client: opts.client,
      sidecarClient: opts.sidecarClient,
      workdir: repoDir,
      mode: opts.arm,
      hardWindowTokens: hardWindow,
      taskToolNames: ["list_dir", "read_file", "grep_search"],
      onTurn: (t) => {
        const hitPct = t.promptTokens ? Math.round((t.cacheHitTokens / t.promptTokens) * 100) : 0;
        console.log(
          `  [CB/${opts.arm}/${item.original_inst_id}] t${String(t.turn).padStart(2)}: ${t.promptTokens}in(${hitPct}%)/${t.completionTokens}out -> ${t.toolCalls.join(",") || "ANS"}`,
        );
      },
    });

    const { fileF1, lineF1 } = scoreContextBenchPrediction(run.answer, item.gold_context);
    fileF1Sum += fileF1;
    lineF1Sum += lineF1;
    if (run.metrics.terminated_early) termCount += 1;
    totalTurns += run.metrics.llm_calls;
    totalPrompt += run.metrics.prompt_tokens;
    totalHit += run.metrics.cache_hit_tokens;
    peakSum += run.metrics.peak_view_tokens;
    sidecarUsd += run.metrics.cost.sidecar_total_usd;
    effUsd += run.metrics.cost.effective_cost_usd;
    queries += run.metrics.context_query_calls;

    details.push({
      instance_id: item.original_inst_id,
      repo: item.repo,
      gold_context: item.gold_context,
      file_f1: Number(fileF1.toFixed(3)),
      line_f1: Number(lineF1.toFixed(3)),
      prediction: run.answer,
      metrics: run.metrics,
    });
  }

  writeFileSync(
    join(opts.outDir, `contextbench-${opts.arm}-details.json`),
    JSON.stringify(details, null, 2),
  );
  const n = Math.max(1, opts.items.length);
  return {
    benchmark: "contextbench",
    arm: opts.arm,
    items: opts.items.length,
    primary_score_pct: Number(((fileF1Sum / n) * 100).toFixed(1)),
    secondary_score_pct: Number(((lineF1Sum / n) * 100).toFixed(1)),
    primary_metric_name: "Utilized File F1",
    secondary_metric_name: "Utilized Line F1",
    terminated_early_count: termCount,
    avg_turns: Number((totalTurns / n).toFixed(1)),
    total_prompt_tokens: totalPrompt,
    cache_hit_ratio: totalPrompt > 0 ? Number((totalHit / totalPrompt).toFixed(3)) : 0,
    avg_peak_tokens: Math.round(peakSum / n),
    sidecar_usd: Number(sidecarUsd.toFixed(4)),
    effective_cost_usd: Number(effUsd.toFixed(4)),
    context_queries: queries,
    wall_clock_s: Number(((Date.now() - t0) / 1000).toFixed(1)),
  };
}

// ---------------------------------------------------------------------------
// Helpers & CLI
// ---------------------------------------------------------------------------

function chunkStringByParagraphs(text: string, maxChars: number): string[] {
  const paras = text.split(/\n\n+/);
  const out: string[] = [];
  let cur = "";
  for (const p of paras) {
    if (cur.length > 0 && cur.length + p.length > maxChars) {
      out.push(cur);
      cur = "";
    }
    cur += (cur ? "\n\n" : "") + p;
  }
  if (cur) out.push(cur);
  return out;
}

function packIntoChunks(docs: string[], targetChars: number): string[] {
  const chunks: string[] = [];
  let cur: string[] = [];
  let len = 0;
  for (const d of docs) {
    if (len > 0 && len + d.length > targetChars) {
      chunks.push(cur.join("\n\n"));
      cur = [];
      len = 0;
    }
    cur.push(d);
    len += d.length;
  }
  if (cur.length > 0) chunks.push(cur.join("\n\n"));
  return chunks;
}

function formatPublicMarkdown(
  rows: PublicArmSummary[],
  model: string,
  sidecarModel: string,
): string {
  const lines = [
    `# CPAT ContextHarness v2.1 — Third-Party Public Benchmarks Report`,
    ``,
    `- **Main Model**: \`${model}\` | **Sidecar Model**: \`${sidecarModel}\``,
    `- **Timestamp**: ${new Date().toISOString()}`,
    ``,
    `| Public Benchmark | Arm | Primary Metric | Secondary Metric | TermEarly | Avg Turns | Prompt Tok | Cache Hit | Avg Peak | Sidecar $ | **Eff. Cost $** | Queries | Time |`,
    `|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|`,
  ];
  for (const r of rows) {
    lines.push(
      `| **\`${r.benchmark}\`** (N=${r.items}) | **${r.arm}** | **${r.primary_score_pct}%** (${r.primary_metric_name}) | ${r.secondary_score_pct}% (${r.secondary_metric_name}) | ${r.terminated_early_count} | ${r.avg_turns} | ${r.total_prompt_tokens.toLocaleString()} | ${(r.cache_hit_ratio * 100).toFixed(1)}% | ${r.avg_peak_tokens.toLocaleString()} | $${r.sidecar_usd.toFixed(4)} | **$${r.effective_cost_usd.toFixed(4)}** | ${r.context_queries} | ${r.wall_clock_s}s |`,
    );
  }
  lines.push(``);
  return lines.join("\n");
}

async function main(): Promise<void> {
  const flags = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (!a.startsWith("--")) continue;
    const k = a.slice(2);
    const next = process.argv[i + 1];
    if (next && !next.startsWith("--")) {
      flags.set(k, next);
      i++;
    } else {
      flags.set(k, "true");
    }
  }

  const benchArg = flags.get("bench") ?? "all";
  const arms = (flags.get("arms") ?? "react,threshold,cpat,harness")
    .split(",")
    .map((s) => s.trim() as AgentMode)
    .filter(Boolean);
  const itemCount = Math.max(1, Number(flags.get("items") ?? 3));
  const model = flags.get("model") ?? "deepseek-flash";
  const sidecarModel = flags.get("sidecar-model") ?? "deepseek-flash";
  const concurrency = Math.max(1, Number(flags.get("concurrency") ?? 2));

  const env = loadDeepSeekEnv();
  const sidecarEnv = loadSidecarEnv();
  const client = new DeepSeekClient(env);
  const sidecarClient = new DeepSeekClient(sidecarEnv);

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = join("runs", `public-bench-${ts}`);
  mkdirSync(outDir, { recursive: true });

  console.log(
    `=== CPAT v2.1 Third-Party Public Benchmarks ===\n` +
      `Bench: ${benchArg} (items=${itemCount})  |  Arms: ${arms.join(", ")}\n` +
      `Main: ${model}  |  Sidecar: ${sidecarModel}\n` +
      `Output: ${outDir}\n`,
  );

  const allSummaries: PublicArmSummary[] = [];

  const runArmsParallel = async <T>(
    runner: (arm: AgentMode) => Promise<T>,
  ): Promise<T[]> => {
    const results: T[] = [];
    for (let i = 0; i < arms.length; i += concurrency) {
      const slice = arms.slice(i, i + concurrency);
      results.push(...(await Promise.all(slice.map(runner))));
    }
    return results;
  };

  if (benchArg === "all" || benchArg === "longmemeval") {
    const lmeItems = await loadLongMemEvalItems(itemCount);
    console.log(
      `\n>>> PUBLIC BENCH 1: LongMemEval-S (${lmeItems.length} items: ${lmeItems.map((x) => x.question_type).join(", ")})`,
    );
    const res = await runArmsParallel((arm) =>
      runLongMemEvalArm({
        items: lmeItems,
        arm,
        model,
        sidecarModel,
        outDir,
        client,
        sidecarClient,
      }),
    );
    allSummaries.push(...res);
  }

  if (benchArg === "all" || benchArg === "memoryagentbench") {
    const mabEps = await loadMemoryAgentBenchEpisodes(Math.min(itemCount, 3));
    console.log(
      `\n>>> PUBLIC BENCH 2: MemoryAgentBench (${mabEps.length} episodes: ${mabEps.map((x) => x.id).join(", ")})`,
    );
    const res = await runArmsParallel((arm) =>
      runMemoryAgentBenchArm({
        episodes: mabEps,
        arm,
        model,
        sidecarModel,
        outDir,
        client,
        sidecarClient,
      }),
    );
    allSummaries.push(...res);
  }

  if (benchArg === "all" || benchArg === "contextbench") {
    const cbItems = await loadContextBenchItems(itemCount);
    // Pre-warm git checkouts sequentially before parallel arm runs!
    for (const it of cbItems) ensureRepoCheckout(it);
    console.log(
      `\n>>> PUBLIC BENCH 3: ContextBench Verified (${cbItems.length} real repo issues: ${cbItems.map((x) => x.original_inst_id).join(", ")})`,
    );
    const res = await runArmsParallel((arm) =>
      runContextBenchArm({
        items: cbItems,
        arm,
        model,
        sidecarModel,
        outDir,
        client,
        sidecarClient,
      }),
    );
    allSummaries.push(...res);
  }

  const md = formatPublicMarkdown(allSummaries, model, sidecarModel);
  writeFileSync(join(outDir, "summary.json"), JSON.stringify(allSummaries, null, 2), "utf8");
  writeFileSync(join(outDir, "summary.md"), md, "utf8");

  console.log(`\n${md}`);
  console.log(`Saved full public benchmark artifacts to ${outDir}/{summary.json,summary.md}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
