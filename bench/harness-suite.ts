/**
 * Dedicated 3-Suite Context Engineering Benchmark for CPAT ContextHarness.
 *
 * Evaluates all four pillars (Context, Behavior, Effectiveness, True Cost) across
 * four arms (`react` | `threshold` | `cpat` [v1 inline tool] | `harness` [v2]):
 *
 *   1. `bounded-research` (32K bounded window, ~75K multi-dossier corpus):
 *      Tests global aggregation, 2-hop dependency chaining, and deep-buried
 *      section-7 code revisit (`INC-...` + `SHA-...`) with grep disabled.
 *      Directly tests CPAT v1's historical 32K small-window cost bottleneck.
 *
 *   2. `ephemeral-stream` (22K bounded window, ~48K one-shot telemetry stream):
 *      Stateful `pull_telemetry_batch` cursor that CANNOT be re-pulled once
 *      consumed, carrying 160 fine-grained `TRC-...` trace spans buried past
 *      char 9,000. Directly tests lossless indexed recovery (`context_query`)
 *      where static file re-reads are impossible and pure summaries drop needles.
 *
 *   3. `pollution-spec` (22K bounded window, decoy staging logs + mid-flight
 *      user spec override):
 *      Lures initial exploration into 11K+ tokens of staging `RedisShardPool`
 *      decoy logs (`SLA=500ms`), then issues an authoritative user override to
 *      `LocalRingBuffer` (`SLA=150ms`). Tests AC3 isolated pollution purging,
 *      subtask isolation, and freedom from historical anchoring.
 *
 * Usage:
 *   node bench/harness-suite.ts [--suite all|bounded-research|ephemeral-stream|pollution-spec]
 *                               [--arms react,threshold,cpat,harness]
 *                               [--model deepseek-flash] [--sidecar-model deepseek-flash]
 *                               [--smoke] [--dry-run]
 */

import { mkdirSync, writeFileSync } from "node:fs";
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
import { normalizeAnswer } from "./f1.ts";

type SuiteName = "bounded-research" | "ephemeral-stream" | "pollution-spec";

interface EvalItem {
  id: string;
  category: string;
  prompt: string;
  /** Every token in `mustInclude` must appear (normalized) in the answer. */
  mustInclude: string[];
  /** Any token in `mustExclude` appearing in the answer fails the item (tests pollution resistance). */
  mustExclude?: string[];
}

interface SuiteSpec {
  name: SuiteName;
  description: string;
  hardWindow: number;
  maxTurns: number;
  corpusTokensEst: number;
  taskToolNames: string[];
  setupWorkdir: (dir: string) => { extraToolsFactory?: () => CustomToolSuite };
  items: EvalItem[];
}

interface ArmSuiteResult {
  suite: SuiteName;
  arm: AgentMode;
  hard_window: number;
  corpus_tokens_est: number;
  accuracy_pct: number;
  revisit_or_needle_pct: number;
  items_passed: number;
  items_total: number;
  terminated_early: boolean;
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  cache_hit_ratio: number;
  peak_view_tokens: number;
  mean_view_tokens: number;
  effective_cost_usd: number;
  sidecar_cost_usd: number;
  sidecar_calls: number;
  context_query_calls: number;
  scopes_submitted: number;
  agent_patches_applied: number;
  runtime_fallback_offloads: number;
  critical_path_ms: number;
  wall_clock_s: number;
  per_item: Array<{
    id: string;
    category: string;
    passed: boolean;
    mustInclude: string[];
    mustExclude?: string[];
    prediction: string;
  }>;
  api_error?: string;
}

// ---------------------------------------------------------------------------
// Suite 1: Bounded-Window Deep Research (32K window, 16 dossiers ~76K tokens)
// ---------------------------------------------------------------------------

const PROJECTS = [
  "Aurora", "Basilisk", "Cobalt", "Drifter", "Ember", "Falcon", "Granite", "Halcyon",
  "Ironwood", "Juniper", "Kestrel", "Lantern", "Meridian", "Nimbus", "Obsidian", "Quasar",
];

const RESEARCHERS = [
  "Dr. Reyes", "Dr. Nakamura", "Dr. Okonkwo", "Dr. Volkov", "Dr. Singh", "Dr. Larsson",
  "Dr. Haddad", "Dr. Costa", "Dr. Park", "Dr. Bauer", "Dr. Moreau", "Dr. Ivanova",
  "Dr. Cohen", "Dr. Adebayo", "Dr. Tanaka", "Dr. Fischer",
];

function buildBoundedResearchSuite(smoke: boolean): SuiteSpec {
  const count = smoke ? 6 : 14;
  const hardWindow = smoke ? 15_000 : 32_000;
  const docTargetChars = smoke ? 11_000 : 16_500; // ~3.3K (smoke) or ~5K (full) tok/doc

  const docs = Array.from({ length: count }, (_, i) => {
    const codename = PROJECTS[i];
    const lead = RESEARCHERS[i];
    // Deterministic distinct accuracy values: highest at slot count-2, lowest at slot 3
    const rawAcc = 61.2 + ((i * 17 + 5) % count) * 2.4;
    const accuracy = `${rawAcc.toFixed(1)}%`;
    const depIdx = (i + (count <= 6 ? 2 : 3)) % count;
    const dependsOn = PROJECTS[depIdx];
    const incidentCode = `INC-${4100 + i * 137}${String.fromCharCode(65 + (i % 20))}`;
    const checksum = `SHA-${(0x9a1000 + i * 0x37f1).toString(16).toUpperCase()}`;
    return {
      slot: i + 1,
      codename,
      lead,
      rawAcc,
      accuracy,
      depIdx,
      dependsOn,
      incidentCode,
      checksum,
    };
  });

  const renderDoc = (d: (typeof docs)[number]): string => {
    const header = [
      `# Technical Dossier ${d.slot}: Project ${d.codename}`,
      `Lead Researcher: ${d.lead}`,
      `Primary Benchmark Accuracy: ${d.accuracy}`,
      `Upstream Architecture Dependency: Project ${d.dependsOn}`,
      ``,
      `## Sections 1–6: Experimental Methodology & Ablation Notes`,
    ].join("\n");

    const fillerUnit =
      `Subsystem telemetry for ${d.codename} confirms steady pipeline throughput across distributed workers, ` +
      `maintaining deterministic checkpoint serialization, bounded queue backpressure, and repeatable gradient ` +
      `accumulation steps under synthetic load profiles without packet loss or schema skew.\n`;

    const footer = [
      ``,
      `## Section 7: Post-Mortem & Release Audit (Deep Archive Record)`,
      `Post-Mortem Incident Code: ${d.incidentCode}`,
      `Verified Release Checksum: ${d.checksum}`,
    ].join("\n");

    const needed = Math.max(800, docTargetChars - header.length - footer.length);
    const repeats = Math.ceil(needed / fillerUnit.length);
    return `${header}\n${fillerUnit.repeat(repeats)}${footer}\n`;
  };

  const sampleRendered = docs.map(renderDoc);
  const corpusTokensEst = sampleRendered.reduce((s, t) => s + estimateTokens(t), 0);

  const best = [...docs].sort((a, b) => b.rawAcc - a.rawAcc)[0];
  const worst = [...docs].sort((a, b) => a.rawAcc - b.rawAcc)[0];
  const hop1 = docs[docs[0].depIdx];
  const hop2 = docs[hop1.depIdx];
  const revA = docs[1]; // Basilisk (early doc #2)
  const revB = docs[Math.min(count - 1, 4)]; // Ember (mid doc #5)

  const items: EvalItem[] = [
    {
      id: "Q1_global_max",
      category: "aggregation",
      prompt:
        `There are ${count} project dossiers in corpus/ (doc_1.txt .. doc_${count}.txt). ` +
        `Read ALL ${count} dossiers using read_file (do not skip any). ` +
        `Which project achieved the HIGHEST Primary Benchmark Accuracy across all ${count} dossiers, and what is its exact percentage?`,
      mustInclude: [best.codename, best.accuracy],
    },
    {
      id: "Q2_two_hop",
      category: "multi_hop",
      prompt:
        `Follow Project ${docs[0].codename}'s Upstream Architecture Dependency two hops ` +
        `(${docs[0].codename} -> hop1 -> hop2). What is hop2's project codename, Lead Researcher, and exact Primary Benchmark Accuracy?`,
      mustInclude: [hop2.codename, hop2.lead, hop2.accuracy],
    },
    {
      id: "Q3_revisit_early",
      category: "revisit",
      prompt:
        `Revisit Project ${revA.codename} (from doc_${revA.slot}.txt, examined near the start): ` +
        `report its exact Section 7 Post-Mortem Incident Code (INC-...) and Verified Release Checksum (SHA-...).`,
      mustInclude: [revA.incidentCode, revA.checksum],
    },
    ...(smoke
      ? []
      : [
          {
            id: "Q4_global_min",
            category: "aggregation",
            prompt:
              `Across all ${count} dossiers, which project had the LOWEST Primary Benchmark Accuracy, who is its Lead Researcher, and what is that exact percentage?`,
            mustInclude: [worst.codename, worst.lead, worst.accuracy],
          },
          {
            id: "Q5_revisit_mid",
            category: "revisit",
            prompt:
              `Revisit Project ${revB.codename} (doc_${revB.slot}.txt): report its exact Section 7 Post-Mortem Incident Code (INC-...) and Verified Release Checksum (SHA-...).`,
            mustInclude: [revB.incidentCode, revB.checksum],
          },
        ]),
  ];

  return {
    name: "bounded-research",
    description: `Bounded ${hardWindow}-token window over ${count} whole-read dossiers (~${corpusTokensEst} tok)`,
    hardWindow,
    maxTurns: smoke ? 36 : 75,
    corpusTokensEst,
    taskToolNames: ["list_dir", "read_file"],
    setupWorkdir: (dir: string) => {
      const cdir = join(dir, "corpus");
      mkdirSync(cdir, { recursive: true });
      docs.forEach((d, idx) => {
        writeFileSync(join(cdir, `doc_${d.slot}.txt`), sampleRendered[idx], "utf8");
      });
      return {};
    },
    items,
  };
}

// ---------------------------------------------------------------------------
// Suite 2: Ephemeral Non-Rereadable Telemetry Stream (Verbatim Needle Recall)
// ---------------------------------------------------------------------------

function buildEphemeralStreamSuite(smoke: boolean): SuiteSpec {
  const batchCount = smoke ? 5 : 10;
  const spansPerBatch = 16;
  const hardWindow = smoke ? 13_000 : 22_000;
  const targetCharsPerBatch = smoke ? 10_500 : 14_500;

  const SERVICES = [
    "svc-auth-edge", "svc-ledger-core", "svc-order-router", "svc-risk-engine", "svc- billing-sync".replace(" ", ""),
    "svc-quota-gate", "svc-token-vault", "svc-index-shard", "svc-stream-mux", "svc-audit-sink",
  ];

  const batches = Array.from({ length: batchCount }, (_, bIdx) => {
    const batchNum = bIdx + 1;
    const errRateNum = Number((1.4 + ((bIdx * 7 + 3) % batchCount) * 0.85).toFixed(2));
    const errRate = `${errRateNum.toFixed(2)}%`;
    const degradedSvc = SERVICES[bIdx % SERVICES.length];

    const spans = Array.from({ length: spansPerBatch }, (__, sIdx) => {
      const spanNum = sIdx + 1;
      const traceId = `TRC-${String(batchNum).padStart(2, "0")}${String(spanNum).padStart(2, "0")}`;
      const node = `worker-az${(bIdx % 3) + 1}-${10 + spanNum}`;
      const p99 = 310 + bIdx * 47 + spanNum * 19;
      const fault = `FLT-${300 + batchNum * 31 + spanNum * 7}${String.fromCharCode(68 + (sIdx % 18))}`;
      return { traceId, node, p99, fault };
    });

    return { batchNum, errRateNum, errRate, degradedSvc, spans };
  });

  const renderBatch = (b: (typeof batches)[number]): string => {
    const head = [
      `=== TELEMETRY STREAM BATCH #${b.batchNum} OF ${batchCount} (ONE-SHOT CURSOR — NON-REREADABLE) ===`,
      `Window Peak Error Rate: ${b.errRate}`,
      `Primary Degraded Service: ${b.degradedSvc}`,
      `--- Gateway Access & Health Noise ---`,
    ].join("\n");

    const noiseLine =
      `[INFO] gateway-ingress status=200 upstream_ok=true keepalive=warm tls=1.3 bytes_tx=4096 retry=0\n`;

    // Place the 16 fine-grained trace spans past char 7,000+ so 6,000-char
    // blind head-slicing (artifact_get) and 300-char threshold previews miss them!
    const spanBlock = [
      `--- Distributed Slow-Trace Span Table (${b.spans.length} spans) ---`,
      ...b.spans.map(
        (s) =>
          `SPAN_RECORD trace_id=${s.traceId} | node=${s.node} | p99_ms=${s.p99} | fault_code=${s.fault}`,
      ),
      `=== END OF BATCH #${b.batchNum} ===`,
    ].join("\n");

    const needed = Math.max(6_800, targetCharsPerBatch - head.length - spanBlock.length);
    const noiseRepeats = Math.ceil(needed / noiseLine.length);
    return `${head}\n${noiseLine.repeat(noiseRepeats)}${spanBlock}\n`;
  };

  const renderedBatches = batches.map(renderBatch);
  const corpusTokensEst = renderedBatches.reduce((s, t) => s + estimateTokens(t), 0);

  const worstBatch = [...batches].sort((a, b) => b.errRateNum - a.errRateNum)[0];
  const needle1 = batches[1].spans[13]; // Batch #2, span #14 -> TRC-0214 (deep in table!)
  const needle2 = batches[Math.min(batchCount - 1, 3)].spans[14]; // Batch #4, span #15 -> TRC-0415
  const cmpA = batches[0].spans[11]; // Batch #1, span #12 -> TRC-0112
  const cmpB = batches[batchCount - 2].spans[9]; // late batch, span #10

  const pullToolDef: ToolDefinition = {
    type: "function",
    function: {
      name: "pull_telemetry_batch",
      description:
        "Pull the next unconsumed telemetry batch from the live incident stream. " +
        "IMPORTANT: Stateful one-shot cursor — each batch is returned ONLY ONCE and can NEVER be re-pulled later.",
      parameters: {
        type: "object",
        properties: {},
      },
    },
  };

  const items: EvalItem[] = [
    {
      id: "Q1_stream_peak",
      category: "aggregation",
      prompt:
        `Call pull_telemetry_batch repeatedly until all ${batchCount} batches have been consumed (END_OF_STREAM). ` +
        `Note: batches are one-shot and CANNOT be re-pulled once consumed! ` +
        `Across all ${batchCount} batches, which Batch # had the highest Window Peak Error Rate, what was its exact percentage, and what was its Primary Degraded Service?`,
      mustInclude: [String(worstBatch.batchNum), worstBatch.errRate, worstBatch.degradedSvc],
    },
    {
      id: "Q2_needle_batch2",
      category: "revisit",
      prompt:
        `From earlier Batch #2 (already consumed from the stream), report the exact node, p99_ms integer, and fault_code (FLT-...) for trace_id=${needle1.traceId}.`,
      mustInclude: [needle1.node, String(needle1.p99), needle1.fault],
    },
    ...(smoke
      ? []
      : [
          {
            id: "Q3_needle_batch4",
            category: "revisit",
            prompt:
              `From earlier Batch #4, report the exact node, p99_ms integer, and fault_code (FLT-...) for trace_id=${needle2.traceId}.`,
            mustInclude: [needle2.node, String(needle2.p99), needle2.fault],
          },
          {
            id: "Q4_cross_batch_needle",
            category: "revisit",
            prompt:
              `Look up trace_id=${cmpA.traceId} (Batch #1) and trace_id=${cmpB.traceId} (Batch #${batchCount - 1}): ` +
              `report both exact p99_ms integers and both fault_code values.`,
            mustInclude: [String(cmpA.p99), cmpA.fault, String(cmpB.p99), cmpB.fault],
          },
        ]),
  ];

  return {
    name: "ephemeral-stream",
    description: `One-shot non-rereadable stream (${batchCount} batches ~${corpusTokensEst} tok in ${hardWindow} window, ${batchCount * spansPerBatch} buried TRC spans)`,
    hardWindow,
    maxTurns: smoke ? 28 : 55,
    corpusTokensEst,
    taskToolNames: [], // NO filesystem tools — strictly ephemeral stream!
    setupWorkdir: () => ({
      extraToolsFactory: (): CustomToolSuite => {
        let cursor = 0;
        return {
          defs: [pullToolDef],
          dispatch: (name: string) => {
            if (name !== "pull_telemetry_batch") return undefined;
            if (cursor >= renderedBatches.length) {
              return `END_OF_STREAM: All ${renderedBatches.length} batches have already been consumed. Earlier batches CANNOT be re-pulled from the live stream.`;
            }
            const out = renderedBatches[cursor];
            cursor += 1;
            return out;
          },
        };
      },
    }),
    items,
  };
}

// ---------------------------------------------------------------------------
// Suite 3: Context Pollution & Mid-Flight Spec Override (AC3 + Scope Isolation)
// ---------------------------------------------------------------------------

function buildPollutionSpecSuite(smoke: boolean): SuiteSpec {
  const hardWindow = smoke ? 15_000 : 22_000;
  const decoyChars = smoke ? 16_000 : 24_000; // ~5K (smoke) or ~7.3K (full) tok per bulky file

  const padLog = (header: string, line: string, footer: string): string => {
    const n = Math.ceil(Math.max(1000, decoyChars - header.length - footer.length) / line.length);
    return `${header}\n${line.repeat(n)}${footer}\n`;
  };

  const ticket = [
    `INCIDENT TICKET #INC-901`,
    `Initial hypothesis from on-call: tail latency spikes above SLA_CAP=500ms.`,
    `Files in workspace:`,
    `  - logs/staging_gateway.log`,
    `  - logs/staging_redis_dump.log`,
    `  - logs/legacy_memcached.log`,
    `  - prod/config.json`,
    `  - prod/ring_buffer.ts`,
  ].join("\n");

  const stagingGw = padLog(
    `=== STAGING GATEWAY LOG (env=staging, configured SLA_CAP=500ms) ===`,
    `[WARN] env=staging backend=RedisShardPool retry_backoff=120ms error=ERR-REDIS-909 timeout exceeding 500ms SLA_CAP\n`,
    `=== SUMMARY: 412 occurrences of ERR-REDIS-909 in RedisShardPool under SLA_CAP=500ms ===`,
  );

  const stagingRedis = padLog(
    `=== STAGING REDIS SHARD DUMP (env=staging, backend=RedisShardPool) ===`,
    `[TRACE] RedisShardPool slot_migration=stalled shard=07 client_timeout=500ms code=ERR-REDIS-909\n`,
    `=== DUMP END: Root cause in staging environment is RedisShardPool ERR-REDIS-909 (required_slots_500ms=6000) ===`,
  );

  const legacyMem = padLog(
    `=== LEGACY MEMCACHED AUDIT LOG (archived subsystem) ===`,
    `[INFO] memcached_pool active_connections=0 traffic_share=0.0% status=decommissioned_in_2025\n`,
    `=== AUDIT CONCLUSION: prod traffic_share=0.0%; memcached is completely unused in production ===`,
  );

  const prodConfig = JSON.stringify(
    {
      environment: "production",
      queue_backend: "LocalRingBuffer",
      redis_enabled: false,
      memcached_enabled: false,
      throughput_rps: 4000,
      safety_factor: 3,
      legacy_sla_ms_deprecated: 500,
      note: "Compute required_slots = Math.ceil((throughput_rps * active_sla_ms / 1000) * safety_factor)",
    },
    null,
    2,
  );

  const prodRingBuffer = padLog(
    [
      `// Production Queue Backend: LocalRingBuffer`,
      `export class LocalRingBuffer {`,
      `  constructor(private readonly capacity: number) {}`,
      `  // DEFECT MARKER [BUG-RB-404]: modulo wrap uses (head + 1) % (capacity - 1), dropping the last slot!`,
      `  nextIndex(head: number): number { return (head + 1) % (this.capacity - 1); }`,
      `}`,
      `/* Detailed production soak telemetry below */`,
    ].join("\n"),
    `// soak-trace prod LocalRingBuffer enqueue_ns=410 consumer_lag=nominal wrap_anomaly_ref=BUG-RB-404\n`,
    `// End of prod/ring_buffer.ts — verified defect ID: BUG-RB-404`,
  );

  const corpusTokensEst =
    estimateTokens(ticket) +
    estimateTokens(stagingGw) +
    estimateTokens(stagingRedis) +
    estimateTokens(legacyMem) +
    estimateTokens(prodConfig) +
    estimateTokens(prodRingBuffer);

  // Under authoritative override active_sla_ms = 150:
  // required_slots = (4000 * 150 / 1000) * 3 = 1800! (Polluted 500ms gives 6000!)
  const items: EvalItem[] = [
    {
      id: "Q1_initial_staging_scan",
      category: "initial_lure",
      prompt:
        `Read incident_ticket.txt, logs/staging_gateway.log, and logs/staging_redis_dump.log using read_file. ` +
        `What backend class name, error code (ERR-...), and SLA_CAP (in ms) are reported in those staging logs?`,
      mustInclude: ["RedisShardPool", "ERR-REDIS-909", "500"],
    },
    {
      id: "Q2_override_and_prod_fix",
      category: "pollution_override",
      prompt:
        `WAIT — critical correction! Ignore all staging logs, do NOT use RedisShardPool, and discard the old 500ms SLA! ` +
        `Those were staging decoys. Authoritative production spec: active_sla_ms is now 150ms (NOT 500ms). ` +
        `Read logs/legacy_memcached.log, prod/config.json, and prod/ring_buffer.ts using read_file, then state ONLY ` +
        `(do NOT mention superseded staging backend names or old 500/6000 staging numbers in your reply): ` +
        `(1) the true production backend class name, (2) the exact defect marker ID (BUG-...) in ring_buffer.ts, and ` +
        `(3) the exact integer required_slots under active_sla_ms=150.`,
      mustInclude: ["LocalRingBuffer", "BUG-RB-404", "1800"],
      mustExclude: ["6000"],
    },
    {
      id: "Q3_clean_consistency",
      category: "revisit",
      prompt:
        `Final verification (answer concisely without mentioning superseded staging backends or old slot counts): ` +
        `state ONLY the active production backend class, the active SLA in ms, the exact required_slots integer, and the BUG-... ID.`,
      mustInclude: ["LocalRingBuffer", "150", "1800", "BUG-RB-404"],
      mustExclude: ["RedisShardPool", "6000", "ERR-REDIS-909"],
    },
  ];

  return {
    name: "pollution-spec",
    description: `Decoy staging logs + mid-flight user spec override (~${corpusTokensEst} tok in ${hardWindow} window)`,
    hardWindow,
    maxTurns: smoke ? 25 : 40,
    corpusTokensEst,
    taskToolNames: ["list_dir", "read_file"],
    setupWorkdir: (dir: string) => {
      mkdirSync(join(dir, "logs"), { recursive: true });
      mkdirSync(join(dir, "prod"), { recursive: true });
      writeFileSync(join(dir, "incident_ticket.txt"), ticket, "utf8");
      writeFileSync(join(dir, "logs/staging_gateway.log"), stagingGw, "utf8");
      writeFileSync(join(dir, "logs/staging_redis_dump.log"), stagingRedis, "utf8");
      writeFileSync(join(dir, "logs/legacy_memcached.log"), legacyMem, "utf8");
      writeFileSync(join(dir, "prod/config.json"), prodConfig, "utf8");
      writeFileSync(join(dir, "prod/ring_buffer.ts"), prodRingBuffer, "utf8");
      return {};
    },
    items,
  };
}

// ---------------------------------------------------------------------------
// Scoring & Runner
// ---------------------------------------------------------------------------

function scoreItem(prediction: string, item: EvalItem): boolean {
  if (!prediction.trim()) return false;
  const normPred = normalizeAnswer(prediction);
  const rawLower = prediction.toLowerCase();

  for (const req of item.mustInclude) {
    const normReq = normalizeAnswer(req);
    if (!normPred.includes(normReq) && !rawLower.includes(req.toLowerCase())) {
      return false;
    }
  }
  if (item.mustExclude) {
    for (const bad of item.mustExclude) {
      if (rawLower.includes(bad.toLowerCase())) {
        return false;
      }
    }
  }
  return true;
}

async function runSingleArm(opts: {
  suite: SuiteSpec;
  arm: AgentMode;
  model: string;
  sidecarModel: string;
  baseOutDir: string;
  client: DeepSeekClient;
  sidecarClient: DeepSeekClient;
}): Promise<ArmSuiteResult> {
  const { suite, arm, model, sidecarModel, baseOutDir, client, sidecarClient } = opts;
  const armDir = join(baseOutDir, `${suite.name}-${arm}`);
  const workdir = join(armDir, "workspace");
  mkdirSync(workdir, { recursive: true });

  const { extraToolsFactory } = suite.setupWorkdir(workdir);
  const extraTools = extraToolsFactory?.();

  const config: CpatConfig = {
    model,
    sidecarModel,
    maxContextTokens: suite.hardWindow,
    softLimitRatio: 0.68,
    mustActRatio: 0.82,
    criticalRatio: 0.93,
    generational: false,
    allowReplace: false,
    allowRedact: false,
    strictTools: true,
    maxTurns: suite.maxTurns,
    runDir: join(armDir, "run"),
    verbose: false,
  };

  const t0 = Date.now();
  let run: (RunResult & { runtime?: unknown; harness?: unknown }) | undefined;
  let apiError: string | undefined;

  try {
    run = await runAgent({
      task: suite.items[0].prompt,
      followups: suite.items.slice(1).map((it) => it.prompt),
      config,
      client,
      sidecarClient,
      workdir: resolve(workdir),
      mode: arm,
      hardWindowTokens: suite.hardWindow,
      taskToolNames: suite.taskToolNames,
      extraTools,
      onTurn: (t) => {
        const cachePct =
          t.promptTokens > 0
            ? `${Math.round((t.cacheHitTokens / t.promptTokens) * 100)}%`
            : "0%";
        const patchNote = t.patchSummary ? ` [${t.patchSummary}]` : "";
        console.log(
          `  [${suite.name}/${arm}] t${String(t.turn).padStart(2)}: ${t.promptTokens}in(hit ${cachePct})/${t.completionTokens}out -> ${t.toolCalls.join(",") || "ANSWER"}${patchNote}`,
        );
      },
    });
  } catch (err) {
    apiError = err instanceof Error ? err.message : String(err);
    console.error(`  [${suite.name}/${arm}] ERROR: ${apiError}`);
  }

  const wallClockS = Number(((Date.now() - t0) / 1000).toFixed(1));
  const perItem = suite.items.map((item, idx) => {
    const pred = run?.answers[idx] ?? "";
    return {
      id: item.id,
      category: item.category,
      passed: scoreItem(pred, item),
      mustInclude: item.mustInclude,
      mustExclude: item.mustExclude,
      prediction: pred,
    };
  });

  const passed = perItem.filter((p) => p.passed).length;
  const revisitItems = perItem.filter(
    (p) => p.category === "revisit" || p.category === "pollution_override",
  );
  const revisitPassed = revisitItems.filter((p) => p.passed).length;

  const m = run?.metrics;
  const res: ArmSuiteResult = {
    suite: suite.name,
    arm,
    hard_window: suite.hardWindow,
    corpus_tokens_est: suite.corpusTokensEst,
    accuracy_pct: Number(((passed / suite.items.length) * 100).toFixed(1)),
    revisit_or_needle_pct:
      revisitItems.length > 0
        ? Number(((revisitPassed / revisitItems.length) * 100).toFixed(1))
        : 100,
    items_passed: passed,
    items_total: suite.items.length,
    terminated_early: m?.terminated_early ?? false,
    llm_calls: m?.llm_calls ?? 0,
    prompt_tokens: m?.prompt_tokens ?? 0,
    completion_tokens: m?.completion_tokens ?? 0,
    cache_hit_ratio: m?.cache_hit_ratio ?? 0,
    peak_view_tokens: m?.peak_view_tokens ?? 0,
    mean_view_tokens: m?.mean_view_tokens ?? 0,
    effective_cost_usd: m?.cost.effective_cost_usd ?? 0,
    sidecar_cost_usd: m?.cost.sidecar_total_usd ?? 0,
    sidecar_calls: m?.cost.sidecar_calls ?? 0,
    context_query_calls: m?.context_query_calls ?? 0,
    scopes_submitted: m?.scopes_submitted ?? 0,
    agent_patches_applied: m?.agent_patches_applied ?? 0,
    runtime_fallback_offloads: m?.runtime_fallback_offloads ?? 0,
    critical_path_ms: m?.cost.harness_critical_path_ms ?? 0,
    wall_clock_s: wallClockS,
    per_item: perItem,
    ...(apiError ? { api_error: apiError } : {}),
  };

  writeFileSync(join(armDir, "result.json"), JSON.stringify(res, null, 2), "utf8");
  return res;
}

function renderMarkdownReport(
  results: ArmSuiteResult[],
  model: string,
  sidecarModel: string,
  smoke: boolean,
): string {
  const lines: string[] = [
    `# CPAT ContextHarness Dedicated Benchmark Report (${smoke ? "Smoke" : "Full"})`,
    ``,
    `- **Main Model**: \`${model}\` | **Sidecar Model**: \`${sidecarModel}\``,
    `- **Timestamp**: ${new Date().toISOString()}`,
    ``,
    `| Suite | Arm | TermEarly | Acc % | Revisit/Override % | Turns | Prompt Tok | Cache Hit | Peak Tok | Mean Tok | Sidecar $ | Total Eff. $ | Queries |`,
    `|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|`,
  ];

  for (const r of results) {
    lines.push(
      `| \`${r.suite}\` | **${r.arm}** | ${r.terminated_early ? "YES" : "no"} | **${r.accuracy_pct}%** (${r.items_passed}/${r.items_total}) | ${r.revisit_or_needle_pct}% | ${r.llm_calls} | ${r.prompt_tokens.toLocaleString()} | ${(r.cache_hit_ratio * 100).toFixed(1)}% | ${r.peak_view_tokens.toLocaleString()} | ${r.mean_view_tokens.toLocaleString()} | $${r.sidecar_cost_usd.toFixed(4)} | **$${r.effective_cost_usd.toFixed(4)}** | ${r.context_query_calls} |`,
    );
  }
  lines.push(``);
  return lines.join("\n");
}

function parseCli(argv: string[]) {
  const flags = new Map<string, string>();
  const bools = new Set<string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags.set(key, next);
      i++;
    } else {
      bools.add(key);
    }
  }
  const suiteArg = flags.get("suite") ?? "all";
  const arms = (flags.get("arms") ?? "react,threshold,cpat,harness")
    .split(",")
    .map((s) => s.trim() as AgentMode)
    .filter(Boolean);
  const model = flags.get("model") ?? "deepseek-flash";
  const sidecarModel = flags.get("sidecar-model") ?? "deepseek-flash";
  const smoke = bools.has("smoke");
  const dryRun = bools.has("dry-run");
  const concurrency = Math.max(1, Number(flags.get("concurrency") ?? 2));
  return { suiteArg, arms, model, sidecarModel, smoke, dryRun, concurrency };
}

async function main(): Promise<void> {
  const cli = parseCli(process.argv.slice(2));
  const allSuites: SuiteSpec[] = [
    buildBoundedResearchSuite(cli.smoke),
    buildEphemeralStreamSuite(cli.smoke),
    buildPollutionSpecSuite(cli.smoke),
  ];
  const selectedSuites =
    cli.suiteArg === "all"
      ? allSuites
      : allSuites.filter((s) => s.name === cli.suiteArg);

  if (selectedSuites.length === 0) {
    throw new Error(`Unknown --suite "${cli.suiteArg}"`);
  }

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = join("runs", `harness-suite-${cli.smoke ? "smoke" : "full"}-${ts}`);
  mkdirSync(outDir, { recursive: true });

  if (cli.dryRun) {
    for (const s of selectedSuites) {
      console.log(`[dry-run] ${s.name}: ${s.description}`);
      for (const it of s.items) {
        console.log(`  - ${it.id} (${it.category}): mustInclude=${JSON.stringify(it.mustInclude)} mustExclude=${JSON.stringify(it.mustExclude ?? [])}`);
      }
    }
    return;
  }

  const env = loadDeepSeekEnv();
  const sidecarEnv = loadSidecarEnv();
  const client = new DeepSeekClient(env);
  const sidecarClient = new DeepSeekClient(sidecarEnv);

  console.log(
    `=== CPAT ContextHarness Dedicated Suite (${cli.smoke ? "SMOKE" : "FULL"}) ===\n` +
      `Main: ${cli.model}  |  Sidecar: ${cli.sidecarModel}  |  Arms: ${cli.arms.join(", ")}\n` +
      `Output: ${outDir}\n`,
  );

  const allResults: ArmSuiteResult[] = [];

  for (const suite of selectedSuites) {
    console.log(`\n>>> SUITE: ${suite.name} — ${suite.description}`);
    // Run arms in chunks of `cli.concurrency`
    for (let i = 0; i < cli.arms.length; i += cli.concurrency) {
      const batch = cli.arms.slice(i, i + cli.concurrency);
      const batchRes = await Promise.all(
        batch.map((arm) =>
          runSingleArm({
            suite,
            arm,
            model: cli.model,
            sidecarModel: cli.sidecarModel,
            baseOutDir: outDir,
            client,
            sidecarClient,
          }),
        ),
      );
      allResults.push(...batchRes);
    }
  }

  const md = renderMarkdownReport(allResults, cli.model, cli.sidecarModel, cli.smoke);
  writeFileSync(join(outDir, "summary.json"), JSON.stringify(allResults, null, 2), "utf8");
  writeFileSync(join(outDir, "summary.md"), md, "utf8");

  console.log(`\n${md}`);
  console.log(`Saved full artifacts to ${outDir}/{summary.json,summary.md}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
