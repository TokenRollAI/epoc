import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  createCpatPiExtension,
  type PiBeforeAgentStartEvent,
  type PiBeforeProviderRequestEvent,
  type PiContextEvent,
  type PiExtensionAPI,
  type PiExtensionContext,
  type PiSessionBeforeCompactEvent,
  type PiToolResultEvent,
  type PiTurnEndEvent,
} from "../extensions/pi-cpat.ts";
import { extractCodeSymbolSpans } from "../src/sidecar/skeleton.ts";

function buildMockPiHost() {
  const handlers = new Map<string, Array<(event: any, ctx: PiExtensionContext) => any>>();
  const tools = new Map<string, any>();
  const commands = new Map<string, any>();

  const api: PiExtensionAPI = {
    on(event: string, handler: (event: any, ctx: PiExtensionContext) => any) {
      const arr = handlers.get(event) ?? [];
      arr.push(handler);
      handlers.set(event, arr);
    },
    registerTool(tool) {
      tools.set(tool.name, tool);
    },
    registerCommand(name, opts) {
      commands.set(name, opts);
    },
  };

  return { api, handlers, tools, commands };
}

test("extractCodeSymbolSpans computes accurate source-file Lstart-Lend with baseLineOffset", () => {
  const pyCode = [
    "class SlicedLowLevelWCS:",
    "    def __init__(self, wcs, slices):",
    "        self._wcs = wcs",
    "        self._slices = slices",
    "",
    "    def pixel_to_world_values(self, *pixel_arrays):",
    "        world = self._wcs.pixel_to_world_values(*pixel_arrays)",
    "        return world",
    "",
    "def helper_transform(x):",
    "    return x * 2",
  ];

  const spans = extractCodeSymbolSpans(pyCode, 120);
  assert.equal(spans.length, 4);
  assert.deepEqual(
    spans.map((s) => `${s.kind} ${s.name} L${s.startLine}-L${s.endLine}`),
    [
      "class SlicedLowLevelWCS L120-L127",
      "def __init__ L121-L123",
      "def pixel_to_world_values L125-L127",
      "def helper_transform L129-L130",
    ],
  );
});

test("pi-cpat extension: GC hysteresis preserves prefix below watermark and emits context_edit + lossless context_query above watermark", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-test-"));
  const metricsPath = path.join(tmpDir, "metrics.json");
  const host = buildMockPiHost();

  const ext = createCpatPiExtension({
    maxContextTokens: 14_000,
    strategy: "sliding",
    sidecarClient: null, // deterministic skeleton mode offline
    metricsFile: metricsPath,
    storeRootDir: tmpDir,
  });
  ext(host.api);

  assert.ok(host.tools.has("context_query"), "context_query tool must be registered");
  assert.ok(host.commands.has("cpat"), "/cpat command must be registered");

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 14_000 },
    sessionManager: {
      getSessionId: () => "sess-1",
    },
  };

  // Verify minimal system hint injection
  const startEvent: PiBeforeAgentStartEvent = {
    type: "before_agent_start",
    systemPrompt: "Base prompt",
    systemPromptOptions: {},
    prompt: "Locate pixel_to_world_values bug and secret token NEEDLE-9841",
  };
  for (const h of host.handlers.get("before_agent_start") ?? []) await h(startEvent, ctx);
  assert.ok(String(startEvent.systemPromptOptions.appendSystemPrompt).includes("[CPAT Context Harness]"));

  const bigCodeA = Array.from({ length: 180 }, (_, i) =>
    i === 42
      ? `def target_solver_${i}(alpha):\n    return "NEEDLE-9841" # exact marker`
      : `    x_${i} = x + ${i} * 123456789`,
  ).join("\n");

  const bigCodeB = Array.from({ length: 180 }, (_, i) =>
    `def other_module_fn_${i}(y): return y - ${i} * 987654321`,
  ).join("\n");

  // Turn 1: single moderate read — still under High-Watermark when turnIndex===0
  const turn1: PiTurnEndEvent = {
    type: "turn_end",
    turnIndex: 0,
    outcome: "completed",
    messageEntryId: "m1",
    toolResultEntryIds: ["tr1"],
    entries: [],
    continue: false,
    message: {
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "call_1",
          name: "read",
          arguments: { path: "pkg/solver.py", offset: 100, limit: 180 },
        },
      ],
    },
    toolResults: [
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "read",
        content: [{ type: "text", text: bigCodeA }],
      },
    ],
    context: {
      contextMessages: [
        { role: "user", content: "Find bug" },
        { role: "toolResult", content: [{ type: "text", text: bigCodeA }] },
      ],
    },
  };

  const res1 = await host.handlers.get("turn_end")![0](turn1, ctx);
  assert.equal(res1, undefined, "Turn 1 stays untouched to preserve Prefix Cache");

  // Turn 2: second bulky read pushes cumulative uncompacted tokens past High-Watermark -> drains Turn 1 via context_edit!
  const turn2: PiTurnEndEvent = {
    type: "turn_end",
    turnIndex: 1,
    outcome: "completed",
    messageEntryId: "m2",
    toolResultEntryIds: ["tr2"],
    entries: [],
    continue: false,
    message: {
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "call_2",
          name: "read",
          arguments: { path: "pkg/other.py", offset: 1, limit: 180 },
        },
      ],
    },
    toolResults: [
      {
        role: "toolResult",
        toolCallId: "call_2",
        toolName: "read",
        content: [{ type: "text", text: bigCodeB }],
      },
    ],
    context: {
      contextMessages: [
        { role: "user", content: "Find bug" },
        {
          role: "toolResult",
          content: [{ type: "text", text: bigCodeA + "\n" + bigCodeB + "\n" + bigCodeB + "\n" + bigCodeB }],
        },
      ],
    },
  };

  const res2 = await host.handlers.get("turn_end")![0](turn2, ctx);
  assert.ok(res2 && Array.isArray(res2.entries), "Turn 2 should emit context_edit entries");
  const edit = res2.entries.find((e: any) => e.type === "context_edit" && e.targetId === "tr1");
  assert.ok(edit, "tr1 should be compacted via native context_edit");
  assert.ok(
    edit.replacement.content[0].text.includes("L142-L280: def target_solver_42"),
    `Skeleton must preserve true file offset L142 (= offset 100 + idx 42), got: ${edit.replacement.content[0].text}`,
  );

  // Extract block id from skeleton header `[context:skeleton #b_...`
  const bidMatch = edit.replacement.content[0].text.match(/#(b_\d+)/);
  assert.ok(bidMatch, "Skeleton header must include block id");
  const bid = bidMatch![1];

  // Verify lossless recovery via `context_query` with both keyword and true file line slice!
  const qTool = host.tools.get("context_query")!;
  const qHit = await qTool.execute("q1", { query: "NEEDLE-9841" }, undefined, undefined, ctx);
  assert.ok(qHit.content[0].text.includes("NEEDLE-9841"));
  assert.ok(qHit.content[0].text.includes("L143:"));

  const sliceHit = await qTool.execute(
    "q2",
    { block_ids: [bid], start_line: 142, end_line: 144 },
    undefined,
    undefined,
    ctx,
  );
  assert.ok(sliceHit.content[0].text.includes("L142: def target_solver_42"));
});

test("pi-cpat extension: pre-call context shield prevents overflow on massive seeded history while keeping verbatim needles searchable", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-seed-"));
  const host = buildMockPiHost();
  const ext = createCpatPiExtension({
    maxContextTokens: 8_000,
    strategy: "sliding",
    sidecarClient: null,
    storeRootDir: tmpDir,
  });
  ext(host.api);

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 8_000 },
    sessionManager: { getSessionId: () => "sess-seed" },
  };

  // Seed 20 historical session messages (~18,000 tokens > 8,000 window) with 1 buried needle
  const messages = Array.from({ length: 20 }, (_, i) => ({
    role: "user",
    content:
      i === 7
        ? `Session #8 transcript: User moved to apartment 42B-ZIRCON on 2026-04-19.\n` +
          "Discussion notes about daily commute and neighborhood parks. ".repeat(65)
        : `Session #${i + 1} transcript:\n` +
          `Routine conversation about cooking, books, and weekend hikes in session ${i + 1}. `.repeat(65),
  }));
  messages.push({
    role: "user",
    content: "Which exact apartment code did I move to?",
  });

  const ctxEvent: PiContextEvent = { type: "context", messages };
  const projected = await host.handlers.get("context")![0](ctxEvent, ctx);
  assert.ok(projected?.messages, "Pre-call shield should project oversized history");

  const compactTotalChars = projected.messages.reduce(
    (s: number, m: any) => s + (typeof m.content === "string" ? m.content.length : JSON.stringify(m.content).length),
    0,
  );
  const rawTotalChars = messages.reduce((s, m) => s + m.content.length, 0);
  assert.ok(compactTotalChars < rawTotalChars * 0.45, "Projected context should shrink >55%");

  // Needle remains 100% retrievable via context_query!
  const qRes = await host.tools.get("context_query")!.execute(
    "q_seed",
    { query: "apartment 42B-ZIRCON" },
    undefined,
    undefined,
    ctx,
  );
  assert.ok(qRes.content[0].text.includes("42B-ZIRCON"));

  // Premature threshold compaction is cancelled when tokensBefore < 82% window
  const cmpCancel = await host.handlers.get("session_before_compact")![0](
    {
      type: "session_before_compact",
      reason: "threshold",
      willRetry: false,
      signal: new AbortController().signal,
      preparation: {
        messagesToSummarize: messages.slice(0, 5),
        turnPrefixMessages: [],
        tokensBefore: 4_500,
        firstKeptEntryId: "e5",
      },
    } satisfies PiSessionBeforeCompactEvent,
    ctx,
  );
  assert.deepEqual(cmpCancel, { cancel: true });
});

test("pi-cpat extension: default cache-first mode emits ZERO mid-stream edits and replaces threshold compaction with $0 lossless checkpoint", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-cachefirst-"));
  const host = buildMockPiHost();
  const ext = createCpatPiExtension({
    maxContextTokens: 400_000,
    storeRootDir: tmpDir,
  });
  ext(host.api);

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 400_000 },
    sessionManager: { getSessionId: () => "sess-cf-400k" },
  };

  const batch1 = "--- ARC 1 (UserProfile-1) | SESSION 4 (Date: 2026-02-11) ---\nUser graduated with PhD in Glaciology (code GLAC-7712).\n" + " Conversation filler.".repeat(300);
  const batch2 = "--- ARC 2 (UserProfile-2) | SESSION 9 (Date: 2026-05-19) ---\nUser adopted rescue greyhound named Zephyr.\n" + " Conversation filler.".repeat(300);

  for (let t = 0; t < 2; t++) {
    const res = await host.handlers.get("turn_end")![0](
      {
        type: "turn_end",
        turnIndex: t,
        outcome: "completed",
        messageEntryId: `m_${t}`,
        toolResultEntryIds: [`tr_${t}`],
        entries: [],
        continue: false,
        message: {
          role: "assistant",
          content: [{ type: "toolCall", id: `c_${t}`, name: "read_next_batch", arguments: { count: 2 } }],
        },
        toolResults: [
          {
            role: "toolResult",
            toolCallId: `c_${t}`,
            toolName: "read_next_batch",
            content: [{ type: "text", text: t === 0 ? batch1 : batch2 }],
          },
        ],
        context: { contextMessages: [] },
      } satisfies PiTurnEndEvent,
      ctx,
    );
    assert.equal(res, undefined, "cache-first must NEVER emit mid-stream context_edit");
  }

  // When 384K ceiling finally triggers session_before_compact, cache-first returns lossless structural directory at $0 LLM cost
  const cmp = await host.handlers.get("session_before_compact")![0](
    {
      type: "session_before_compact",
      reason: "threshold",
      willRetry: false,
      signal: new AbortController().signal,
      preparation: {
        messagesToSummarize: [{ role: "toolResult", content: [{ type: "text", text: batch1 }] }],
        turnPrefixMessages: [],
        tokensBefore: 385_000,
        firstKeptEntryId: "tr_1",
      },
    } satisfies PiSessionBeforeCompactEvent,
    ctx,
  );

  assert.ok(cmp?.compaction?.summary.includes("CPAT Cache-First Lossless Checkpoint"));
  assert.ok(cmp.compaction.summary.includes("ARC 1 (UserProfile-1)"));
  assert.equal(cmp.compaction.usage, undefined, "$0 deterministic checkpoint makes zero sidecar LLM calls");

  // Archived verbatim needle remains 100% retrievable via context_query!
  const q = await host.tools.get("context_query")!.execute("q_cf", { query: "Glaciology GLAC-7712" }, undefined, undefined, ctx);
  assert.ok(q.content[0].text.includes("GLAC-7712"));
});

test("pi-cpat extension: Pre-Cache tool_result gate recovers 200KB host-truncated bash & 3200-line read outputs before KV Cache while passing dense narrative streams untouched", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-precache-"));
  const host = buildMockPiHost();
  const ext = createCpatPiExtension({
    maxContextTokens: 400_000,
    storeRootDir: tmpDir,
  });
  ext(host.api);

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 400_000 },
    sessionManager: { getSessionId: () => "sess-precache" },
  };

  // 1. Simulate 4,000-line bash test log where early line 45 has AssertionError,
  //    while Pi's built-in 50KB tail truncation only kept lines 3,000..4,000!
  const fullBashLines = Array.from({ length: 4000 }, (_, i) =>
    i === 44
      ? "AssertionError: ROOT_CAUSE_TOKEN_8821 mismatch in gateway_checksum(frame)"
      : `[2026-11-17T04:00:00Z] INFO worker-${i % 8} processed routine heartbeat seq=${100000 + i} latency_ms=12`,
  );
  const fullLogFile = path.join(tmpDir, "pi-bash-full.log");
  fs.writeFileSync(fullLogFile, fullBashLines.join("\n"), "utf8");
  const truncatedTailOnly = fullBashLines.slice(3000).join("\n");

  const bashRes = await host.handlers.get("tool_result")![0](
    {
      toolName: "bash",
      toolCallId: "tc_bash_1",
      input: { command: "pytest -vv" },
      content: [{ type: "text", text: truncatedTailOnly }],
      details: { fullOutputPath: fullLogFile },
    } satisfies PiToolResultEvent,
    ctx,
  );

  assert.ok(bashRes?.content?.[0]?.text, "Oversized/truncated bash output must be projected pre-cache");
  const projectedBash = bashRes.content[0].text as string;
  assert.ok(
    projectedBash.includes("ROOT_CAUSE_TOKEN_8821"),
    "Pre-cache projection must surface early AssertionError dropped by Pi's 50KB tail truncation!",
  );
  assert.ok(projectedBash.length < truncatedTailOnly.length * 0.35, "Projected text should save >65% prefill bytes");

  // 2. Simulate 3,200-line Python file truncated by Pi's `read` at line 2,000
  const bigPyLines = Array.from({ length: 3200 }, (_, i) => {
    if (i === 2949) return "def deep_tail_solver(matrix, eps):";
    if (i === 2950) return '    """Solves tail eigen-system OMEGA-314."""';
    if (i === 2951) return "    return matrix * eps";
    return `x_${i} = ${i} + 1`;
  });
  const pyPath = path.join(tmpDir, "huge_module.py");
  fs.writeFileSync(pyPath, bigPyLines.join("\n"), "utf8");

  const readRes = await host.handlers.get("tool_result")![0](
    {
      toolName: "read",
      toolCallId: "tc_read_1",
      input: { path: "huge_module.py" },
      content: [{ type: "text", text: bigPyLines.slice(0, 2000).join("\n") }],
      details: { truncation: { truncated: true, totalLines: 3200, outputLines: 2000 } },
    } satisfies PiToolResultEvent,
    ctx,
  );

  assert.ok(readRes?.content?.[0]?.text.includes("L2950-L2952: def deep_tail_solver"));
  const bidMatch = (readRes.content[0].text as string).match(/#(b_\d+)/);
  assert.ok(bidMatch);

  // Slice lines 2950..2952 via context_query!
  const sliceOut = await host.tools.get("context_query")!.execute(
    "q_py",
    { block_ids: [bidMatch![1]], start_line: 2950, end_line: 2952 },
    undefined,
    undefined,
    ctx,
  );
  assert.ok(sliceOut.content[0].text.includes("OMEGA-314"));

  // 3. Untruncated 32K-token noisy bash compile log (>24K coding cap) is ALSO projected pre-cache!
  const giantCompileLines = Array.from({ length: 2400 }, (_, i) =>
    i === 612
      ? "ERROR: src/router/handshake.rs:188: undefined symbol `ROUTING_EPOCH_V4`"
      : `[build ${i}/2400] Compiling crate_module_${i % 12} --opt-level=3 --target=aarch64-apple-darwin`,
  ).join("\n");
  const giantCompileRes = await host.handlers.get("tool_result")![0](
    {
      toolName: "bash",
      toolCallId: "tc_bash_untrunc_32k",
      input: { command: "cargo build --release" },
      content: [{ type: "text", text: giantCompileLines }],
    } satisfies PiToolResultEvent,
    ctx,
  );
  assert.ok(
    giantCompileRes?.content?.[0]?.text?.includes("ROUTING_EPOCH_V4"),
    "32K untruncated bash compile dump (>24K cap) must be projected pre-cache with diagnostic error preserved",
  );

  // 4. Dense non-coding reading stream (<48K tokens, low redundancy) passes untouched into KV Cache!
  const uniqueSentences = Array.from(
    { length: 400 },
    (_, i) => `Paragraph ${String.fromCharCode(65 + (i % 26))}-${(i * 37) % 997}: Distinct historical biography entry discussing scholar_${i} and treaty_${i * 13}.`,
  ).join("\n");
  const streamRes = await host.handlers.get("tool_result")![0](
    {
      toolName: "read_next_batch",
      toolCallId: "tc_stream_1",
      input: { count: 2 },
      content: [{ type: "text", text: uniqueSentences }],
    } satisfies PiToolResultEvent,
    ctx,
  );
  assert.equal(streamRes, undefined, "Dense narrative stream below 48K must enter KV Cache untouched");
});

test("pi-cpat extension: Cache-Forked Epoch Summary reuses exact warm wire prefix at 99% KV Cache hit and chains multi-epoch state", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-fork-"));
  const metricsFile = path.join(tmpDir, "fork-metrics.json");
  const host = buildMockPiHost();

  let capturedUrl = "";
  let capturedBody: any = null;

  const mockForkFetch = (async (url: string | URL | Request, init?: RequestInit) => {
    capturedUrl = String(url);
    capturedBody = JSON.parse(String(init?.body ?? "{}"));
    return new Response(
      JSON.stringify({
        choices: [
          {
            message: {
              role: "assistant",
              content:
                "1. **Goal & Current Execution State**: Ingested batches 1-14; continuing stream until END_OF_STREAM.\n2. **Causal Decisions & Verified Facts**: Confirmed migration key `EPOCH-KEY-7741` in `src/auth/v2.ts` L40-L88.\n3. **Falsified Hypotheses & Guardrails**: Legacy v1 header format rejected.",
            },
          },
        ],
        usage: {
          prompt_tokens: 384_000,
          prompt_cache_hit_tokens: 380_000,
          prompt_cache_miss_tokens: 4_000,
          completion_tokens: 120,
          total_tokens: 384_120,
        },
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }) as typeof fetch;

  const ext = createCpatPiExtension({
    maxContextTokens: 400_000,
    storeRootDir: tmpDir,
    metricsFile,
    cacheForkFetch: mockForkFetch,
    cacheForkApiKey: "test-key",
    cacheForkBaseUrl: "https://api.deepseek.com/v1",
  });
  ext(host.api);

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 400_000 },
    sessionManager: { getSessionId: () => "sess-fork-384k" },
  };

  const warmTools = [{ type: "function", function: { name: "read_next_batch", parameters: {} } }];
  const warmMessages = [
    { role: "system", content: "System prompt" },
    { role: "user", content: "Ingest all batches and answer." },
    { role: "assistant", content: null, tool_calls: [{ id: "tc1", type: "function", function: { name: "read_next_batch", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "tc1", content: "--- BATCH 1 ---\nMigration token EPOCH-KEY-7741." + " text".repeat(120) },
  ];

  // Simulate Pi firing before_provider_request on the warm 380K turn
  await host.handlers.get("before_provider_request")![0](
    {
      type: "before_provider_request",
      payload: {
        model: "deepseek-flash",
        tools: warmTools,
        messages: warmMessages,
      },
    } satisfies PiBeforeProviderRequestEvent,
    ctx,
  );

  // Now simulate hitting 384K threshold compaction
  const cmp = await host.handlers.get("session_before_compact")![0](
    {
      type: "session_before_compact",
      reason: "threshold",
      willRetry: false,
      signal: new AbortController().signal,
      preparation: {
        previousSummary: "Epoch #0 note: repository root initialized.",
        messagesToSummarize: [{ role: "toolResult", content: [{ type: "text", text: "Batch 1..14 content ".repeat(30) }] }],
        turnPrefixMessages: [],
        tokensBefore: 384_500,
        firstKeptEntryId: "e_keep_15",
      },
    } satisfies PiSessionBeforeCompactEvent,
    ctx,
  );

  assert.equal(capturedUrl, "https://api.deepseek.com/v1/chat/completions");
  assert.deepEqual(capturedBody.tools, warmTools, "Cache-fork request must preserve exact tools declaration");
  assert.deepEqual(
    capturedBody.messages.slice(0, warmMessages.length),
    warmMessages,
    "Cache-fork request must preserve 100% byte-identical warm message prefix",
  );
  assert.equal(capturedBody.messages.length, warmMessages.length + 1);
  assert.equal(capturedBody.messages[warmMessages.length].role, "user");

  assert.ok(cmp?.compaction?.summary.includes("Epoch Macro-Narrative & Causal Decision Ledger"));
  assert.ok(cmp.compaction.summary.includes("EPOCH-KEY-7741"));
  assert.equal(cmp.compaction.usage.input, 4_000, "PiUsage.input must record ONLY uncached tokens");
  assert.equal(cmp.compaction.usage.cacheRead, 380_000, "PiUsage.cacheRead must record 380K cached tokens");
  assert.ok(
    cmp.compaction.usage.cost.total < 0.0025,
    `384K Cache-Forked Summary should cost ~$0.0018 (got $${cmp.compaction.usage.cost.total})`,
  );

  const savedMetrics = JSON.parse(fs.readFileSync(metricsFile, "utf8"));
  assert.equal(savedMetrics.cacheForkSummaries, 1);
  assert.equal(savedMetrics.cacheForkCachedTokens, 380_000);
});

test("pi-cpat extension: Explicit Task-Boundary & Pollution-Safe ScopeTree folding purges tainted reasoning with zero suffix shift while keeping raw facts searchable", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "cpat-pi-fold-"));
  const metricsFile = path.join(tmpDir, "fold-metrics.json");
  const host = buildMockPiHost();

  const ext = createCpatPiExtension({
    maxContextTokens: 400_000,
    boundaryFoldMinTokens: 500,
    storeRootDir: tmpDir,
    metricsFile,
  });
  ext(host.api);

  assert.ok(host.tools.has("context_checkpoint"), "context_checkpoint tool must be registered");
  assert.ok(host.commands.has("cpat-fold"), "/cpat-fold command must be registered");

  const pollutedRawTool =
    "legacy_v1_config.yaml:\n  auth_header: X-Legacy-Dead-Token\n  production_v2_override_port: 9443\n" +
    "  # deprecated v1 commentary line\n".repeat(120);
  const pollutedAssistantMsg = {
    role: "assistant",
    content: [
      {
        type: "text",
        text: "WRONG HYPOTHESIS: We must use X-Legacy-Dead-Token everywhere! ".repeat(60),
      },
      {
        type: "toolCall",
        id: "call_polluted_1",
        name: "read",
        arguments: { path: "legacy_v1_config.yaml" },
      },
    ],
  };
  const pollutedToolResultMsg = {
    role: "toolResult",
    toolCallId: "call_polluted_1",
    toolName: "read",
    content: [{ type: "text", text: pollutedRawTool }],
  };

  const phase1Entries = [
    {
      sourceEntry: { id: "u_1", type: "message" },
      messages: [{ role: "user", content: "Check auth configuration" }],
    },
    {
      sourceEntry: { id: "a_polluted_1", type: "message" },
      messages: [pollutedAssistantMsg],
    },
    {
      sourceEntry: { id: "tr_polluted_1", type: "message" },
      messages: [pollutedToolResultMsg],
    },
  ];

  const ctx: PiExtensionContext = {
    cwd: tmpDir,
    model: { id: "deepseek-flash", provider: "deepseek", contextWindow: 400_000 },
    sessionManager: {
      getSessionId: () => "sess-pollution-fold",
      buildSessionProjection: () => ({
        entries: phase1Entries,
        messages: phase1Entries.flatMap((e) => e.messages),
      }),
    },
  };

  // Path A: User explicitly rejects wrong hypothesis in Phase 2 prompt (sub-384K!)
  const startRes = await host.handlers.get("before_agent_start")![0](
    {
      type: "before_agent_start",
      systemPrompt: "Base",
      systemPromptOptions: {},
      prompt: "Ignore the previous hypothesis — legacy_v1 X-Legacy-Dead-Token is deprecated! Configure the v2 override port instead.",
    } satisfies PiBeforeAgentStartEvent,
    ctx,
  );

  assert.ok(startRes?.message, "Explicit user pollution reset must emit AC3 scope_checkpoint message");
  const ckptText = startRes.message.content[0].text as string;
  assert.ok(ckptText.includes("[cpat:scope_checkpoint #"));
  assert.ok(
    !ckptText.includes("WRONG HYPOTHESIS"),
    "AC3 Authoritative-Only Reset must NEVER leak polluted assistant text into checkpoint!",
  );

  // Verify Turn 1 `context` immediately strips polluted Phase 1 assistant + toolResult messages
  const turn1Ctx = await host.handlers.get("context")![0](
    {
      type: "context",
      messages: [
        ...phase1Entries.flatMap((e) => e.messages),
        { role: "user", content: "Ignore the previous hypothesis..." },
        { role: "custom", content: [{ type: "text", text: ckptText }] },
      ],
    } satisfies PiContextEvent,
    ctx,
  );
  assert.ok(turn1Ctx?.messages, "Turn 1 context must strip Phase 1 polluted messages");
  assert.equal(turn1Ctx.messages.length, 3, "Only [u_1, u_2, scope_checkpoint] should remain");

  // Verify Turn 1 `turn_end` permanently commits `context_edit(targetId, null)` for Phase 1's polluted pair
  const turn1End = await host.handlers.get("turn_end")![0](
    {
      type: "turn_end",
      turnIndex: 1,
      outcome: "completed",
      messageEntryId: "a_clean_2",
      toolResultEntryIds: [],
      entries: [],
      continue: false,
      message: { role: "assistant", content: [{ type: "text", text: "Checking lossless index for v2 port." }] },
      toolResults: [],
      context: { contextEntries: phase1Entries, contextMessages: turn1Ctx.messages },
    } satisfies PiTurnEndEvent,
    ctx,
  );
  assert.ok(turn1End?.entries, "Turn 1 turn_end must commit persistent context_edit omissions");
  const omittedIds = turn1End.entries
    .filter((e: any) => e.type === "context_edit" && e.replacement === null)
    .map((e: any) => e.targetId);
  assert.deepEqual(omittedIds, ["a_polluted_1", "tr_polluted_1"]);

  // Verify ground-truth fact (`production_v2_override_port: 9443`) from folded toolResult is STILL 100% searchable!
  const qPort = await host.tools.get("context_query")!.execute(
    "q_port",
    { query: "production_v2_override_port" },
    undefined,
    undefined,
    ctx,
  );
  assert.ok(qPort.content[0].text.includes("9443"));

  // Path B: Agent-invoked `context_checkpoint` tool folds intermediate subtask turns at `turn_end`
  const ckptToolOut = await host.tools.get("context_checkpoint")!.execute(
    "tc_ckpt_1",
    {
      mode: "subtask_complete",
      label: "port-verification",
      summary: "Verified production_v2_override_port is 9443.",
      verified_facts: ["legacy_v1_config.yaml: production_v2_override_port = 9443"],
    },
    undefined,
    undefined,
    ctx,
  );
  assert.ok(ckptToolOut.content[0].text.includes("9443"));

  const phase2Entries = [
    ...phase1Entries,
    { sourceEntry: { id: "a_clean_2", type: "message" }, messages: [{ role: "assistant", content: "step" }] },
    { sourceEntry: { id: "a_explore_3", type: "message" }, messages: [{ role: "assistant", content: "intermediate exploration ".repeat(40) }] },
    { sourceEntry: { id: "tr_explore_3", type: "message" }, messages: [{ role: "toolResult", content: "intermediate output ".repeat(40) }] },
    { sourceEntry: { id: "a_ckpt_4", type: "message" }, messages: [{ role: "assistant", content: "checkpoint call" }] },
    { sourceEntry: { id: "tr_ckpt_4", type: "message" }, messages: [{ role: "toolResult", content: ckptToolOut.content }] },
  ];

  const turnCkptEnd = await host.handlers.get("turn_end")![0](
    {
      type: "turn_end",
      turnIndex: 3,
      outcome: "completed",
      messageEntryId: "a_ckpt_4",
      toolResultEntryIds: ["tr_ckpt_4"],
      entries: [],
      continue: false,
      message: {
        role: "assistant",
        content: [{ type: "toolCall", id: "tc_ckpt_1", name: "context_checkpoint", arguments: {} }],
      },
      toolResults: [{ role: "toolResult", toolCallId: "tc_ckpt_1", toolName: "context_checkpoint", content: ckptToolOut.content }],
      context: { contextEntries: phase2Entries, contextMessages: [] },
    } satisfies PiTurnEndEvent,
    ctx,
  );

  const agentFoldEdits = (turnCkptEnd?.entries ?? [])
    .filter((e: any) => e.type === "context_edit" && e.replacement === null)
    .map((e: any) => e.targetId);
  assert.deepEqual(
    agentFoldEdits,
    ["a_explore_3", "tr_explore_3"],
    "Agent context_checkpoint must fold ONLY the active scope span strictly between lastFoldAnchorEntryId and current turn!",
  );
});
