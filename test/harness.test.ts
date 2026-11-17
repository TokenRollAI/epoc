import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { ChatClient, ChatOptions } from "../src/deepseek/client.ts";
import { CostTracker } from "../src/harness/cost.ts";
import { ContextHarness } from "../src/harness/harness.ts";
import { resetIdCounter } from "../src/util/misc.ts";
import type { ChatResponse, ToolCall } from "../src/types.ts";

function tmpRunDir(): string {
  return mkdtempSync(join(tmpdir(), "cpat-harness-test-"));
}

function makeToolCall(id: string, name: string, args: Record<string, unknown>): ToolCall {
  return {
    id,
    type: "function",
    function: { name, arguments: JSON.stringify(args) },
  };
}

test("harness view compiler keeps strict byte-identical prefix across normal bulky reads", async () => {
  resetIdCounter();
  const harness = new ContextHarness({
    maxContextTokens: 64_000,
    workingTailTokens: 40_000, // keep inline so we isolate render order stability
    runDir: tmpRunDir(),
  });

  harness.ingestSystem("Base system.");
  harness.ingestUser("Investigate docs.");

  const tc1 = makeToolCall("c1", "read_file", { path: "a.txt" });
  harness.ingestAssistant({ role: "assistant", content: "", tool_calls: [tc1] });
  harness.ingestToolResult(tc1, "DOC_A_LINE\n".repeat(400));
  const turn1 = await harness.compileTurn();
  harness.recordMainCall("deepseek-flash", { prompt_tokens: 1800, completion_tokens: 40, total_tokens: 1840 }, turn1.view);

  // Add a second bulky read; unlike legacy generational newest-first sorting,
  // turn1's ENTIRE message array must remain an exact prefix of turn2's messages!
  const tc2 = makeToolCall("c2", "read_file", { path: "b.txt" });
  harness.ingestAssistant({ role: "assistant", content: "", tool_calls: [tc2] });
  harness.ingestToolResult(tc2, "DOC_B_LINE\n".repeat(400));
  const turn2 = await harness.compileTurn();

  assert.equal(turn2.messages.length, turn1.messages.length + 2);
  for (let i = 0; i < turn1.messages.length; i++) {
    assert.deepEqual(turn2.messages[i], turn1.messages[i]);
  }
});

test("ScopeTree pops completed child scope back to byte-identical parent prefix while keeping archive searchable", async () => {
  resetIdCounter();
  const harness = new ContextHarness({
    maxContextTokens: 32_000,
    runDir: tmpRunDir(),
  });

  harness.ingestSystem("System.");
  harness.ingestUser("Find root cause.");
  const beforeScope = await harness.compileTurn();

  // Open scope
  const openTc = makeToolCall("c_open", "scope_open", {
    label: "check_redis",
    goal: "Check staging redis logs",
  });
  harness.ingestAssistant({
    role: "assistant",
    content: "Opening scope.",
    reasoning_content: "Let us open a scope first.",
    tool_calls: [openTc],
  });
  const openOut = await harness.tryHandleToolCall(openTc);
  harness.ingestToolResult(openTc, openOut!);

  // Bulky read inside scope with a buried code
  const readTc = makeToolCall("c_read", "read_file", { path: "redis.log" });
  harness.ingestAssistant({ role: "assistant", content: "", tool_calls: [readTc] });
  harness.ingestToolResult(
    readTc,
    "NOISE_LINE\n".repeat(300) + "\nBURIED_TRACE=TRC-9981 host=redis-stage-2\n",
  );

  // Submit scope
  const subTc = makeToolCall("c_sub", "scope_submit", {
    summary: "Redis timeout is staging-only decoy.",
    key_facts: ["redis-stage-2 is staging only"],
  });
  harness.ingestAssistant({
    role: "assistant",
    content: "Submitting scope.",
    reasoning_content: "Closing scope now.",
    tool_calls: [subTc],
  });
  const subOut = await harness.tryHandleToolCall(subTc);
  harness.ingestToolResult(subTc, subOut!);

  const afterScope = await harness.compileTurn();

  // 1. Parent prefix before scope_open is 100% byte-identical
  for (let i = 0; i < beforeScope.messages.length; i++) {
    assert.deepEqual(afterScope.messages[i], beforeScope.messages[i]);
  }
  // 2. Entire child trajectory collapsed into ONE summary message
  assert.equal(afterScope.messages.length, beforeScope.messages.length + 1);
  assert.match(afterScope.messages[afterScope.messages.length - 1].content, /completed scope/);
  assert.match(afterScope.messages[afterScope.messages.length - 1].content, /staging-only decoy/);

  // 3. No orphan tool messages anywhere
  for (let i = 0; i < afterScope.messages.length; i++) {
    assert.notEqual(afterScope.messages[i].role, "tool");
  }

  // 4. Buried trace inside the collapsed scope is STILL 1-hop recoverable via context_query!
  const queryTc = makeToolCall("c_q", "context_query", {
    query: "TRC-9981",
    mode: "snippets",
  });
  const queryOut = await harness.tryHandleToolCall(queryTc);
  assert.ok(queryOut);
  assert.match(queryOut!, /TRC-9981/);
  assert.match(queryOut!, /redis-stage-2/);
});

test("ShadowDigester + DecisionEngine batch-degrade older blocks to FactDigest while keeping newest chain full", async () => {
  resetIdCounter();
  const sidecarCalls: ChatOptions[] = [];
  const mockSidecar: ChatClient = {
    async chat(opts: ChatOptions): Promise<ChatResponse> {
      sidecarCalls.push(opts);
      const sys = opts.messages[0]?.content ?? "";
      if (sys.includes("batch context-digestion sidecar")) {
        const userText = opts.messages[1]?.content ?? "";
        const digests: Record<string, unknown> = {};
        for (const m of userText.matchAll(/=== BLOCK (tool_\d+)[\s\S]*?(INC-\d+[A-Z])/g)) {
          digests[m[1]] = {
            summary: `Batched dossier digest for ${m[1]}.`,
            verbatim_facts: [m[2], "accuracy 88.4%"],
            ruled_out: [],
            query_hint: "Raw lines in archive.",
          };
        }
        return {
          message: { role: "assistant", content: JSON.stringify({ digests }) },
          finish_reason: "stop",
          usage: { prompt_tokens: 900, completion_tokens: 110, total_tokens: 1010 },
        };
      }
      if (sys.includes("context-digestion sidecar")) {
        const userText = opts.messages[1]?.content ?? "";
        const codeMatch = userText.match(/INC-\d+[A-Z]/);
        return {
          message: {
            role: "assistant",
            content: JSON.stringify({
              summary: "Dossier digested.",
              verbatim_facts: codeMatch ? [codeMatch[0], "accuracy 88.4%"] : ["accuracy 88.4%"],
              ruled_out: [],
              query_hint: "Raw lines in archive.",
            }),
          },
          finish_reason: "stop",
          usage: { prompt_tokens: 600, completion_tokens: 70, total_tokens: 670 },
        };
      }
      // Tactical judge response
      return {
        message: {
          role: "assistant",
          content: JSON.stringify({
            phase: "exploring",
            pin_ids: [],
            digest_ids: ["tool_004", "tool_006"],
            landmark_ids: [],
            reason: "digested earlier reads",
          }),
        },
        finish_reason: "stop",
        usage: { prompt_tokens: 300, completion_tokens: 50, total_tokens: 350 },
      };
    },
  };

  const harness = new ContextHarness(
    {
      maxContextTokens: 12_000,
      workingTailTokens: 2_500,
      minDigestTokens: 200,
      keepRecentFullChains: 1,
      runDir: tmpRunDir(),
    },
    { sidecarClient: mockSidecar },
  );

  harness.ingestSystem("System.");
  harness.ingestUser("Read three dossiers.");

  for (let i = 1; i <= 3; i++) {
    const tc = makeToolCall(`call_${i}`, "read_file", { path: `doc_${i}.txt` });
    harness.ingestAssistant({
      role: "assistant",
      content:
        "Analyzing intermediate dossier evidence across sections to compare candidate metrics carefully before proceeding to the next step of verification.",
      tool_calls: [tc],
    });
    harness.ingestToolResult(
      tc,
      `DOSSIER ${i}\n` + "PADDING PROSE WORDS ".repeat(220) + `\nINC-100${i}Z\n`,
    );
  }

  const compiled = await harness.compileTurn();
  const toolMsgs = compiled.messages.filter((m) => m.role === "tool");
  assert.equal(toolMsgs.length, 3);

  // Older 2 tool results degraded to [context:digest] preserving INC codes; newest stays full!
  assert.match(toolMsgs[0].content, /\[context:digest/);
  assert.match(toolMsgs[0].content, /INC-1001Z/);
  assert.match(toolMsgs[1].content, /\[context:digest/);
  assert.match(toolMsgs[1].content, /INC-1002Z/);
  assert.doesNotMatch(toolMsgs[2].content, /\[context:digest/);
  assert.match(toolMsgs[2].content, /PADDING PROSE WORDS/);

  // Lazy batch digestion: 0 eager calls on tool_008 (kept full), 1 tactical judge
  // + 1 single batched call covering BOTH tool_004 & tool_006 = only 2 calls total!
  assert.equal(sidecarCalls.length, 2);
  assert.ok(harness.costBreakdown().sidecar_total_usd > 0);
});

test("AC3 isolated pollution audit excludes assistant text during Stage 1 and purges tainted assumptions", async () => {
  resetIdCounter();
  const stage1Prompts: string[] = [];
  const mockSidecar: ChatClient = {
    async chat(opts: ChatOptions): Promise<ChatResponse> {
      const sys = opts.messages[0]?.content ?? "";
      const usr = opts.messages[1]?.content ?? "";
      if (sys.includes("AC3 protocol")) {
        stage1Prompts.push(usr);
        return {
          message: {
            role: "assistant",
            content: JSON.stringify({
              spec_updated: true,
              authoritative_spec: "Use LocalRingBuffer with 150ms SLA.",
              discard_terms: ["redis_cluster_hyp", "500ms_old"],
              verified_facts: ["LocalRingBuffer active in prod"],
            }),
          },
          finish_reason: "stop",
          usage: { prompt_tokens: 250, completion_tokens: 60, total_tokens: 310 },
        };
      }
      return {
        message: {
          role: "assistant",
          content: JSON.stringify({
            summary: "ok",
            verbatim_facts: [],
            ruled_out: [],
            query_hint: "",
          }),
        },
        finish_reason: "stop",
        usage: { prompt_tokens: 100, completion_tokens: 30, total_tokens: 130 },
      };
    },
  };

  const harness = new ContextHarness(
    { maxContextTokens: 20_000, runDir: tmpRunDir() },
    { sidecarClient: mockSidecar },
  );

  harness.ingestSystem("System.");
  harness.ingestUser("Debug latency under 500ms_old.");
  // Polluted assistant turn anchoring on wrong hypothesis
  harness.ingestAssistant({
    role: "assistant",
    content: "SECRET_POLLUTED_THOUGHT: The bug is definitely in redis_cluster_hyp!",
  });

  // User mid-flight override
  harness.ingestUser(
    "Wait — ignore staging! Do not use redis; production uses LocalRingBuffer and updated SLA is 150ms.",
  );

  const compiled = await harness.compileTurn();

  // 1. Stage 1 prompt MUST NOT contain the assistant's polluted thought ("Pollution is contagious" law!)
  assert.equal(stage1Prompts.length, 1);
  assert.doesNotMatch(stage1Prompts[0], /SECRET_POLLUTED_THOUGHT/);

  // 2. Tainted assistant block was purged from active view and replaced by clean checkpoint
  const allRendered = compiled.messages.map((m) => m.content).join("\n");
  assert.doesNotMatch(allRendered, /SECRET_POLLUTED_THOUGHT/);
  assert.match(allRendered, /AUTHORITATIVE SPECIFICATION CHECKPOINT/);
  assert.match(allRendered, /LocalRingBuffer/);
});

test("Sidecar failure degrades gracefully to deterministic Skeleton without failing main turn", async () => {
  resetIdCounter();
  const brokenSidecar: ChatClient = {
    async chat(): Promise<ChatResponse> {
      throw new Error("503 sidecar unavailable");
    },
  };

  const harness = new ContextHarness(
    {
      maxContextTokens: 8_000,
      workingTailTokens: 1_500,
      minDigestTokens: 200,
      keepRecentFullChains: 0,
      runDir: tmpRunDir(),
    },
    { sidecarClient: brokenSidecar },
  );

  harness.ingestSystem("System.");
  harness.ingestUser("Scan.");
  const tc = makeToolCall("c1", "read_file", { path: "x.txt" });
  harness.ingestAssistant({ role: "assistant", content: "", tool_calls: [tc] });
  harness.ingestToolResult(
    tc,
    "=== Section 1 ===\n" + "filler text ".repeat(700) + "\nIncident: INC-4242K accuracy 91.2%\n",
  );

  const compiled = await harness.compileTurn();
  const toolMsg = compiled.messages.find((m) => m.role === "tool")!;
  assert.match(toolMsg.content, /\[context:skeleton/);
  assert.match(toolMsg.content, /INC-4242K/);
  assert.match(toolMsg.content, /91\.2%/);
});

test("EconomicCommitGate blocks high-penalty mid-prefix edits when cache is warm and pressure is low", () => {
  const tracker = new CostTracker({
    mainPricing: { inputMissPerM: 2.0, inputHitPerM: 0.2, outputPerM: 8.0 },
    sidecarPricing: { inputMissPerM: 0.27, inputHitPerM: 0.027, outputPerM: 1.1 },
  });
  // Warm cache: 95% hit rate
  tracker.recordMain({
    prompt_tokens: 40_000,
    prompt_cache_hit_tokens: 38_000,
    prompt_cache_miss_tokens: 2_000,
    completion_tokens: 100,
    total_tokens: 40_100,
  });

  // Freeing 600 tokens while invalidating 30,000 cached suffix tokens loses money!
  const bad = tracker.evaluateCommit({
    freedTokens: 600,
    suffixCachedTokens: 30_000,
    expectedRemainingTurns: 4,
    force: false,
  });
  assert.equal(bad.shouldCommit, false);
  assert.ok(bad.netBenefitUsd < 0);

  // Tail-aligned batch freeing 12,000 tokens with 0 cached suffix invalidated wins!
  const good = tracker.evaluateCommit({
    freedTokens: 12_000,
    suffixCachedTokens: 0,
    expectedRemainingTurns: 6,
    force: false,
  });
  assert.equal(good.shouldCommit, true);
  assert.ok(good.netBenefitUsd > 0);
});
