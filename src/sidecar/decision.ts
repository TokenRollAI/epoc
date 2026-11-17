import type { ChatClient } from "../deepseek/client.ts";
import type { CostTracker } from "../harness/cost.ts";
import type { BlockStore } from "../runtime/blocks.ts";
import type { ContentStore, Journal } from "../runtime/stores.ts";
import { applyContextUpdate } from "../runtime/patch.ts";
import type {
  ContextBlock,
  ContextOperation,
  ResolutionLevel,
} from "../types.ts";
import { ShadowDigester } from "./digester.ts";

const CORRECTION_CUE_RE =
  /(?:\b(?:ignore|disregard|actually|wait|instead|wrong|updated|override|correction|do\s+not\s+use|not\s+redis|deprecated)\b|忽略|不要|不对|更正|改成|作废|不是)/i;

const ISOLATED_POLLUTION_PROMPT = `You are a structurally isolated Specification & Context Pollution Auditor (AC3 protocol).
You ONLY see authoritative User messages and objective Tool observation summaries — all prior Assistant hypotheses have been physically excluded so wrong assumptions cannot infect you.

Determine whether a later user message or objective observation overrides, corrects, or invalidates an earlier premise or decoy clue.
Return ONLY valid JSON:
{
  "spec_updated": boolean,
  "authoritative_spec": "concise consolidated active user specification & exact constraints",
  "discard_terms": ["specific superseded module names, decoy error codes, or old numbers that must be purged from prior subjective history"],
  "verified_facts": ["objective facts from tool observations that remain valid"]
}`;

const TACTICAL_DECISION_PROMPT = `You are the Context Decision Controller for a bounded-window autonomous agent.
Inspect the active task goal, latest assistant step, and candidate older "full" blocks eligible for resolution degradation.

Rubric:
1. Phase suppression (SelfCompact rule): Set "phase": "mid_derivation" ONLY if the latest assistant turn is actively cross-comparing raw lines across these exact candidate blocks and has NOT extracted its conclusion yet. Otherwise set "phase": "exploring" or "subtask_complete".
2. For each candidate block id:
   - Put in "pin_ids" ONLY if its full verbatim body (not just its FactDigest) is still needed inline on the very next turn.
   - Put in "landmark_ids" if it was a completely irrelevant dead-end / empty check whose 1-line landmark suffices.
   - Put all remaining digested blocks in "digest_ids" (replaces multi-thousand-token raw text with its exact FactDigest + indexed query handle).

Return ONLY valid JSON:
{"phase":"exploring"|"mid_derivation"|"subtask_complete","pin_ids":string[],"digest_ids":string[],"landmark_ids":string[],"reason":string}`;

export interface DecisionTurnInput {
  estUsedTokens: number;
  maxContextTokens: number;
  workingTailTokens: number;
  softLimitRatio: number;
  mustActRatio: number;
  criticalRatio: number;
  minDigestTokens: number;
  keepRecentFullChains: number;
  expectedHorizonTurns: number;
  lastSentBlockIds: ReadonlySet<string>;
  questionBoundary: boolean;
  newUserMessageSinceLastTurn: boolean;
}

export class DecisionEngine {
  private readonly client?: ChatClient;
  private readonly model: string;
  private readonly blocks: BlockStore;
  private readonly content: ContentStore;
  private readonly journal: Journal;
  private readonly cost: CostTracker;
  private readonly digester: ShadowDigester;

  constructor(opts: {
    client?: ChatClient;
    model: string;
    blocks: BlockStore;
    content: ContentStore;
    journal: Journal;
    cost: CostTracker;
    digester: ShadowDigester;
  }) {
    this.client = opts.client;
    this.model = opts.model;
    this.blocks = opts.blocks;
    this.content = opts.content;
    this.journal = opts.journal;
    this.cost = opts.cost;
    this.digester = opts.digester;
  }

  async evaluateBeforeTurn(input: DecisionTurnInput): Promise<{
    committed: boolean;
    freedTokens: number;
    reason: string;
  }> {
    let totalFreed = 0;
    const reasons: string[] = [];

    // 1. Check for AC3 Context Pollution / Mid-Flight Spec Override
    if (input.newUserMessageSinceLastTurn && this.hasPollutionTrigger()) {
      const ac3 = await this.runIsolatedPollutionAudit();
      if (ac3.committed) {
        totalFreed += ac3.freedTokens;
        reasons.push(ac3.reason);
      }
    }

    // Recompute current budget pressure after any AC3 audit
    const currentUsed = Math.max(0, input.estUsedTokens - totalFreed);
    const softLimit = input.maxContextTokens * input.softLimitRatio;
    const mustActLimit = input.maxContextTokens * input.mustActRatio;
    const criticalLimit = input.maxContextTokens * input.criticalRatio;

    const rawTailTokens = this.blocks
      .visible()
      .filter(
        (b) =>
          (b.active_resolution ?? "full") === "full" &&
          b.epistemic !== "authoritative" &&
          !b.protected,
      )
      .reduce((s, b) => s + b.token_count, 0);

    // 2. Tier-1 Fast Gate: if within working-tail micro-batch & below soft limit
    //    and not at a question boundary, keep 100% prefix cache untouched!
    if (
      currentUsed < softLimit &&
      rawTailTokens <= input.workingTailTokens &&
      !input.questionBoundary
    ) {
      return {
        committed: totalFreed > 0,
        freedTokens: totalFreed,
        reason: reasons.join("; ") || "fast-gate noop (cache preserved)",
      };
    }

    // 3. Identify candidate "full" chain groups older than keepRecentFullChains.
    //    GC Hysteresis: accumulate until >= 2 blocks (or >= 45% of workingTailTokens)
    //    can drain together into the low watermark (`keepRecentFullChains = 1`),
    //    eliminating 1-block-per-turn thrashing and ensuring suffixCachedTokens == 0!
    const keepRecent = input.questionBoundary ? 0 : input.keepRecentFullChains;
    const eligible = this.selectDegradableBlocks(input.minDigestTokens, keepRecent);
    const eligibleTokens = eligible.reduce((s, b) => s + b.token_count, 0);
    const underHighPressure =
      currentUsed >= mustActLimit || input.questionBoundary;

    if (
      eligible.length > 0 &&
      (underHighPressure ||
        eligible.length >= 2 ||
        eligibleTokens >= input.workingTailTokens * 0.6)
    ) {
      // Optional Tier-2 Cheap Tactical Judge when the latest assistant step
      // emitted substantive prose reasoning (potential mid-derivation), skipping
      // unnecessary judge calls during pure sequential tool-pulling loops.
      let phase = input.questionBoundary ? "subtask_complete" : "exploring";
      const targetMap = new Map<string, ResolutionLevel>();
      for (const b of eligible) {
        targetMap.set(b.id, "digest");
      }

      const lastAsst = [...this.blocks.visible()]
        .reverse()
        .find((b) => b.kind === "assistant_message");
      const hasSubstantiveThought = (lastAsst?.token_count ?? 0) >= 40;

      if (
        this.client &&
        !input.questionBoundary &&
        currentUsed < mustActLimit &&
        eligible.length >= 2 &&
        hasSubstantiveThought
      ) {
        const judged = await this.runTacticalJudge(eligible);
        if (judged) {
          phase = judged.phase;
          if (phase === "mid_derivation" && currentUsed < mustActLimit) {
            this.journal.append("decision", {
              trigger: "working_tail",
              phase,
              committed: false,
              reason: `suppressed mid-derivation: ${judged.reason}`,
              net_benefit_est: 0,
              ops_count: 0,
            });
            return {
              committed: totalFreed > 0,
              freedTokens: totalFreed,
              reason: reasons.join("; ") || "suppressed (mid_derivation)",
            };
          }
          for (const pinId of judged.pinIds) targetMap.delete(pinId);
          for (const lmId of judged.landmarkIds) {
            if (targetMap.has(lmId)) targetMap.set(lmId, "landmark");
          }
        }
      }

      if (targetMap.size > 0) {
        // Estimate savings before running lazy batch digestion so EconomicCommitGate
        // prevents paying ANY sidecar digest cost when keeping warm cache is cheaper!
        const estFreedApprox = Array.from(targetMap.keys()).reduce((s, id) => {
          const b = this.blocks.get(id);
          return s + (b ? Math.max(0, b.token_count - 160) : 0);
        }, 0);
        const suffixCached = computeSuffixCachedTokens(
          this.blocks,
          Array.from(targetMap.keys()),
          input.lastSentBlockIds,
        );
        const econ = this.cost.evaluateCommit({
          freedTokens: estFreedApprox,
          suffixCachedTokens: suffixCached,
          expectedRemainingTurns: input.expectedHorizonTurns,
          force: underHighPressure,
        });

        if (econ.shouldCommit) {
          // Lazy Just-in-Time Batched Digestion: digest ONLY the blocks that
          // are actually degrading to "digest" right now, packed up to 4/call!
          const needDigestBlocks = eligible.filter(
            (b) => targetMap.get(b.id) === "digest" && !b.resolutions?.digest,
          );
          if (needDigestBlocks.length > 0) {
            const waitedMs = await this.digester.digestBatch(
              needDigestBlocks,
              this.latestUserTaskHint(),
              currentUsed >= criticalLimit ? 2600 : 4200,
            );
            if (waitedMs > 0) this.cost.addCriticalPathMs(waitedMs);
          }

          // Finalize resolution: use "digest" if ready, otherwise zero-wait "skeleton"
          for (const [id, res] of targetMap) {
            if (res === "digest") {
              const b = this.blocks.get(id);
              if (!b?.resolutions?.digest) targetMap.set(id, "skeleton");
            }
          }

          const ops: ContextOperation[] = buildResolutionOps(targetMap);
          const outcome = applyContextUpdate(this.blocks, this.content, ops, {
            allowReplace: false,
            allowRedact: false,
          });
          this.journal.append("patch", {
            actor: "harness",
            reason: `batch resolution degradation (${phase})`,
            operations: ops,
            result: {
              ...outcome,
              budget: {
                max: input.maxContextTokens,
                soft_limit: Math.floor(softLimit),
                used: Math.max(0, currentUsed - outcome.freed_tokens),
              },
            },
          });
          this.journal.append("decision", {
            trigger: input.questionBoundary ? "question_boundary" : "working_tail",
            phase,
            committed: outcome.ok,
            reason: `degraded ${targetMap.size} blocks`,
            net_benefit_est: econ.netBenefitUsd,
            ops_count: ops.length,
          });
          if (outcome.ok) {
            totalFreed += outcome.freed_tokens;
            reasons.push(`degraded ${targetMap.size} blocks (-${outcome.freed_tokens} tok)`);
          }
        } else {
          this.journal.append("decision", {
            trigger: "working_tail",
            phase,
            committed: false,
            reason: "deferred by EconomicCommitGate (warm cache cheaper)",
            net_benefit_est: econ.netBenefitUsd,
            ops_count: 0,
          });
        }
      }
    }

    // 4. Auto-Scope / Epoch Condensation (LSM Tier 2 -> Tier 3): fold completed
    //    waves of degraded chains at subtask boundaries or softLimit pressure.
    const afterDegradeUsed = Math.max(0, input.estUsedTokens - totalFreed);
    if (afterDegradeUsed >= softLimit * 0.85 || input.questionBoundary) {
      const epoch = this.tryCondenseOldestEpoch(
        input.maxContextTokens,
        softLimit,
        input.questionBoundary ? 5 : 6,
      );
      if (epoch.freedTokens > 0) {
        totalFreed += epoch.freedTokens;
        reasons.push(`auto-folded scope epoch (-${epoch.freedTokens} tok)`);
      }
    }

    // 5. Hard Critical Safety Net: if STILL >= criticalLimit (e.g. a single turn
    //    parallel-read 6+ huge files), batch-digest with tight cap & zero-wait
    //    skeleton fallback so the main model never crashes on a hard window.
    const afterEpochUsed = Math.max(0, input.estUsedTokens - totalFreed);
    if (afterEpochUsed >= criticalLimit) {
      const newestBulky = this.selectDegradableBlocks(input.minDigestTokens, 0);
      if (newestBulky.length > 0) {
        const waited = await this.digester.digestBatch(
          newestBulky,
          this.latestUserTaskHint(),
          2600,
        );
        if (waited > 0) this.cost.addCriticalPathMs(waited);
        const emergencyMap = new Map<string, ResolutionLevel>();
        for (const b of newestBulky) {
          emergencyMap.set(b.id, b.resolutions?.digest ? "digest" : "skeleton");
        }
        const ops = buildResolutionOps(emergencyMap);
        const outcome = applyContextUpdate(this.blocks, this.content, ops, {
          allowReplace: false,
          allowRedact: false,
        });
        if (outcome.ok && outcome.freed_tokens > 0) {
          totalFreed += outcome.freed_tokens;
          reasons.push(`critical safety degradation (-${outcome.freed_tokens} tok)`);
        }
      }
    }

    return {
      committed: totalFreed > 0,
      freedTokens: totalFreed,
      reason: reasons.join("; ") || "noop",
    };
  }

  private selectDegradableBlocks(
    minTokens: number,
    keepRecentBulkyChains: number,
  ): ContextBlock[] {
    const groups = this.blocks.chainGroups(this.blocks.visible());
    const bulkyGroups = groups.filter((g) =>
      g.some(
        (b) =>
          b.kind === "tool_result" &&
          (b.active_resolution ?? "full") === "full" &&
          b.epistemic !== "authoritative" &&
          !b.protected &&
          !b.pinned &&
          b.token_count >= minTokens,
      ),
    );

    const cutoff = Math.max(0, bulkyGroups.length - keepRecentBulkyChains);
    const selected: ContextBlock[] = [];
    for (let i = 0; i < cutoff; i++) {
      for (const b of bulkyGroups[i]) {
        if (
          b.kind === "tool_result" &&
          (b.active_resolution ?? "full") === "full" &&
          b.epistemic !== "authoritative" &&
          !b.protected &&
          !b.pinned &&
          b.token_count >= minTokens
        ) {
          selected.push(b);
        }
      }
    }
    return selected;
  }

  private hasPollutionTrigger(): boolean {
    const users = this.blocks
      .all()
      .filter((b) => b.kind === "user_message" && typeof b.content === "string");
    if (users.length < 2) return false;
    const latestUser = users[users.length - 1].content as string;
    return CORRECTION_CUE_RE.test(latestUser);
  }

  /**
   * AC3 Structurally Isolated Pollution Audit:
   * Stage 1 inspects ONLY user_message blocks and objective tool observation
   * cards — physically excluding all assistant_message / reasoning_trace text!
   */
  private async runIsolatedPollutionAudit(): Promise<{
    committed: boolean;
    freedTokens: number;
    reason: string;
  }> {
    if (!this.client) {
      return { committed: false, freedTokens: 0, reason: "" };
    }

    const allBlocks = this.blocks.all();
    const userBlocks = allBlocks.filter((b) => b.kind === "user_message");
    if (userBlocks.length < 2) {
      return { committed: false, freedTokens: 0, reason: "" };
    }
    const latestUserId = userBlocks[userBlocks.length - 1].id;
    const latestUserIdx = this.blocks.order.indexOf(latestUserId);

    // Stage 1: Build physically isolated view (ZERO assistant text!)
    const userTranscript = userBlocks
      .map((u, idx) => `[User Message #${idx + 1} (${u.id})]:\n${String(u.content)}`)
      .join("\n\n");

    const obsCards = allBlocks
      .filter((b) => b.kind === "tool_result")
      .slice(-12)
      .map((b) => {
        const d = b.resolutions?.digest;
        const brief = d
          ? `${d.summary} | facts: ${d.verbatim_facts.slice(0, 6).join(", ")}`
          : b.description;
        return `- [${b.id}] ${brief}`;
      })
      .join("\n");

    const t0 = Date.now();
    try {
      const resp = await this.client.chat({
        model: this.model,
        messages: [
          { role: "system", content: ISOLATED_POLLUTION_PROMPT },
          {
            role: "user",
            content: `=== AUTHORITATIVE USER MESSAGES ===\n${userTranscript}\n\n=== OBJECTIVE TOOL OBSERVATIONS ===\n${obsCards || "(none)"}`,
          },
        ],
        temperature: 0,
        maxTokens: 1400,
        reasoningEffort: "low",
        responseFormat: { type: "json_object" },
      });
      const ms = Date.now() - t0;
      this.cost.recordSidecar(resp.usage);
      this.cost.addCriticalPathMs(ms);
      this.journal.append("sidecar_call", {
        purpose: "pollution_audit",
        model: this.model,
        usage: resp.usage,
        duration_ms: ms,
      });

      const parsed = safeJson(resp.message.content ?? "");
      if (!parsed || parsed.spec_updated !== true) {
        return { committed: false, freedTokens: 0, reason: "" };
      }

      const cleanSpec =
        typeof parsed.authoritative_spec === "string"
          ? parsed.authoritative_spec.trim()
          : "";
      const discardTerms = Array.isArray(parsed.discard_terms)
        ? parsed.discard_terms
            .map((x) => String(x).trim().toLowerCase())
            .filter((s) => s.length >= 3)
        : [];
      const verifiedFacts = Array.isArray(parsed.verified_facts)
        ? parsed.verified_facts.map((x) => String(x).trim()).filter(Boolean)
        : [];

      // Stage 2: Locate prior closed chains before latestUserIdx that are
      // tainted by `discardTerms` (subjective assumptions or decoy reads).
      const taintedRawIds: string[] = [];
      for (let i = 0; i < latestUserIdx; i++) {
        const b = this.blocks.get(this.blocks.order[i]);
        if (!b || b.visibility !== "model") continue;
        if (
          b.epistemic === "authoritative" ||
          b.kind === "system_prompt" ||
          b.kind === "user_message" ||
          b.kind === "task_state" ||
          b.protected
        ) {
          continue;
        }
        const hay = `${b.description}\n${typeof b.content === "string" ? b.content : b.content.summary}`.toLowerCase();
        if (discardTerms.some((term) => hay.includes(term))) {
          taintedRawIds.push(b.id);
        }
      }

      const closedTaintedIds = this.blocks
        .expandToChainClosure(taintedRawIds)
        .filter((id) => {
          const idx = this.blocks.order.indexOf(id);
          const b = this.blocks.get(id);
          return (
            idx >= 0 &&
            idx < latestUserIdx &&
            b !== undefined &&
            b.visibility === "model" &&
            b.epistemic !== "authoritative" &&
            !b.protected
          );
        });

      if (closedTaintedIds.length === 0 && !cleanSpec) {
        return { committed: false, freedTokens: 0, reason: "" };
      }

      const checkpointLines = [
        `[AUTHORITATIVE SPECIFICATION CHECKPOINT — AC3 Verified]`,
        `Active Specification: ${cleanSpec}`,
        ...(discardTerms.length > 0
          ? [`Superseded / Invalidated Premises (DO NOT USE): ${discardTerms.join(", ")}`]
          : []),
        ...(verifiedFacts.length > 0
          ? [`Retained Objective Facts: ${verifiedFacts.join(" | ")}`]
          : []),
      ];

      if (closedTaintedIds.length > 0) {
        const op: ContextOperation = {
          op: "purge_pollution",
          ids: closedTaintedIds,
          clean_checkpoint: {
            description: "AC3 verified specification checkpoint",
            content: checkpointLines.join("\n"),
          },
        };
        const outcome = applyContextUpdate(this.blocks, this.content, [op], {
          allowReplace: false,
          allowRedact: false,
        });
        if (outcome.ok) {
          this.journal.append("patch", {
            actor: "harness",
            reason: "AC3 isolated pollution purge",
            operations: [op],
            result: {
              ...outcome,
              budget: { max: 0, soft_limit: 0, used: 0 },
            },
          });
          return {
            committed: true,
            freedTokens: outcome.freed_tokens,
            reason: `AC3 purged ${closedTaintedIds.length} polluted blocks (-${outcome.freed_tokens} tok)`,
          };
        }
      }
    } catch {
      // Never block main agent if sidecar audit errors out.
    }
    return { committed: false, freedTokens: 0, reason: "" };
  }

  private async runTacticalJudge(eligible: ContextBlock[]): Promise<{
    phase: string;
    pinIds: string[];
    landmarkIds: string[];
    reason: string;
  } | undefined> {
    if (!this.client) return undefined;
    const visible = this.blocks.visible();
    const userSpec = visible
      .filter((b) => b.kind === "user_message")
      .slice(-2)
      .map((b) => String(b.content).slice(0, 400))
      .join("\n---\n");
    const lastAsst = [...visible]
      .reverse()
      .find((b) => b.kind === "assistant_message");

    const cards = eligible
      .map((b) => {
        const d = b.resolutions?.digest;
        return (
          `- id=${b.id} (${b.kind}, ${b.token_count} tok): ${b.description}` +
          (d ? `\n  digest: ${d.summary} | facts=${d.verbatim_facts.slice(0, 5).join("; ")}` : "")
        );
      })
      .join("\n");

    const t0 = Date.now();
    try {
      const resp = await this.client.chat({
        model: this.model,
        messages: [
          { role: "system", content: TACTICAL_DECISION_PROMPT },
          {
            role: "user",
            content:
              `Active user goal:\n${userSpec}\n\n` +
              `Latest assistant step:\n${lastAsst ? lastAsst.description : "(none)"}\n\n` +
              `Candidate "full" blocks:\n${cards}`,
          },
        ],
        temperature: 0,
        maxTokens: 1000,
        reasoningEffort: "low",
        responseFormat: { type: "json_object" },
      });
      const ms = Date.now() - t0;
      this.cost.recordSidecar(resp.usage);
      this.cost.addCriticalPathMs(ms);
      this.journal.append("sidecar_call", {
        purpose: "decision",
        model: this.model,
        usage: resp.usage,
        duration_ms: ms,
        target_ids: eligible.map((b) => b.id),
      });

      const parsed = safeJson(resp.message.content ?? "");
      if (!parsed) return undefined;
      const toIds = (v: unknown) =>
        Array.isArray(v) ? v.map((x) => String(x).trim()).filter(Boolean) : [];
      return {
        phase: typeof parsed.phase === "string" ? parsed.phase : "exploring",
        pinIds: toIds(parsed.pin_ids),
        landmarkIds: toIds(parsed.landmark_ids),
        reason: typeof parsed.reason === "string" ? parsed.reason : "",
      };
    } catch {
      return undefined;
    }
  }

  private latestUserTaskHint(): string {
    const users = this.blocks
      .all()
      .filter((b) => b.kind === "user_message" && typeof b.content === "string");
    if (users.length === 0) return "";
    const first = String(users[0].content).slice(0, 240);
    const last =
      users.length > 1 ? String(users[users.length - 1].content).slice(0, 240) : "";
    return last ? `${first}\nCurrent question: ${last}` : first;
  }

  /**
   * Condense oldest contiguous degraded chains into a single Epoch Summary
   * preserving all verbatim_facts & source block references.
   */
  private tryCondenseOldestEpoch(
    maxContextTokens: number,
    softLimit: number,
    minRunGroups = 6,
  ): { freedTokens: number } {
    const visible = this.blocks.visible();
    const groups = this.blocks.chainGroups(visible);
    // Find contiguous run of >= minRunGroups non-authoritative, already-degraded groups
    const runGroups: ContextBlock[][] = [];
    for (const g of groups) {
      const canFold = g.every(
        (b) =>
          b.epistemic !== "authoritative" &&
          b.kind !== "system_prompt" &&
          b.kind !== "user_message" &&
          b.kind !== "task_state" &&
          b.kind !== "budget_report" &&
          !b.protected &&
          (b.active_resolution ?? "full") !== "full",
      );
      if (canFold) {
        runGroups.push(g);
        if (runGroups.length >= 8) break;
      } else if (runGroups.length >= minRunGroups) {
        break;
      } else {
        runGroups.length = 0;
      }
    }
    if (runGroups.length < minRunGroups) return { freedTokens: 0 };

    const flatBlocks = runGroups.flat();
    const ids = flatBlocks.map((b) => b.id);
    // Verify contiguity in store.order for `fold` or use `compact` with preserve list
    const facts: string[] = [];
    const summaries: string[] = [];
    for (const b of flatBlocks) {
      const d = b.resolutions?.digest;
      if (d) {
        summaries.push(`[#${b.id}] ${d.summary}`);
        facts.push(...d.verbatim_facts);
      } else if (b.kind === "tool_result") {
        summaries.push(`[#${b.id}] ${b.description}`);
      }
    }
    const dedupFacts = Array.from(new Set(facts));
    const epochText = [
      `[Epoch Archive Summary — condensed ${ids.length} blocks (${ids[0]}..${ids[ids.length - 1]}); all raw lines remain queryable via context_query]`,
      ...summaries.slice(0, 16),
      ...(dedupFacts.length > 0
        ? [`Consolidated Verbatim Facts: ${dedupFacts.join(" | ")}`]
        : []),
    ].join("\n");

    const op: ContextOperation = {
      op: "compact",
      ids,
      output: {
        description: `Condensed epoch (${ids[0]}..${ids[ids.length - 1]})`,
        content: epochText,
      },
      preserve: dedupFacts.length > 0 ? dedupFacts.slice(0, 12) : ["source block references"],
      drop: ["individual per-call wrapper boilerplate"],
    };

    const outcome = applyContextUpdate(this.blocks, this.content, [op], {
      allowReplace: false,
      allowRedact: false,
    });
    if (outcome.ok) {
      this.journal.append("patch", {
        actor: "harness",
        reason: "LSM epoch condensation",
        operations: [op],
        result: {
          ...outcome,
          budget: {
            max: maxContextTokens,
            soft_limit: Math.floor(softLimit),
            used: 0,
          },
        },
      });
      return { freedTokens: outcome.freed_tokens };
    }
    return { freedTokens: 0 };
  }
}

function buildResolutionOps(
  targetMap: Map<string, ResolutionLevel>,
): ContextOperation[] {
  const byRes = new Map<ResolutionLevel, string[]>();
  for (const [id, res] of targetMap) {
    const arr = byRes.get(res) ?? [];
    arr.push(id);
    byRes.set(res, arr);
  }
  const ops: ContextOperation[] = [];
  for (const [resolution, ids] of byRes) {
    ops.push({ op: "set_resolution", ids, resolution });
  }
  return ops;
}

function estimateResolutionSavings(
  store: BlockStore,
  targetMap: Map<string, ResolutionLevel>,
): number {
  let saved = 0;
  for (const [id, res] of targetMap) {
    const b = store.get(id);
    if (!b) continue;
    const prev = b.active_resolution;
    b.active_resolution = res;
    const nextText = store.renderedText(b);
    b.active_resolution = prev;
    const nextTok = Math.ceil(nextText.length * 0.32);
    saved += Math.max(0, b.token_count - nextTok);
  }
  return saved;
}

function computeSuffixCachedTokens(
  store: BlockStore,
  modifiedIds: string[],
  lastSentBlockIds: ReadonlySet<string>,
): number {
  if (modifiedIds.length === 0 || lastSentBlockIds.size === 0) return 0;
  const visible = store.visible();
  const modSet = new Set(modifiedIds);
  const firstIdx = visible.findIndex((b) => modSet.has(b.id));
  if (firstIdx === -1) return 0;

  let suffixCached = 0;
  for (let i = firstIdx + 1; i < visible.length; i++) {
    const b = visible[i];
    if (!lastSentBlockIds.has(b.id)) continue;
    if (modSet.has(b.id)) {
      // Modified block shrinks to ~140 tokens; only that tiny stub needs prefill
      suffixCached += Math.min(b.token_count, 140);
    } else {
      suffixCached += b.token_count;
    }
  }
  return suffixCached;
}

function safeJson(raw: string): Record<string, unknown> | undefined {
  const cleaned = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  if (!cleaned) return undefined;
  try {
    const v = JSON.parse(cleaned);
    return typeof v === "object" && v !== null ? (v as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}
