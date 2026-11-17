import type { ChatUsage, CostBreakdown, ModelPricing } from "../types.ts";

/**
 * Published-tier reference pricing (USD per 1M tokens) with standard 90%
 * prefix-cache discount (`inputHitPerM = 0.1 * inputMissPerM`).
 *
 * Used consistently across ALL benchmark arms (`react`, `threshold`, `cpat`,
 * `harness`) so cost comparisons account fairly for:
 *   1. Prefix cache hit discounts vs cache-miss re-prefills
 *   2. Expensive output/reasoning token generation
 *   3. Every single cheap-model sidecar call
 */
export function pricingForModel(model: string): ModelPricing {
  const m = model.toLowerCase();
  if (m.includes("pro") || m.includes("reasoner") || m.includes("opus")) {
    return {
      inputMissPerM: 2.0,
      inputHitPerM: 0.2,
      outputPerM: 8.0,
    };
  }
  // Flash / Chat tier (e.g. deepseek-flash / DeepSeek-V4.1-Flash)
  return {
    inputMissPerM: 0.27,
    inputHitPerM: 0.027,
    outputPerM: 1.1,
  };
}

export function computeSingleCallUsd(usage: ChatUsage, pricing: ModelPricing): {
  hitUsd: number;
  missUsd: number;
  outUsd: number;
  totalUsd: number;
} {
  const hit = usage.prompt_cache_hit_tokens ?? 0;
  const miss =
    usage.prompt_cache_miss_tokens ?? Math.max(0, usage.prompt_tokens - hit);
  const out = usage.completion_tokens ?? 0;
  const hitUsd = (hit / 1_000_000) * pricing.inputHitPerM;
  const missUsd = (miss / 1_000_000) * pricing.inputMissPerM;
  const outUsd = (out / 1_000_000) * pricing.outputPerM;
  return { hitUsd, missUsd, outUsd, totalUsd: hitUsd + missUsd + outUsd };
}

export class CostTracker {
  private mainPrompt = 0;
  private mainHit = 0;
  private mainMiss = 0;
  private mainOut = 0;
  private mainHitUsd = 0;
  private mainMissUsd = 0;
  private mainOutUsd = 0;

  private sidecarCalls = 0;
  private sidecarPrompt = 0;
  private sidecarOut = 0;
  private sidecarUsd = 0;

  private criticalPathMs = 0;
  private cachedFractionEwma = 0.5;
  private mainCalls = 0;

  private readonly mainPricing: ModelPricing;
  private readonly sidecarPricing: ModelPricing;

  constructor(opts: { mainPricing: ModelPricing; sidecarPricing: ModelPricing }) {
    this.mainPricing = opts.mainPricing;
    this.sidecarPricing = opts.sidecarPricing;
  }

  recordMain(usage: ChatUsage): void {
    const hit = usage.prompt_cache_hit_tokens ?? 0;
    const miss =
      usage.prompt_cache_miss_tokens ?? Math.max(0, usage.prompt_tokens - hit);
    this.mainCalls += 1;
    this.mainPrompt += usage.prompt_tokens;
    this.mainHit += hit;
    this.mainMiss += miss;
    this.mainOut += usage.completion_tokens;

    const c = computeSingleCallUsd(usage, this.mainPricing);
    this.mainHitUsd += c.hitUsd;
    this.mainMissUsd += c.missUsd;
    this.mainOutUsd += c.outUsd;

    if (usage.prompt_tokens > 0) {
      const frac = Math.min(1, Math.max(0, hit / usage.prompt_tokens));
      this.cachedFractionEwma =
        this.mainCalls === 1 ? frac : 0.6 * frac + 0.4 * this.cachedFractionEwma;
    }
  }

  recordSidecar(usage: ChatUsage): void {
    this.sidecarCalls += 1;
    this.sidecarPrompt += usage.prompt_tokens;
    this.sidecarOut += usage.completion_tokens;
    const c = computeSingleCallUsd(usage, this.sidecarPricing);
    this.sidecarUsd += c.totalUsd;
  }

  addCriticalPathMs(ms: number): void {
    this.criticalPathMs += Math.max(0, ms);
  }

  cachedFraction(): number {
    return this.cachedFractionEwma;
  }

  /**
   * Cache-Aware Economic Commit Gate (synthesizing Self-GC + CPAT cef6582):
   *
   * Evaluates whether committing an in-prefix edit at cut-point `k` saves more
   * future prompt cost than the immediate one-time cache-invalidation penalty
   * on already-cached blocks sitting after `k`:
   *
   *   Benefit = H * freedTokens * effectiveKeepRatePerTok
   *   Penalty = suffixCachedTokens * (P_miss - P_hit) + commitSidecarUsd
   */
  evaluateCommit(opts: {
    freedTokens: number;
    suffixCachedTokens: number;
    expectedRemainingTurns: number;
    commitSidecarUsd?: number;
    force?: boolean;
  }): { shouldCommit: boolean; netBenefitUsd: number } {
    if (opts.freedTokens <= 0) {
      return { shouldCommit: Boolean(opts.force), netBenefitUsd: 0 };
    }
    const cf = this.cachedFractionEwma;
    // Even if cached, keeping tokens across H turns pays at least P_hit (plus
    // (1-cf)*P_miss whenever prefix shifts or cache expires).
    const effectiveRatePerM =
      cf * this.mainPricing.inputHitPerM + (1 - cf) * this.mainPricing.inputMissPerM;
    const futureSavingsUsd =
      (opts.expectedRemainingTurns * opts.freedTokens * effectiveRatePerM) / 1_000_000;
    const cacheBreakPenaltyUsd =
      (opts.suffixCachedTokens *
        Math.max(0, this.mainPricing.inputMissPerM - this.mainPricing.inputHitPerM)) /
      1_000_000;
    const net = futureSavingsUsd - cacheBreakPenaltyUsd - (opts.commitSidecarUsd ?? 0);
    return {
      shouldCommit: Boolean(opts.force) || net > 0,
      netBenefitUsd: Number(net.toFixed(6)),
    };
  }

  snapshot(): CostBreakdown {
    const effective =
      this.mainHitUsd + this.mainMissUsd + this.mainOutUsd + this.sidecarUsd;
    const uncachedMain =
      (this.mainPrompt / 1_000_000) * this.mainPricing.inputMissPerM +
      this.mainOutUsd;
    const r = (n: number) => Number(n.toFixed(6));
    return {
      main_prompt_tokens: this.mainPrompt,
      main_cache_hit_tokens: this.mainHit,
      main_cache_miss_tokens: this.mainMiss,
      main_completion_tokens: this.mainOut,
      sidecar_calls: this.sidecarCalls,
      sidecar_prompt_tokens: this.sidecarPrompt,
      sidecar_completion_tokens: this.sidecarOut,
      main_cache_hit_usd: r(this.mainHitUsd),
      main_cache_miss_usd: r(this.mainMissUsd),
      main_output_usd: r(this.mainOutUsd),
      sidecar_total_usd: r(this.sidecarUsd),
      effective_cost_usd: r(effective),
      uncached_main_equivalent_usd: r(uncachedMain),
      harness_critical_path_ms: Math.round(this.criticalPathMs),
    };
  }
}
