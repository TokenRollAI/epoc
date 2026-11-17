/**
 * Enforces an uncompacted hard context window (`PI_HARD_WINDOW_TOKENS`, e.g. 400,000)
 * for the `full` baseline arm in Pi CLI:
 *   1. Cancels Pi's built-in auto-compaction so `full` never compresses.
 *   2. Aborts cleanly as soon as prompt tokens exceed `PI_HARD_WINDOW_TOKENS`,
 *      reproducing a true 400K-window provider overflow without sending >400K requests.
 */

import { estimateTokens } from "../src/util/misc.ts";

export default function piHardWindowGuard(pi: {
  on: (event: string, handler: (event: any, ctx: any) => any) => void;
}) {
  const hardLimit = Number(process.env.PI_HARD_WINDOW_TOKENS) || 400_000;

  pi.on("session_before_compact", () => {
    return { cancel: true };
  });

  pi.on("context", (event: { messages: Array<{ content?: unknown }> }, ctx: { abort?: () => void }) => {
    let est = 1_800;
    for (const m of event.messages ?? []) {
      if (typeof m.content === "string") est += estimateTokens(m.content);
      else if (Array.isArray(m.content)) est += estimateTokens(JSON.stringify(m.content));
    }
    if (est > hardLimit) {
      ctx?.abort?.();
      throw new Error(`ContextWindowExceededError: estimated ${est} tokens exceeds ${hardLimit} hard window`);
    }
    return undefined;
  });

  pi.on("turn_end", (event: { message?: { usage?: { input?: number; cacheRead?: number; cacheWrite?: number } } }, ctx: { abort?: () => void }) => {
    const u = event.message?.usage;
    const promptTok = Number(u?.input ?? 0) + Number(u?.cacheRead ?? 0) + Number(u?.cacheWrite ?? 0);
    if (promptTok > hardLimit) {
      ctx?.abort?.();
      throw new Error(`ContextWindowExceededError: prompt ${promptTok} tokens exceeded ${hardLimit} hard window`);
    }
    return undefined;
  });
}
