import type {
  ChatMessage,
  ChatResponse,
  ChatUsage,
  ToolDefinition,
} from "../types.ts";
import type { DeepSeekEnv } from "../util/env.ts";

export interface ChatOptions {
  model: string;
  messages: ChatMessage[];
  tools?: ToolDefinition[];
  toolChoice?: "auto" | "required" | "none";
  temperature?: number;
  maxTokens?: number;
  reasoningEffort?: "low" | "high" | "max";
  responseFormat?: { type: "json_object" };
}

export interface ChatClient {
  chat(opts: ChatOptions): Promise<ChatResponse>;
}

/**
 * Minimal OpenAI-compatible client for DeepSeek chat completions.
 * Zero dependencies — uses global fetch.
 */
export class DeepSeekClient implements ChatClient {
  private readonly url: string;
  private readonly apiKey: string;
  private strictRejected = false;
  private effortRejected = false;

  constructor(env: DeepSeekEnv) {
    this.url = `${env.baseUrl}/chat/completions`;
    this.apiKey = env.apiKey;
  }

  async chat(opts: ChatOptions): Promise<ChatResponse> {
    let tools = opts.tools;
    let attempt = 0;
    // One retry path: if the endpoint rejects strict tool schemas or
    // reasoning_effort, strip the unsupported knob and retry.
    for (;;) {
      attempt += 1;
      if (this.strictRejected && tools) tools = stripStrict(tools);
      const useEffort = !this.effortRejected && opts.reasoningEffort;
      const res = await fetch(this.url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: opts.model,
          messages: opts.messages,
          ...(tools && tools.length > 0 ? { tools } : {}),
          ...(opts.toolChoice ? { tool_choice: opts.toolChoice } : {}),
          ...(opts.temperature !== undefined
            ? { temperature: opts.temperature }
            : {}),
          ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
          ...(useEffort ? { reasoning_effort: opts.reasoningEffort } : {}),
          ...(opts.responseFormat ? { response_format: opts.responseFormat } : {}),
        }),
      });

      if (!res.ok) {
        const body = await res.text();
        const strictIssue =
          res.status === 400 && /strict/i.test(body) && !this.strictRejected;
        if (strictIssue && tools) {
          this.strictRejected = true;
          continue;
        }
        const effortIssue =
          res.status === 400 &&
          /reasoning_effort|effort/i.test(body) &&
          !this.effortRejected &&
          Boolean(opts.reasoningEffort);
        if (effortIssue) {
          this.effortRejected = true;
          continue;
        }
        if (res.status >= 500 && attempt < 3) {
          await sleep(1000 * attempt);
          continue;
        }
        throw new Error(`DeepSeek API ${res.status}: ${body.slice(0, 2000)}`);
      }

      const data = (await res.json()) as {
        choices: Array<{
          message: ChatMessage;
          finish_reason: string;
        }>;
        usage?: ChatUsage & {
          prompt_tokens_details?: { cached_tokens?: number };
          completion_tokens_details?: { reasoning_tokens?: number };
        };
      };
      const choice = data.choices?.[0];
      if (!choice) throw new Error("DeepSeek API returned no choices");
      const rawUsage = data.usage;
      const promptTok = rawUsage?.prompt_tokens ?? 0;
      const hitTok =
        rawUsage?.prompt_cache_hit_tokens ??
        rawUsage?.prompt_tokens_details?.cached_tokens ??
        0;
      const missTok =
        rawUsage?.prompt_cache_miss_tokens ?? Math.max(0, promptTok - hitTok);
      return {
        message: choice.message,
        finish_reason: choice.finish_reason,
        usage: {
          prompt_tokens: promptTok,
          completion_tokens: rawUsage?.completion_tokens ?? 0,
          total_tokens: rawUsage?.total_tokens ?? promptTok + (rawUsage?.completion_tokens ?? 0),
          prompt_cache_hit_tokens: hitTok,
          prompt_cache_miss_tokens: missTok,
          ...(rawUsage?.completion_tokens_details?.reasoning_tokens !== undefined
            ? { reasoning_tokens: rawUsage.completion_tokens_details.reasoning_tokens }
            : {}),
        },
      };
    }
  }
}

function stripStrict(tools: ToolDefinition[]): ToolDefinition[] {
  return tools.map((t) => ({
    ...t,
    function: { ...t.function, strict: undefined },
  }));
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
