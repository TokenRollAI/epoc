import type { HarnessMode, ToolDefinition } from "../types.ts";

export const contextQueryTool: ToolDefinition = {
  type: "function",
  function: {
    name: "context_query",
    description:
      "Search or slice the lossless session archive across ALL previously ingested tool outputs and messages " +
      "(including blocks compressed into [context:digest]/[context:skeleton] or archived inside completed scopes). " +
      "Use this whenever you need exact verbatim lines, buried codes/IDs, trace spans, formulas, or details not shown " +
      "in an inline digest — it returns targeted line snippets in one fast step without re-reading entire files.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description:
            "Exact identifier/code (e.g. 'INC-7Q42', 'TRC-0209'), regex pattern, keywords, or specific question.",
        },
        mode: {
          type: "string",
          enum: ["snippets", "grep", "raw_range", "ask"],
          description:
            "'snippets' (default): fast BM25 + exact-code ranked line windows. " +
            "'grep': case-insensitive regex match with surrounding lines. " +
            "'raw_range': exact line slice Lstart..Lend from block_ids[0]. " +
            "'ask': sidecar sub-reader synthesizes an exact cited answer from top archive snippets.",
        },
        block_ids: {
          type: "array",
          items: { type: "string" },
          description:
            "Optional block id filter (e.g. ['tool_004']). Omit to search across ALL session blocks.",
        },
        start_line: {
          type: "number",
          description: "1-based start line (used when mode='raw_range').",
        },
        end_line: {
          type: "number",
          description: "1-based end line (used when mode='raw_range').",
        },
      },
      required: ["query"],
    },
  },
};

export const scopeOpenTool: ToolDefinition = {
  type: "function",
  function: {
    name: "scope_open",
    description:
      "Open an isolated subtask scope before investigating a distinct hypothesis, subsystem, or batch of files. " +
      "When finished, call scope_submit so all intermediate exploration logs collapse cleanly into your concise result.",
    parameters: {
      type: "object",
      properties: {
        label: {
          type: "string",
          description: "Short identifier for the subtask (e.g. 'audit_ringbuffer', 'scan_batch_1').",
        },
        goal: {
          type: "string",
          description: "What this subtask scope aims to verify or extract.",
        },
      },
      required: ["label", "goal"],
    },
  },
};

export const scopeSubmitTool: ToolDefinition = {
  type: "function",
  function: {
    name: "scope_submit",
    description:
      "Complete the currently active subtask scope, collapsing its raw intermediate tool calls into a compact " +
      "result block while keeping all raw outputs searchable via context_query.",
    parameters: {
      type: "object",
      properties: {
        summary: {
          type: "string",
          description: "Concise conclusion of the subtask (include what was verified or ruled out).",
        },
        key_facts: {
          type: "array",
          items: { type: "string" },
          description: "Exact verbatim identifiers, numbers, formulas, or file:line facts to preserve inline.",
        },
      },
      required: ["summary"],
    },
  },
};

export function harnessToolDefinitions(mode: HarnessMode): ToolDefinition[] {
  if (mode === "transparent") return [];
  if (mode === "query_only") return [contextQueryTool];
  return [contextQueryTool, scopeOpenTool, scopeSubmitTool];
}

export function augmentSystemPromptForHarness(
  basePrompt: string,
  mode: HarnessMode,
): string {
  if (mode === "transparent") return basePrompt;

  const lines = [
    basePrompt.trimEnd(),
    "",
    "## Autonomous Context Harness",
    "- Your working context is automatically managed by a background Context Harness: older bulky tool outputs are converted into `[context:digest #id]` or `[context:skeleton #id]` cards while 100% of their raw lines remain indexed in the lossless session archive.",
    "- Whenever you need an exact buried code, trace span, line, or detail not shown in a digest card, call `context_query` (supports `snippets`, `grep`, `raw_range`, and `ask`) instead of guessing or re-reading entire large files.",
  ];
  if (mode === "cooperative") {
    lines.push(
      "- Optional subtask scoping: for isolated multi-step detours or subsystem checks, you may call `scope_open` at entry and `scope_submit` with exact `key_facts` upon completion to collapse noisy intermediate traces.",
    );
  }
  return lines.join("\n");
}
