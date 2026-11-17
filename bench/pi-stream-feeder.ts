/**
 * Minimal one-pass stream feeder extension for Pi CLI benchmark runs
 * (`LongMemEval-S` & `MemoryAgentBench`).
 *
 * Reads chunks from `PI_STREAM_CHUNKS_FILE` (JSON `{ chunks: string[] }`) and
 * exposes `read_next_batch` so all 4 Pi arms (`full`, `pi-native`,
 * `billion-context`, `cpat`) experience authentic multi-turn context growth
 * inside Pi CLI without being able to re-grep raw files off disk.
 */

import fs from "node:fs";

interface StreamPayload {
  title?: string;
  chunks: string[];
}

export default function piStreamFeeder(pi: {
  registerTool: (tool: {
    name: string;
    label?: string;
    description: string;
    parameters: Record<string, unknown>;
    execute: (
      id: string,
      params: Record<string, unknown>,
    ) => Promise<{ content: Array<{ type: "text"; text: string }>; details?: unknown }>;
  }) => void;
}) {
  const filePath = process.env.PI_STREAM_CHUNKS_FILE;
  if (!filePath || !fs.existsSync(filePath)) return;

  const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as StreamPayload;
  const chunks = payload.chunks ?? [];
  let cursor = 0;

  pi.registerTool({
    name: "read_next_batch",
    label: "Read Next Stream Batch",
    description: `Ingest the next sequential batch(es) from the one-pass stream (${chunks.length} total batches). Once read, batches cannot be re-pulled from this stream.`,
    parameters: {
      type: "object",
      properties: {
        count: {
          type: "number",
          description: "How many sequential batches to ingest in this call (default 2, max 3).",
        },
      },
      additionalProperties: false,
    },
    async execute(_id, params) {
      if (cursor >= chunks.length) {
        return {
          content: [
            {
              type: "text",
              text: `END_OF_STREAM: All ${chunks.length}/${chunks.length} batches have already been ingested. Proceed to answer all questions now.`,
            },
          ],
          details: { cursor, total: chunks.length, done: true },
        };
      }
      const maxBatch = Math.max(1, Number(process.env.PI_STREAM_MAX_BATCH_COUNT) || 3);
      const rawCount = typeof params.count === "number" ? Math.floor(params.count) : Math.min(2, maxBatch);
      const take = Math.max(1, Math.min(maxBatch, rawCount));
      const parts: string[] = [];
      for (let k = 0; k < take && cursor < chunks.length; k++) {
        parts.push(`=== BATCH ${cursor + 1} OF ${chunks.length} ===\n${chunks[cursor]}`);
        cursor += 1;
      }
      const footer =
        cursor >= chunks.length
          ? `\n\n[STREAM STATUS: ${cursor}/${chunks.length} — END_OF_STREAM reached. All batches ingested.]`
          : `\n\n[STREAM STATUS: ${cursor}/${chunks.length} ingested — ${chunks.length - cursor} remaining. Call read_next_batch again.]`;
      return {
        content: [{ type: "text", text: parts.join("\n\n") + footer }],
        details: { cursor, total: chunks.length, done: cursor >= chunks.length },
      };
    },
  });
}
