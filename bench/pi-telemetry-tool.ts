/**
 * Benchmark helper extension for `oversized-tools` suite (`bench/pi-billion-bench.ts`).
 *
 * Exposes `fetch_telemetry_dump` — simulating an MCP / observability tool (like Datadog,
 * Kubernetes pod logs, or OpenTelemetry trace export) that returns a large (~105K-token)
 * JSON-lines cluster dump WITHOUT host shell truncation, so we can measure how each
 * 400K harness handles oversized low-entropy tool outputs in `pi --mode json`.
 */

import fs from "node:fs";

export default function telemetryToolExtension(pi: any): void {
  const dumpFile = process.env.PI_TELEMETRY_DUMP_FILE;
  if (!dumpFile) return;

  pi.registerTool({
    name: "fetch_telemetry_dump",
    label: "Fetch Cluster Telemetry Dump",
    description:
      "Export the raw cluster shard telemetry & OOM trace records from the incident telemetry buffer.",
    parameters: {
      type: "object",
      properties: {
        cluster: {
          type: "string",
          description: "Cluster name (e.g. 'prod-eu-west').",
        },
      },
      additionalProperties: false,
    },
    async execute() {
      const raw = fs.readFileSync(dumpFile, "utf8");
      return {
        content: [{ type: "text", text: raw }],
      };
    },
  });
}
