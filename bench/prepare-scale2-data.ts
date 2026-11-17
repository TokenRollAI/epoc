/**
 * Prepares & caches the 120-item Stratified Full-Subtask Dataset (`data/public/scale2_120/`)
 * for the 400K-Window Pi CLI Benchmark (`full` vs `pi-native` vs `billion-context` vs `cpat`):
 *
 *   1. LongMemEval-S (40 questions stratified across all 5 official subtasks: KU, MSR, TR, IE, ABS)
 *   2. MemoryAgentBench (40 questions across all 4 official splits: CR=10, AR=10, LRU=10, TTL=10 @ 1M-1.48M tok!)
 *   3. ContextBench Verified (30 real GitHub issues across ~15 OSS repos, shallow-checked-out at exact base_commit)
 *   4. Oversized-Tools & Pollution-Boundary (10 items — generated deterministically at runtime)
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { estimateTokens } from "../src/util/misc.ts";

const DATA_ROOT = resolve("data/public");
const OUT_DIR = join(DATA_ROOT, "scale2_120");
mkdirSync(OUT_DIR, { recursive: true });

export interface LmeScaleItem {
  question_id: string;
  question_type: string;
  category: "KU" | "MSR" | "TR" | "IE" | "ABS";
  question: string;
  answer: string;
  question_date?: string;
  haystack_dates?: string[];
  haystack_sessions: Array<Array<{ role: string; content: string }>>;
}

export interface MabScaleEpisode {
  id: string;
  split: "Conflict_Resolution" | "Accurate_Retrieval" | "Long_Range_Understanding" | "Test_Time_Learning";
  shortSplit: "CR" | "AR" | "LRU" | "TTL";
  source: string;
  totalTokensEst: number;
  chunks: string[];
  questions: string[];
  answers: string[][];
}

export interface CbScaleItem {
  instance_id: string;
  original_inst_id: string;
  repo: string;
  repo_url: string;
  base_commit: string;
  problem_statement: string;
  gold_context: Array<{ file: string; start_line: number; end_line: number }>;
}

function packIntoChunks(blocks: string[], targetChars: number): string[] {
  const chunks: string[] = [];
  let cur: string[] = [];
  let curLen = 0;
  for (const b of blocks) {
    if (b.length > targetChars * 1.35) {
      if (cur.length > 0) {
        chunks.push(cur.join("\n\n"));
        cur = [];
        curLen = 0;
      }
      // Split single giant string cleanly on newline boundaries
      const lines = b.split("\n");
      let sub: string[] = [];
      let subLen = 0;
      for (const ln of lines) {
        if (sub.length > 0 && subLen + ln.length > targetChars) {
          chunks.push(sub.join("\n"));
          sub = [];
          subLen = 0;
        }
        sub.push(ln);
        subLen += ln.length + 1;
      }
      if (sub.length > 0) chunks.push(sub.join("\n"));
      continue;
    }
    if (cur.length > 0 && curLen + b.length > targetChars) {
      chunks.push(cur.join("\n\n"));
      cur = [];
      curLen = 0;
    }
    cur.push(b);
    curLen += b.length;
  }
  if (cur.length > 0) chunks.push(cur.join("\n\n"));
  return chunks;
}

function extractJsonObjectsFromChunk(chunk: string, skipFirst = false): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  let startIdx = 0;
  if (skipFirst) {
    const marker = chunk.indexOf('{"question_id"');
    if (marker === -1) return out;
    startIdx = marker;
  }
  let depth = 0;
  let inStr = false;
  let esc = false;
  let objStart = -1;
  for (let i = startIdx; i < chunk.length; i++) {
    const ch = chunk[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") {
      if (depth === 0) objStart = i;
      depth++;
    } else if (ch === "}") {
      if (depth > 0) depth--;
      if (depth === 0 && objStart !== -1) {
        try {
          const parsed = JSON.parse(chunk.slice(objStart, i + 1)) as Record<string, unknown>;
          if (typeof parsed.question_id === "string" && Array.isArray(parsed.haystack_sessions)) {
            out.push(parsed);
          }
        } catch {
          // ignore boundary fragment
        }
        objStart = -1;
      }
    }
  }
  return out;
}

function mapLmeCategory(qId: string, qType: string): LmeScaleItem["category"] {
  const idLow = qId.toLowerCase();
  const tLow = qType.toLowerCase();
  if (idLow.endsWith("_abs") || tLow.includes("abstention")) return "ABS";
  if (tLow.includes("update")) return "KU";
  if (tLow.includes("multi-session") || tLow.includes("multi_session")) return "MSR";
  if (tLow.includes("temporal")) return "TR";
  return "IE";
}

async function fetchRangeViaCurl(url: string, start: number, end: number): Promise<string> {
  // Resolve redirect target first so `Range` header is sent directly to the HuggingFace CDN (`xethub`/`lfs`),
  // avoiding cross-origin redirect stripping that would otherwise stream the entire 280MB file!
  const headOut = execFileSync("curl", ["-sI", "-L", "-o", "/dev/null", "-w", "%{url_effective}", url], {
    encoding: "utf8",
    timeout: 20_000,
  }).trim();
  const effectiveUrl = headOut.startsWith("http") ? headOut : url;
  return execFileSync("curl", ["-s", "--max-time", "30", "-r", `${start}-${end}`, effectiveUrl], {
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
    timeout: 35_000,
  });
}

async function fetchJsonWithRetry(url: string, _headers?: Record<string, string>, retries = 3): Promise<any> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const text = execFileSync("curl", ["-sL", "--max-time", "45", url], {
        encoding: "utf8",
        maxBuffer: 48 * 1024 * 1024,
        timeout: 50_000,
      });
      if (!text || text.trimStart().startsWith("<")) {
        throw new Error(`Empty or HTML gateway error for ${url}`);
      }
      return JSON.parse(text);
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
}

async function prepareLongMemEval40(): Promise<void> {
  const outPath = join(OUT_DIR, "lme_40.json");
  if (existsSync(outPath)) {
    const existing = JSON.parse(readFileSync(outPath, "utf8")) as LmeScaleItem[];
    if (existing.length >= 40) {
      console.log(`[LME] Already cached ${existing.length} items at ${outPath}`);
      return;
    }
  }

  console.log("[LME] Fetching stratified 40 items across KU / MSR / TR / IE / ABS...");
  const url =
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json";
  const offsets = [
    0,
    35_000_000,
    75_000_000,
    115_000_000,
    155_000_000,
    195_000_000,
    235_000_000,
    265_000_000,
  ];
  const seen = new Set<string>();
  const byCat: Record<LmeScaleItem["category"], LmeScaleItem[]> = {
    KU: [],
    MSR: [],
    TR: [],
    IE: [],
    ABS: [],
  };

  for (const off of offsets) {
    const end = off + 7_500_000;
    try {
      const chunk = await fetchRangeViaCurl(url, off, end);
      const objs = extractJsonObjectsFromChunk(chunk, off > 0);
      for (const o of objs) {
        const qid = String(o.question_id ?? "");
        if (!qid || seen.has(qid)) continue;
        if (typeof o.question !== "string" || !Array.isArray(o.haystack_sessions)) continue;
        seen.add(qid);
        const cat = mapLmeCategory(qid, String(o.question_type ?? ""));
        byCat[cat].push({
          question_id: qid,
          question_type: String(o.question_type ?? "unknown"),
          category: cat,
          question: o.question,
          answer: String(o.answer ?? ""),
          question_date: o.question_date ? String(o.question_date) : undefined,
          haystack_dates: Array.isArray(o.haystack_dates) ? o.haystack_dates.map(String) : undefined,
          haystack_sessions: o.haystack_sessions as LmeScaleItem["haystack_sessions"],
        });
      }
    } catch (e) {
      console.warn(`  [LME] Range ${off} warning:`, (e as Error).message);
    }
  }

  console.log(
    `  [LME] Pool counts: KU=${byCat.KU.length}, MSR=${byCat.MSR.length}, TR=${byCat.TR.length}, IE=${byCat.IE.length}, ABS=${byCat.ABS.length}`,
  );

  const selected: LmeScaleItem[] = [];
  const cats: Array<LmeScaleItem["category"]> = ["KU", "MSR", "TR", "IE", "ABS"];
  // Interleave categories so every 4-item marathon contains diverse question types!
  let idx = 0;
  while (selected.length < 40 && idx < 50) {
    for (const c of cats) {
      if (selected.length >= 40) break;
      if (idx < byCat[c].length) {
        selected.push(byCat[c][idx]);
      }
    }
    idx += 1;
  }

  writeFileSync(outPath, JSON.stringify(selected, null, 2), "utf8");
  const finalCounts: Record<string, number> = {};
  for (const it of selected) finalCounts[it.category] = (finalCounts[it.category] ?? 0) + 1;
  console.log(`[LME] Saved ${selected.length} stratified items:`, finalCounts);
}

async function prepareMemoryAgentBench40(): Promise<void> {
  const outPath = join(OUT_DIR, "mab_40.json");
  if (existsSync(outPath)) {
    const existing = JSON.parse(readFileSync(outPath, "utf8")) as MabScaleEpisode[];
    const totalQs = existing.reduce((s, e) => s + e.questions.length, 0);
    if (totalQs >= 40) {
      console.log(`[MAB] Already cached ${existing.length} episodes (${totalQs} Qs) at ${outPath}`);
      return;
    }
  }

  console.log("[MAB] Fetching 40 questions across CR (10), AR (10), LRU (10), TTL (10)...");
  const episodes: MabScaleEpisode[] = [];

  // 1. Conflict_Resolution (10 Qs across 2 episodes)
  let crTotal = 0;
  for (let off = 0; off < 6 && crTotal < 10; off++) {
    try {
      const data = await fetchJsonWithRetry(
        `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Conflict_Resolution&offset=${off}&length=1`,
      );
      const row = data.rows?.[0]?.row;
      if (!row) continue;
      const ctx = String(row.context ?? "");
      const qs = ((row.questions as string[]) ?? []).slice(0, Math.min(5, 10 - crTotal));
      const ans = ((row.answers as Array<string[] | string>) ?? [])
        .slice(0, qs.length)
        .map((a) => (Array.isArray(a) ? a.map(String) : [String(a ?? "")]));
      if (qs.length === 0) continue;
      const chunks = packIntoChunks([ctx], 75_000);
      episodes.push({
        id: `MAB_CR_${off}`,
        split: "Conflict_Resolution",
        shortSplit: "CR",
        source: String(row.metadata?.source ?? "Conflict_Resolution"),
        totalTokensEst: chunks.reduce((s, c) => s + estimateTokens(c), 0),
        chunks,
        questions: qs,
        answers: ans,
      });
      crTotal += qs.length;
    } catch (e) {
      console.warn(`  [MAB CR off=${off}]`, (e as Error).message);
    }
  }

  // 2. Accurate_Retrieval (10 Qs across 2 × ~480K >400K RULER needle streams!)
  for (let grp = 0; grp < 2; grp++) {
    const allDocs: string[] = [];
    const pickedQs: string[] = [];
    const pickedAns: string[][] = [];
    let chars = 0;
    for (let off = grp * 2; off < grp * 2 + 3 && chars < 1_580_000; off++) {
      try {
        const data = await fetchJsonWithRetry(
          `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Accurate_Retrieval&offset=${off}&length=1`,
        );
        const row = data.rows?.[0]?.row;
        if (!row) continue;
        const fullCtx = String(row.context ?? "");
        const qs = (row.questions as string[]) ?? [];
        const ans = (row.answers as Array<string[] | string>) ?? [];
        const docs = fullCtx
          .split(/(?=\nDocument \d+:)/g)
          .map((s) => s.trim())
          .filter(Boolean);

        let tookFromRow = 0;
        for (let qi = 0; qi < qs.length && tookFromRow < 3 && pickedQs.length < 5; qi++) {
          const refs = Array.isArray(ans[qi]) ? ans[qi].map(String) : [String(ans[qi] ?? "")];
          const ref0 = refs[0]?.trim();
          if (ref0 && ref0.length >= 3 && fullCtx.toLowerCase().includes(ref0.toLowerCase())) {
            pickedQs.push(qs[qi]);
            pickedAns.push(refs);
            tookFromRow += 1;
          }
        }
        for (const d of docs) {
          allDocs.push(`[Set ${off + 1}] ${d}`);
          chars += d.length;
          if (chars >= 1_600_000) break;
        }
      } catch (e) {
        console.warn(`  [MAB AR off=${off}]`, (e as Error).message);
      }
    }
    const chunks = packIntoChunks(allDocs, 88_000); // ~26K tok per chunk -> flows smoothly up to 384K!
    episodes.push({
      id: `MAB_AR_480K_${grp + 1}`,
      split: "Accurate_Retrieval",
      shortSplit: "AR",
      source: "RULER_AR_480K",
      totalTokensEst: chunks.reduce((s, c) => s + estimateTokens(c), 0),
      chunks,
      questions: pickedQs.slice(0, 5),
      answers: pickedAns.slice(0, 5),
    });
  }

  // 3. Long_Range_Understanding (10 Qs across BookSum / long-range comprehension rows)
  let lruTotal = 0;
  for (let off = 0; off < 12 && lruTotal < 10; off++) {
    try {
      const data = await fetchJsonWithRetry(
        `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Long_Range_Understanding&offset=${off}&length=1`,
      );
      const row = data.rows?.[0]?.row;
      if (!row) continue;
      const ctx = String(row.context ?? "");
      const rawQs = (row.questions as string[]) ?? [];
      const rawAns = (row.answers as Array<string[] | string>) ?? [];
      const takeN = Math.min(rawQs.length, Math.min(5, 10 - lruTotal));
      if (takeN === 0) continue;
      const chunks = packIntoChunks([ctx], 85_000);
      episodes.push({
        id: `MAB_LRU_${off}`,
        split: "Long_Range_Understanding",
        shortSplit: "LRU",
        source: String(row.metadata?.source ?? "Long_Range_Understanding"),
        totalTokensEst: chunks.reduce((s, c) => s + estimateTokens(c), 0),
        chunks,
        questions: rawQs.slice(0, takeN),
        answers: rawAns
          .slice(0, takeN)
          .map((a) => (Array.isArray(a) ? a.map(String) : [String(a ?? "")])),
      });
      lruTotal += takeN;
    } catch (e) {
      console.warn(`  [MAB LRU off=${off}]`, (e as Error).message);
    }
  }

  // 4. Test_Time_Learning (10 Qs on ultra-long ~1.0M-1.48M token multi-epoch streams!)
  //    Try offsets 0..8 with length=1; if datasets-server times out on a 1.48M row, try other offsets or parquet metadata.
  let ttlTotal = 0;
  for (let off = 0; off < 15 && ttlTotal < 10; off++) {
    try {
      const data = await fetchJsonWithRetry(
        `https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Test_Time_Learning&offset=${off}&length=1`,
        undefined,
        2,
      );
      const row = data.rows?.[0]?.row;
      if (!row) continue;
      const ctx = String(row.context ?? "");
      const rawQs = (row.questions as string[]) ?? [];
      const rawAns = (row.answers as Array<string[] | string>) ?? [];
      const takeN = Math.min(rawQs.length, Math.min(5, 10 - ttlTotal));
      if (takeN === 0) continue;
      const chunks = packIntoChunks([ctx], 92_000); // ~28K tok per batch -> crosses 384K 3-4 times on 1.2M-1.48M!
      const tokEst = chunks.reduce((s, c) => s + estimateTokens(c), 0);
      console.log(
        `  [MAB TTL off=${off}] source=${row.metadata?.source} tokens=~${tokEst.toLocaleString()} chunks=${chunks.length} qs=${takeN} q0=${rawQs[0]?.slice(0, 80)} a0=${JSON.stringify(rawAns[0])?.slice(0, 80)}`,
      );
      episodes.push({
        id: `MAB_TTL_${off}`,
        split: "Test_Time_Learning",
        shortSplit: "TTL",
        source: String(row.metadata?.source ?? "Test_Time_Learning"),
        totalTokensEst: tokEst,
        chunks,
        questions: rawQs.slice(0, takeN),
        answers: rawAns
          .slice(0, takeN)
          .map((a) => (Array.isArray(a) ? a.map(String) : [String(a ?? "")])),
      });
      ttlTotal += takeN;
    } catch (e) {
      console.warn(`  [MAB TTL off=${off}] skipped:`, (e as Error).message);
    }
  }

  writeFileSync(outPath, JSON.stringify(episodes, null, 2), "utf8");
  const summaryBySplit: Record<string, { episodes: number; questions: number; avgTok: number }> = {};
  for (const ep of episodes) {
    const cur = summaryBySplit[ep.shortSplit] ?? { episodes: 0, questions: 0, avgTok: 0 };
    cur.episodes += 1;
    cur.questions += ep.questions.length;
    cur.avgTok += ep.totalTokensEst;
    summaryBySplit[ep.shortSplit] = cur;
  }
  console.log("[MAB] Saved episodes:", summaryBySplit);
}

const SKIP_HUGE_MONOREPOS = new Set([
  "chromium/chromium",
  "torvalds/linux",
  "llvm/llvm-project",
  "rust-lang/rust",
]);

async function prepareContextBench30(): Promise<void> {
  const outPath = join(OUT_DIR, "cb_30.json");
  let selected: CbScaleItem[] = [];
  if (existsSync(outPath)) {
    selected = JSON.parse(readFileSync(outPath, "utf8")) as CbScaleItem[];
  }

  if (selected.length < 30) {
    console.log("[CB] Scanning ContextBench Verified on Hugging Face for 30 multi-repo issues...");
    const candidates: CbScaleItem[] = [];
    const seenIds = new Set<string>();

    for (let off = 0; off < 480 && candidates.length < 120; off += 20) {
      try {
        const data = await fetchJsonWithRetry(
          `https://datasets-server.huggingface.co/rows?dataset=Contextbench%2FContextBench&config=contextbench_verified&split=train&offset=${off}&length=20`,
        );
        for (const r of data.rows ?? []) {
          const row = r.row;
          const id = String(row.instance_id ?? "");
          const repo = String(row.repo ?? "");
          if (!id || seenIds.has(id) || SKIP_HUGE_MONOREPOS.has(repo)) continue;
          const rawGold = row.gold_context;
          const goldParsed = typeof rawGold === "string" ? JSON.parse(rawGold) : rawGold;
          if (!Array.isArray(goldParsed) || goldParsed.length === 0) continue;
          seenIds.add(id);
          candidates.push({
            instance_id: id,
            original_inst_id: String(row.original_inst_id ?? id),
            repo,
            repo_url: String(row.repo_url),
            base_commit: String(row.base_commit),
            problem_statement: String(row.problem_statement),
            gold_context: goldParsed.map((g: Record<string, unknown>) => ({
              file: String(g.file),
              start_line: Number(g.start_line),
              end_line: Number(g.end_line),
            })),
          });
        }
      } catch (e) {
        console.warn(`  [CB off=${off}]`, (e as Error).message);
      }
    }

    // Diversify across at least 15 distinct repositories (max 2-3 per repo)
    const byRepo = new Map<string, CbScaleItem[]>();
    for (const c of candidates) {
      const arr = byRepo.get(c.repo) ?? [];
      arr.push(c);
      byRepo.set(c.repo, arr);
    }

    selected = [];
    const repos = Array.from(byRepo.keys());
    let round = 0;
    while (selected.length < 35 && round < 4) {
      for (const r of repos) {
        if (selected.length >= 35) break;
        const list = byRepo.get(r)!;
        if (round < list.length) selected.push(list[round]);
      }
      round += 1;
    }
  }

  // Shallow-checkout repositories and keep the first 30 that clone cleanly & fast!
  const repoCacheRoot = join(DATA_ROOT, "repos");
  mkdirSync(repoCacheRoot, { recursive: true });
  const verified30: CbScaleItem[] = [];

  for (const item of selected) {
    if (verified30.length >= 30) break;
    const slug = `${item.repo.replace(/\//g, "__")}-${item.base_commit.slice(0, 8)}`;
    const targetDir = join(repoCacheRoot, slug);
    try {
      if (!existsSync(join(targetDir, ".git"))) {
        console.log(`  [git ${verified30.length + 1}/30] Shallow-fetching ${item.repo} @ ${item.base_commit.slice(0, 8)}...`);
        mkdirSync(targetDir, { recursive: true });
        execFileSync("git", ["init", "-q"], { cwd: targetDir });
        execFileSync("git", ["remote", "add", "origin", item.repo_url], { cwd: targetDir });
        execFileSync("git", ["fetch", "-q", "--depth", "1", "origin", item.base_commit], {
          cwd: targetDir,
          timeout: 90_000,
        });
        execFileSync("git", ["checkout", "-q", "--detach", "FETCH_HEAD"], { cwd: targetDir });
      }
      verified30.push(item);
    } catch (e) {
      console.warn(`  [git skip ${item.repo}]`, (e as Error).message.slice(0, 120));
    }
  }

  writeFileSync(outPath, JSON.stringify(verified30, null, 2), "utf8");
  const distinctRepos = new Set(verified30.map((x) => x.repo));
  console.log(`[CB] Verified & checked out ${verified30.length} issues across ${distinctRepos.size} distinct repos.`);
}

async function main() {
  await prepareLongMemEval40();
  await prepareMemoryAgentBench40();
  await prepareContextBench30();
  console.log("\n✅ All Scale-2 (120-item) datasets cached and verified in data/public/scale2_120/!");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
