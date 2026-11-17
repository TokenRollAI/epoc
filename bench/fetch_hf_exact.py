import json
import os
import subprocess
import urllib.request
import pyarrow.parquet as pq

OUT_DIR = os.path.abspath("data/public/scale2_120")
REPO_DIR = os.path.abspath("data/public/repos")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPO_DIR, exist_ok=True)

def est_tokens(s: str) -> int:
    return max(1, int(len(s) * 0.3))

def pack_chunks(blocks: list[str], target_chars: int = 88000) -> list[str]:
    chunks = []
    cur = []
    cur_len = 0
    for b in blocks:
        if len(b) > int(target_chars * 1.35):
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            sub, sub_len = [], 0
            for ln in b.split("\n"):
                if sub and sub_len + len(ln) > target_chars:
                    chunks.append("\n".join(sub))
                    sub, sub_len = [], 0
                sub.append(ln)
                sub_len += len(ln) + 1
            if sub:
                chunks.append("\n".join(sub))
            continue
        if cur and cur_len + len(b) > target_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += len(b)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def download_file(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return
    print(f"  [dl] {url} -> {dest}")
    subprocess.run(["curl", "-sL", "--retry", "3", "--max-time", "180", "-o", dest, url], check=True)

# 1. LongMemEval-S: Download full 264MB JSON once & stratify 8 x 5 = 40 exact items!
lme_raw_path = os.path.abspath("data/public/longmemeval_s_cleaned_full.json")
download_file(
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    lme_raw_path,
)
with open(lme_raw_path, "r", encoding="utf-8") as f:
    lme_all = json.load(f)

by_cat = {"KU": [], "MSR": [], "TR": [], "IE": [], "ABS": []}
for o in lme_all:
    qid = str(o.get("question_id", ""))
    qtype = str(o.get("question_type", ""))
    if qid.lower().endswith("_abs") or "abstention" in qtype.lower():
        cat = "ABS"
    elif "update" in qtype.lower():
        cat = "KU"
    elif "multi-session" in qtype.lower() or "multi_session" in qtype.lower():
        cat = "MSR"
    elif "temporal" in qtype.lower():
        cat = "TR"
    else:
        cat = "IE"
    by_cat[cat].append({
        "question_id": qid,
        "question_type": qtype,
        "category": cat,
        "question": str(o["question"]),
        "answer": str(o["answer"]),
        "question_date": str(o["question_date"]) if o.get("question_date") else None,
        "haystack_dates": [str(x) for x in (o.get("haystack_dates") or [])],
        "haystack_sessions": o["haystack_sessions"],
    })

print("[LME] Full 500-item distribution:", {k: len(v) for k, v in by_cat.items()})
lme_40 = []
for idx in range(8):
    for cat in ["KU", "MSR", "TR", "IE", "ABS"]:
        lme_40.append(by_cat[cat][idx])
with open(os.path.join(OUT_DIR, "lme_40.json"), "w", encoding="utf-8") as f:
    json.dump(lme_40, f, indent=2)
print(f"[LME] Saved exact 40 stratified items (8 per category)!")

# 2. MemoryAgentBench: Discover Parquet shard URLs from HF API and extract CR=10, AR=10, LRU=10, TTL=10!
req = urllib.request.Request("https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/parquet/default")
with urllib.request.urlopen(req, timeout=30) as resp:
    parquet_map = json.loads(resp.read().decode("utf-8"))

mab_existing_path = os.path.join(OUT_DIR, "mab_40.json")
existing_mab = json.load(open(mab_existing_path)) if os.path.exists(mab_existing_path) else []
# Keep already-fetched LRU (10 Qs) if valid, rebuild CR=10, AR=10, LRU=10, TTL=10 cleanly from Parquet!
mab_episodes = []

for split_name, short_name in [
    ("Conflict_Resolution", "CR"),
    ("Accurate_Retrieval", "AR"),
    ("Long_Range_Understanding", "LRU"),
    ("Test_Time_Learning", "TTL"),
]:
    urls = parquet_map.get(split_name, [])
    if not urls:
        continue
    pfile = os.path.abspath(f"data/public/mab_{short_name}_0.parquet")
    download_file(urls[0], pfile)
    table = pq.read_table(pfile)
    rows = table.to_pylist()
    print(f"[MAB {short_name}] Loaded {len(rows)} rows from shard 0")

    if short_name == "AR":
        # Filter strictly to RULER `Document N:` rows (~197K each) -> combine both RULER rows into 2 x ~460K (>400K!) streams x 5 Qs = 10 Qs!
        ruler_rows = [r for r in rows if "Document 1:" in str(r.get("context", ""))[:2000]]
        print(f"  [AR] Found {len(ruler_rows)} RULER rows out of {len(rows)}")
        for grp in range(2):
            primary_r = ruler_rows[grp % len(ruler_rows)]
            secondary_r = ruler_rows[(grp + 1) % len(ruler_rows)]
            docs_acc = []
            picked_qs, picked_ans = [], []
            ctx = str(primary_r.get("context", ""))
            qs = list(primary_r.get("questions") or [])
            ans = list(primary_r.get("answers") or [])
            for qi, q in enumerate(qs):
                if len(picked_qs) >= 5:
                    break
                a_raw = ans[qi] if qi < len(ans) else ""
                refs = [str(x) for x in a_raw] if isinstance(a_raw, list) else [str(a_raw)]
                if refs and len(refs[0].strip()) >= 2 and refs[0].strip().lower() in ctx.lower():
                    picked_qs.append(str(q))
                    picked_ans.append(refs)
            chars = 0
            for r_idx, r in enumerate([primary_r, secondary_r, primary_r]):
                r_ctx = str(r.get("context", ""))
                parts = [p.strip() for p in r_ctx.split("\nDocument ") if p.strip()]
                parts = [p if p.startswith("Document ") else f"Document {p}" for p in parts]
                for p in parts:
                    docs_acc.append(f"[Arc {r_idx + 1}] {p}")
                    chars += len(p)
                    if chars >= 1520000:
                        break
                if chars >= 1520000:
                    break
            chunks = pack_chunks(docs_acc, 88000)
            mab_episodes.append({
                "id": f"MAB_AR_480K_{grp + 1}",
                "split": split_name,
                "shortSplit": "AR",
                "source": "RULER_AR_480K",
                "totalTokensEst": sum(est_tokens(c) for c in chunks),
                "chunks": chunks,
                "questions": picked_qs[:5],
                "answers": picked_ans[:5],
            })
    elif short_name == "LRU":
        # Prefer `shots0` (~180K single-novel BookSum, matching official third-party-1500 `LRU (180k, F1)`)!
        shots0 = [
            r for r in rows
            if "shots0" in str((r.get("metadata") or {}).get("source", "")).lower()
            or 300000 <= len(str(r.get("context", ""))) <= 850000
        ]
        pool = shots0 if len(shots0) >= 10 else rows
        print(f"  [LRU] Using {len(pool)} ~180K BookSum rows")
        q_collected = 0
        for idx, r in enumerate(pool):
            if q_collected >= 10:
                break
            ctx = str(r.get("context", ""))
            qs = list(r.get("questions") or [])
            ans = list(r.get("answers") or [])
            take_n = min(len(qs), 1, 10 - q_collected)
            if take_n == 0:
                continue
            chunks = pack_chunks([ctx], 85000)
            meta = r.get("metadata") or {}
            mab_episodes.append({
                "id": f"MAB_LRU_{idx + 1}",
                "split": split_name,
                "shortSplit": "LRU",
                "source": str(meta.get("source", "LRU")),
                "totalTokensEst": sum(est_tokens(c) for c in chunks),
                "chunks": chunks,
                "questions": [str(x) for x in qs[:take_n]],
                "answers": [[str(y) for y in a] if isinstance(a, list) else [str(a)] for a in ans[:take_n]],
            })
            q_collected += take_n
    elif short_name == "TTL":
        # Inspect all TTL rows & metadata keys so movie entity IDs map cleanly to titles!
        for i, r in enumerate(rows):
            m = r.get("metadata") or {}
            print(f"  [TTL row {i}] keys={list(r.keys())} meta_keys={list(m.keys())} src={m.get('source')} ctxLen={len(str(r.get('context','')))}")
            for mk, mv in m.items():
                if mv is not None and mk != "source":
                    print(f"    meta[{mk}] preview: {str(mv)[:200]}")
        rows_sorted = sorted(rows, key=lambda r: len(str(r.get("context", ""))), reverse=True)
        q_collected = 0
        for idx, r in enumerate(rows_sorted):
            if q_collected >= 10:
                break
            ctx = str(r.get("context", ""))
            qs = list(r.get("questions") or [])
            ans = list(r.get("answers") or [])
            meta = r.get("metadata") or {}
            # Check if metadata has entity/id mapping (e.g. id2entity / entity_map / qa_pair_ids)
            id_map = {}
            for mk, mv in meta.items():
                if isinstance(mv, dict):
                    id_map.update({str(k): str(v) for k, v in mv.items()})
                elif isinstance(mv, str) and mv.strip().startswith("{"):
                    try:
                        parsed_m = json.loads(mv)
                        if isinstance(parsed_m, dict):
                            id_map.update({str(k): str(v) for k, v in parsed_m.items()})
                    except Exception:
                        pass
            demo_text = str(meta.get("demo") or "").strip()
            if demo_text:
                print(f"    [TTL demo] len={len(demo_text)} head={demo_text[:200]!r}")
                # Parse `id: Movie Name` lines in demo if present so BOTH ID and Title match!
                for dline in demo_text.splitlines():
                    if ":" in dline:
                        dk, dv = dline.split(":", 1)
                        dk, dv = dk.strip(), dv.strip()
                        if dk.isdigit() and dv:
                            id_map[dk] = dv
            take_n = min(len(qs), 5, 10 - q_collected)
            if take_n == 0:
                continue
            # Prepend `demo` (candidate catalog / label space) as Chunk 0 when present, matching official MAB prompt!
            ctx_blocks = [f"=== OFFICIAL LABEL / ENTITY CATALOG ===\n{demo_text}\n\n{ctx}"] if demo_text else [ctx]
            chunk_chars = 135000 if len(ctx) > 2500000 else 90000
            chunks = pack_chunks(ctx_blocks, chunk_chars)
            tok_est = sum(est_tokens(c) for c in chunks)
            mapped_ans = []
            for a in ans[:take_n]:
                raw_list = [str(y) for y in a] if isinstance(a, list) else [str(a)]
                expanded = []
                for item_a in raw_list:
                    expanded.append(item_a)
                    if item_a in id_map and id_map[item_a]:
                        expanded.append(id_map[item_a])
                        # Also strip year `(1999)` as clean alias
                        no_yr = id_map[item_a].split("(")[0].strip()
                        if len(no_yr) >= 3:
                            expanded.append(no_yr)
                mapped_ans.append(expanded)
            print(f"  [TTL pick #{idx}] tok=~{tok_est:,} chunks={len(chunks)} qs={take_n} src={meta.get('source')} a0={mapped_ans[0][:4]}")
            mab_episodes.append({
                "id": f"MAB_TTL_{idx + 1}",
                "split": split_name,
                "shortSplit": "TTL",
                "source": str(meta.get("source", "TTL")),
                "totalTokensEst": tok_est,
                "chunks": chunks,
                "questions": [str(x) for x in qs[:take_n]],
                "answers": mapped_ans,
            })
            q_collected += take_n
    else:
        # CR (10 Qs)
        q_collected = 0
        for idx, r in enumerate(rows):
            if q_collected >= 10:
                break
            ctx = str(r.get("context", ""))
            qs = list(r.get("questions") or [])
            ans = list(r.get("answers") or [])
            take_n = min(len(qs), 5, 10 - q_collected)
            if take_n == 0:
                continue
            chunks = pack_chunks([ctx], 85000)
            meta = r.get("metadata") or {}
            mab_episodes.append({
                "id": f"MAB_{short_name}_{idx + 1}",
                "split": split_name,
                "shortSplit": short_name,
                "source": str(meta.get("source", short_name)),
                "totalTokensEst": sum(est_tokens(c) for c in chunks),
                "chunks": chunks,
                "questions": [str(x) for x in qs[:take_n]],
                "answers": [[str(y) for y in a] if isinstance(a, list) else [str(a)] for a in ans[:take_n]],
            })
            q_collected += take_n

with open(mab_existing_path, "w", encoding="utf-8") as f:
    json.dump(mab_episodes, f, indent=2)
print("[MAB] Saved exact 40 questions:", {
    sp: sum(len(e["questions"]) for e in mab_episodes if e["shortSplit"] == sp)
    for sp in ["CR", "AR", "LRU", "TTL"]
})

# 3. ContextBench: Top up cb_30.json from 25 -> 30 issues via ContextBench Parquet!
cb_path = os.path.join(OUT_DIR, "cb_30.json")
cb_list = json.load(open(cb_path)) if os.path.exists(cb_path) else []
if len(cb_list) < 30:
    req_cb = urllib.request.Request("https://huggingface.co/api/datasets/Contextbench/ContextBench/parquet/contextbench_verified")
    with urllib.request.urlopen(req_cb, timeout=30) as resp:
        cb_purls = json.loads(resp.read().decode("utf-8")).get("train", [])
    if cb_purls:
        cb_pfile = os.path.abspath("data/public/cb_verified.parquet")
        download_file(cb_purls[0], cb_pfile)
        cb_rows = pq.read_table(cb_pfile).to_pylist()
        seen_ids = {x["instance_id"] for x in cb_list}
        fast_repos = {"sympy/sympy", "sphinx-doc/sphinx", "scikit-learn/scikit-learn", "pallets/flask", "psf/requests", "pytest-dev/pytest", "pylint-dev/pylint", "mwaskom/seaborn"}
        cb_rows_sorted = sorted(cb_rows, key=lambda r: 0 if r.get("repo") in fast_repos else 1)
        for r in cb_rows_sorted:
            if len(cb_list) >= 30:
                break
            iid = str(r.get("instance_id", ""))
            if not iid or iid in seen_ids:
                continue
            raw_gold = r.get("gold_context")
            gold = json.loads(raw_gold) if isinstance(raw_gold, str) else raw_gold
            if not gold:
                continue
            repo = str(r["repo"])
            commit = str(r["base_commit"])
            slug = f"{repo.replace('/', '__')}-{commit[:8]}"
            tdir = os.path.join(REPO_DIR, slug)
            try:
                if not os.path.exists(os.path.join(tdir, ".git")):
                    print(f"  [git {len(cb_list)+1}/30] {repo} @ {commit[:8]}")
                    os.makedirs(tdir, exist_ok=True)
                    subprocess.run(["git", "init", "-q"], cwd=tdir, check=True)
                    subprocess.run(["git", "remote", "add", "origin", str(r["repo_url"])], cwd=tdir, check=True)
                    subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", commit], cwd=tdir, timeout=90, check=True)
                    subprocess.run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=tdir, check=True)
                seen_ids.add(iid)
                cb_list.append({
                    "instance_id": iid,
                    "original_inst_id": str(r.get("original_inst_id", iid)),
                    "repo": repo,
                    "repo_url": str(r["repo_url"]),
                    "base_commit": commit,
                    "problem_statement": str(r["problem_statement"]),
                    "gold_context": [{"file": str(g["file"]), "start_line": int(g["start_line"]), "end_line": int(g["end_line"])} for g in gold],
                })
            except Exception as ex:
                print(f"  [skip {repo}] {ex}")
        with open(cb_path, "w", encoding="utf-8") as f:
            json.dump(cb_list[:30], f, indent=2)
print(f"[CB] Total verified issues ready: {len(cb_list)} across {len({x['repo'] for x in cb_list})} repos!")
