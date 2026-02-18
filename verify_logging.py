import json
from pathlib import Path

LOG_PATH = Path(__file__).with_name("rag_logs.jsonl")

REQUIRED_PATHS = [
    ("question",),
    ("answer",),
    ("retrieval_trace", "dense_hits"),
    ("retrieval_trace", "keyword_hits"),
    ("retrieval_trace", "fused_results_pre_rerank"),
    ("final_context", "chunks_sent"),
    ("final_context", "context_text"),
]

def get_path(d, path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def main():
    if not LOG_PATH.exists():
        raise SystemExit(f"❌ {LOG_PATH} not found")

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("❌ rag_logs.jsonl is empty. Make ONE /ask request first.")

    entry = json.loads(lines[-1])

    missing = []
    for path in REQUIRED_PATHS:
        val = get_path(entry, path)
        if val is None:
            missing.append(".".join(path))
    if missing:
        print("❌ Missing fields:")
        for m in missing:
            print("  -", m)
        raise SystemExit(1)

    # Extra sanity checks (should be non-empty)
    dense_n = len(entry["retrieval_trace"]["dense_hits"])
    kw_n = len(entry["retrieval_trace"]["keyword_hits"])
    fused_n = len(entry["retrieval_trace"]["fused_results_pre_rerank"])
    chunks_n = len(entry["final_context"]["chunks_sent"])
    ctx_len = len(entry["final_context"]["context_text"] or "")

    print("✅ Required fields present.")
    print(f"   dense_hits: {dense_n}")
    print(f"   keyword_hits: {kw_n}")
    print(f"   fused_results_pre_rerank: {fused_n}")
    print(f"   final_context.chunks_sent: {chunks_n}")
    print(f"   final_context.context_text length: {ctx_len}")

    if dense_n == 0 or kw_n == 0 or fused_n == 0 or chunks_n == 0 or ctx_len == 0:
        print("⚠️ Fields exist but one of them is empty. This may happen if retrieval returned nothing.")
    else:
        print("✅ Looks good: pipeline logging is working.")

if __name__ == "__main__":
    main()
