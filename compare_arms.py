# Paired exact McNemar tests between result arms (same 361 receipts), from the saved JSONL files.
# Prints (only-first right, only-second right, p) per field. Edit A / the pair list to add arms.
#   uv run compare_arms.py     (shop clustering makes the p-values optimistic)
import json
from sroie import COLUMNS
from stats import mcnemar
def latest(path, sub, method=None):
    out = {}
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        if sub in r["model"] and ("Extract" in sub or "Extract" not in r["model"]) and "error" not in r and (method is None or r.get("method") == method):
            out[r["key"]] = r
    return out
A = {
 "vLLM free B": latest("results_vllm_guided_v2.jsonl", "vLLM", "free"),
 "vLLM guided B": latest("results_vllm_guided_v2.jsonl", "vLLM", "guided-xgrammar"),
 "ollama format B": latest("results_ollama_format_v2.jsonl", "ollama"),
 "HF base B": latest("results_outlines_v3.jsonl", "LFM2.5-VL-1.6B", "baseline"),
 "HF outlines B": latest("results_outlines_v3.jsonl", "LFM2.5-VL-1.6B", "outlines"),
 "HF base inline": latest("results_outlines_v2.jsonl", "LFM2.5-VL-1.6B", "baseline"),
 "HF outlines inline": latest("results_outlines_v2.jsonl", "LFM2.5-VL-1.6B", "outlines"),
 "vLLM free inline": latest("results_vllm_guided_v1.jsonl", "vLLM", "free"),
 "BAML vLLM": latest("results_baml_v2.jsonl", "vLLM"),
 "BAML ollama": latest("results_baml_v2.jsonl", "ollama"),
 "HF Extract base B": latest("results_outlines_v3.jsonl", "Extract", "baseline"),
 "HF Extract outlines B": latest("results_outlines_v3.jsonl", "Extract", "outlines"),
 "HF Extract native base": latest("results_outlines_v1.jsonl", "Extract#native", "baseline"),
 "HF Extract native outlines": latest("results_outlines_v1.jsonl", "Extract#native", "outlines"),
}
for n,a in A.items(): print(n, len(a))
for x,y in [("vLLM guided B","vLLM free B"),("vLLM free B","HF base B"),("vLLM guided B","HF outlines B"),
            ("ollama format B","vLLM guided B"),("ollama format B","HF outlines B"),
            ("HF base B","HF base inline"),("HF outlines B","HF outlines inline"),("vLLM free B","vLLM free inline"),
            ("BAML vLLM","vLLM free B"),("BAML vLLM","vLLM guided B"),("BAML vLLM","HF outlines B"),
            ("BAML ollama","ollama format B"),("BAML ollama","BAML vLLM"),
            ("HF Extract outlines B","HF Extract base B"),("HF Extract outlines B","HF outlines B"),
            ("HF Extract native base","HF Extract base B"),("HF Extract native outlines","HF Extract outlines B"),
            ("HF Extract native outlines","HF outlines B")]:
    ks = sorted(A[x].keys() & A[y].keys()); print(f"\n{x} vs {y} (n={len(ks)}) [only-first, only-second, p]")
    for c in COLUMNS:
        print(f"  {c:<13}", mcnemar([A[x][k]["ok"][c] for k in ks],[A[y][k]["ok"][c] for k in ks]))
