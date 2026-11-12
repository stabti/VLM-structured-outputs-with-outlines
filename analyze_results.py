# Extra analyses on the saved results (no model calls): receipt-level accuracy, shop-clustered
# bootstrap CIs (receipts of one shop are not independent), and an error taxonomy.
#   uv run analyze_results.py
import json
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein

from sroie import FIELDS, norm_date, norm_text, norm_total
from stats import fmt

RNG = np.random.default_rng(0)
N_BOOT = 2000


def latest(path, sub, method=None):
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if sub in r["model"] and ("Extract" in sub or "Extract" not in r["model"]) and "error" not in r and (method is None or r.get("method") == method):
            out[r["key"]] = r
    return out


ARMS = {  # prompt B (or BAML's own equivalent block); LFM2.5-VL-1.6B unless stated
    "HF free": latest("results_outlines_v3.jsonl", "LFM2.5-VL-1.6B", "baseline"),
    "HF outlines": latest("results_outlines_v3.jsonl", "LFM2.5-VL-1.6B", "outlines"),
    "vLLM free": latest("results_vllm_guided_v2.jsonl", "vLLM", "free"),
    "vLLM guided": latest("results_vllm_guided_v2.jsonl", "vLLM", "guided-xgrammar"),
    "ollama format": latest("results_ollama_format_v2.jsonl", "ollama"),
    "BAML vLLM": latest("results_baml_v2.jsonl", "vLLM"),
    "BAML ollama": latest("results_baml_v2.jsonl", "ollama"),
    "Qwen free": latest("results_outlines_v3.jsonl", "Qwen3-VL", "baseline"),
    "Qwen outlines": latest("results_outlines_v3.jsonl", "Qwen3-VL", "outlines"),
}
ARMS = {k: v for k, v in ARMS.items() if v}
print({k: len(v) for k, v in ARMS.items()})
KEYS = sorted(set.intersection(*(set(a) for a in ARMS.values())))
GT = {k: next(iter(ARMS.values()))[k]["gt"] for k in KEYS}
SHOP = {k: norm_text(GT[k]["company"]) for k in KEYS}  # cluster id = normalized GT company
shops = sorted(set(SHOP.values()))
print(f"{len(KEYS)} receipts, {len(shops)} distinct shops (cluster = GT company)\n")


def all_ok(r, lenient):
    cols = [("date_lenient" if f == "date" and lenient else f) for f in FIELDS]
    v = [r["ok"][c] for c in cols if r["ok"][c] is not None]
    return all(v)


def boot(values_by_key):
    """Cluster bootstrap (resample shops) 95% CI of the mean of a per-receipt 0/1 value."""
    by_shop = defaultdict(list)
    for k, v in values_by_key.items():
        by_shop[SHOP[k]].append(v)
    groups = [np.array(by_shop[s], float) for s in shops]
    sums, ns = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = RNG.integers(0, len(groups), (N_BOOT, len(groups)))
    est = sums[idx].sum(1) / ns[idx].sum(1)
    return sum(sums) / sum(ns), *np.percentile(est, [2.5, 97.5])


# 1) receipt-level accuracy -------------------------------------------------------------
print("RECEIPT-LEVEL accuracy (all 4 fields right at once): Wilson CI | shop-clustered bootstrap CI")
for name, a in ARMS.items():
    for lenient in (False, True):
        v = {k: all_ok(a[k], lenient) for k in KEYS}
        p, lo, hi = boot(v)
        print(f"  {name:<14}{'date lenient' if lenient else 'date strict ':<13}{fmt(sum(v.values()), len(KEYS))}   | boot {p:.0%} [{lo:.0%}-{hi:.0%}]")

# 2) clustered CIs per field + clustered paired differences -----------------------------
print("\nPER-FIELD accuracy with shop-clustered bootstrap CI (vs Wilson in the other scripts)")
for name, a in ARMS.items():
    cells = []
    for c in (*FIELDS, "date_lenient"):
        v = {k: a[k]["ok"][c] for k in KEYS if a[k]["ok"][c] is not None}
        p, lo, hi = boot(v)
        cells.append(f"{c[:7]} {p:.0%}[{lo:.0%}-{hi:.0%}]")
    print(f"  {name:<14}" + "  ".join(cells))

PAIRS = [("vLLM guided", "vLLM free"), ("HF outlines", "HF free"), ("Qwen outlines", "Qwen free"),
         ("vLLM free", "HF free"), ("ollama format", "vLLM guided"), ("BAML vLLM", "vLLM free"),
         ("BAML ollama", "ollama format")]
print("\nPAIRED difference (first minus second, points) with shop-clustered bootstrap CI; * = CI excludes 0")
for x, y in PAIRS:
    if x not in ARMS or y not in ARMS:
        continue
    cells = []
    for c in (*FIELDS, "date_lenient"):
        ks = [k for k in KEYS if ARMS[x][k]["ok"][c] is not None]
        d = {k: int(ARMS[x][k]["ok"][c]) - int(ARMS[y][k]["ok"][c]) for k in ks}
        by_shop = defaultdict(list)
        for k, v in d.items():
            by_shop[SHOP[k]].append(v)
        groups = [np.array(by_shop[s], float) for s in shops if s in by_shop]
        sums, ns = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
        idx = RNG.integers(0, len(groups), (N_BOOT, len(groups)))
        est = 100 * sums[idx].sum(1) / ns[idx].sum(1)
        lo, hi = np.percentile(est, [2.5, 97.5])
        cells.append(f"{c[:7]} {100 * sums.sum() / ns.sum():+.1f}[{lo:+.1f},{hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")
    print(f"  {x} − {y}:  " + "  ".join(cells))


# 3) error taxonomy ----------------------------------------------------------------------
def kind(field, pred, gt):
    if pred is None or str(pred).strip() == "":
        return "missing"
    if field == "date":
        return "wrong/other date"  # (strict fail with lenient date right is split out by the caller)
    if field == "total":
        p, g = norm_total(pred), norm_total(gt)
        if p is None:
            return "unparseable"
        return "wrong value (right order of magnitude)" if g and 0.1 < p / g < 10 else "wrong value (far off)"
    p, g = norm_text(pred), norm_text(gt)
    if g in p:
        return "contains GT + extra text"
    if p and p in g:
        return "truncated (part of GT)"
    s = Levenshtein.normalized_similarity(p, g)
    return "near miss (0.6-0.9 similar)" if s >= 0.6 else "different text (<0.6)"


print("\nERROR TAXONOMY of strict failures (GT present), per arm and field")
for name in ("vLLM free", "vLLM guided", "ollama format", "BAML vLLM"):
    if name not in ARMS:
        continue
    print(f"  {name}")
    for f in FIELDS:
        bad = [k for k in KEYS if ARMS[name][k]["ok"][f] is False]
        if f == "date":  # strict-only failures split by whether the lenient date is right
            c = Counter("right date + extra text (time…)" if ARMS[name][k]["ok"]["date_lenient"]
                        else kind(f, ARMS[name][k]["pred"].get(f), GT[k][f]) for k in bad)
        else:
            c = Counter(kind(f, ARMS[name][k]["pred"].get(f), GT[k][f]) for k in bad)
        print(f"    {f:<8}{len(bad):>4} fails: " + "; ".join(f"{t} {n}" for t, n in c.most_common()))
