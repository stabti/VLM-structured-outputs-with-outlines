# Resolution-study figure from results_resolution_v1.jsonl: accuracy vs long side (left), image tokens (right).
#   uv run --with matplotlib plot_resolution.py   -> resolution_study.png
import json

import matplotlib.pyplot as plt

LEVELS = ["384", "512", "768", "1024", "1536", "2048", "native"]
FIELDS = [("company", "company", "#2a78d6", "o", "-"), ("date", "date (strict)", "#eb6834", "s", "-"),
          ("address", "address", "#1baf7a", "^", "-"), ("total", "total", "#e87ba4", "D", "-"),
          ("date_lenient", "date (lenient)", "#6250d6", "v", "--")]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

rows = {}
for line in open("results_resolution_v1.jsonl"):
    if line.strip():
        r = json.loads(line)
        if "error" not in r:
            rows.setdefault(r["method"].removeprefix("free-B-res"), {})[r["key"]] = r
acc = {f: [] for f, *_ in FIELDS}
tok = []
for lv in LEVELS:
    rs = list(rows[lv].values())
    assert len(rs) == 361, (lv, len(rs))
    for f, *_ in FIELDS:
        v = [r["ok"][f] for r in rs if r["ok"][f] is not None]
        acc[f].append(100 * sum(v) / len(v))
    tok.append(sum(r["input_tokens"] for r in rs) / len(rs))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]}, facecolor=SURFACE)
x = range(len(LEVELS))
for f, label, c, m, ls in FIELDS:
    ax.plot(x, acc[f], color=c, marker=m, ls=ls, lw=2, ms=7, mec=SURFACE, mew=1.5, label=label)
    if f.startswith("date"):  # the three ~90% lines end on top of each other: legend only for those
        ax.annotate(label, (x[-1], acc[f][-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axvspan(3.6, 6.4, color="#cde2fb", alpha=0.35, lw=0)
ax.text(5, 4, "same 1954 tokens\n(processor tile cap)", ha="center", fontsize=9, color=MUTED)
ax.set(xticks=list(x), xticklabels=LEVELS, ylim=(0, 100), xlim=(-0.3, 7.3), ylabel="accuracy (%), 361 receipts",
       xlabel="long side of the image sent (px)")
bx.bar(list(x), tok, color="#2a78d6", width=0.7)
for i, t in enumerate(tok):
    bx.text(i, t + 30, f"{t:.0f}", ha="center", fontsize=9, color=INK)
bx.set(xticks=list(x), xticklabels=LEVELS, ylim=(0, 2250), ylabel="mean image + prompt tokens", xlabel="long side (px)")
ax.set_title("Accuracy vs resolution (LFM2.5-VL-1.6B, vLLM, free, prompt B)", fontsize=11, loc="left")
bx.set_title("Tokens seen by the model", fontsize=11, loc="left")
for a in (ax, bx):
    a.set_facecolor(SURFACE)
    a.grid(axis="y", color="#e4e3df", lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(colors=MUTED)
ax.legend(loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.62, 0.1))
fig.tight_layout()
fig.savefig("resolution_study.png", dpi=160, facecolor=SURFACE)
print("saved resolution_study.png", [round(t) for t in tok])
