# BAML arm of the structured-output comparison, on SROIE receipts.
# Paradigm: Schema-Aligned Parsing (Family B) — the model (LFM2.5-VL via ollama,
# BF16 GGUF) generates freely, BAML coerces the output to the Receipt shape.
# Contrast with compare_outlines.py (Family A: token-level masking, HF + outlines).
#
# We judge BAML's output with the SAME yardstick as the outlines run: the same Pydantic
# schema + per-column correctness vs the SROIE ground truth (sroie.score_fields), one
# greedy generation per image, Wilson CIs (stats.py). Every result is appended to a JSONL
# file as it is produced and a rerun resumes. Needs ollama running with the model pulled
# and `uv run baml-cli generate` done (set BAML_LOG=WARN to silence per-call logging).
# The original single-ticket experiment (n=20, temp 0.4) is in git history / README.
# Run cell by cell in the VS Code Interactive Window (# %% markers).

import base64
import io
import json
import os
import time
from itertools import islice

from baml_client import b
from baml_py import Collector, Image
from pydantic import BaseModel, ValidationError

from energy import gpu_energy_j
from sroie import COLUMNS, FIELDS, iter_samples, load_split, score_fields
from stats import fmt

# %% Config (keep identical to compare_outlines.py)
SPLIT = "test"  # 361 receipts
N_IMAGES = None if os.environ.get("FULL") else 5  # smoke test; FULL=1 = the whole split (~7 min vLLM, ~80 min ollama)
# Same BAML function, two inference engines: ollama (llama.cpp, GGUF) or vLLM (separate
# env, `vllm serve LiquidAI/LFM2.5-VL-1.6B ...`, see baml_src/clients.baml).
ENGINE = os.environ.get("ENGINE", "ollama")  # or: ENGINE=vllm uv run compare_baml.py
CLIENT, MODEL = {
    "ollama": ("CustomOllama", "LFM2.5-VL-1.6B (ollama BF16 GGUF)"),
    "vllm": ("CustomVllm", "LFM2.5-VL-1.6B (vLLM bf16)"),
}[ENGINE]
TAG = "v2"  # v1 = French prompt/descriptions, v2 = English (2026-11-10). Bump when the prompt / schema / client settings change (new results file)
RESULTS = f"results_baml_{TAG}.jsonl"
KEYS = load_split(SPLIT)["key"][:N_IMAGES]  # column access only: no image decoding


class Receipt(BaseModel):
    """Same schema used to judge the outlines run — the fair yardstick."""

    company: str
    date: str
    address: str
    total: float


def to_baml_image(pil_image) -> Image:
    """Lossless PNG, so both arms see the same pixels (re-encoding to JPEG would not)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return Image.from_base64("image/png", base64.b64encode(buf.getvalue()).decode())


def score(data: dict | None, gt: dict) -> dict:
    """data: BAML Receipt as a dict, or None if BAML raised (no value extracted = all wrong).
    ok[c] is True/False, or None when the GT is missing (excluded from the denominator)."""
    r = {"baml_ok": data is not None, "valid_schema": False, "pred": {}}
    if data is not None:
        try:
            Receipt.model_validate(data)
            r["valid_schema"] = True
        except ValidationError:
            pass  # SAP still produced values; we score their accuracy below
        r["pred"] = {f: data.get(f) for f in FIELDS}
    r["ok"] = score_fields(r["pred"], gt)
    return r


def load_results() -> list[dict]:
    try:
        with open(RESULTS) as f:
            rows = []
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn last line after a crash
            return rows
    except FileNotFoundError:
        return []


def save_result(rec: dict) -> None:
    with open(RESULTS, "a") as f:  # leading "\n": a torn last line (crash mid-write) never fuses with the next record
        f.write("\n" + json.dumps(rec, ensure_ascii=False, default=str))


# %% Run BAML once per image (free generation + schema-aligned parsing), resuming from RESULTS
collector = Collector(name="sroie")
# Receipts that failed on the infrastructure side ("error") are retried on a rerun.
done = {r["key"] for r in load_results() if r["model"] == MODEL and "error" not in r}
for n, (key, image, gt) in enumerate(islice(iter_samples(SPLIT), N_IMAGES), 1):
    if key in done:
        continue
    e0, t0 = gpu_energy_j(), time.perf_counter()
    error = parse_error = None
    try:
        data = b.ExtractReceipt(
            receipt_img=to_baml_image(image), baml_options={"client": CLIENT, "collector": collector}
        ).model_dump()
    except Exception as e:  # the run stays alive either way
        data, msg = None, f"{type(e).__name__}: {str(e)[:400]}"
        if "Validation" in type(e).__name__:
            parse_error = msg  # the model's text could not be coerced: a real failure, scored as wrong
        else:
            error = msg  # server / connection problem: not the model's fault, retried and reported apart
    elapsed = time.perf_counter() - t0
    energy_j = gpu_energy_j() - e0  # whole-GPU energy while the server worked on this call
    last = collector.last  # what the model actually said, BEFORE SAP coerced it (also on parse failures)
    usage = last.usage if last else None
    r = score(data, gt)
    r.update(
        model=MODEL, key=key, time=elapsed, gt=gt, energy_j=energy_j,
        raw=last.raw_llm_response if last else None,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
    )
    collector.clear()
    if error:
        r["error"] = error
    if parse_error:
        r["parse_error"] = parse_error
    save_result(r)
    ok = "".join("✅" if r["ok"][c] else "·" if r["ok"][c] is None else "❌" for c in COLUMNS)
    print(f"[baml {n} {key}] {elapsed:.1f}s  obj {'✅' if r['baml_ok'] else '❌'}  {ok}" + (f"  ERREUR {error[:160]}" if error else ""))


# %% Summary (counts with Wilson 95% CI)
latest = {}  # a retried receipt appears twice in the file (error, then result): the last one wins
for r in load_results():
    if r["model"] == MODEL and r["key"] in set(KEYS):
        latest[r["key"]] = r
failed = sum("error" in r for r in latest.values())  # infrastructure failures: reported apart, not scored
results = [r for r in latest.values() if "error" not in r]
n = len(results)
avg_time = sum(r["time"] for r in results) / n
avg_energy = sum(r["energy_j"] for r in results) / n
print("=" * 70)
print(f"BAML (SAP) sur {MODEL} — {SPLIT}, {n} images évaluées, {failed} échecs serveur exclus")
print(f"{avg_time:.1f}s et {avg_energy:.0f} J GPU /image")
print("=" * 70)
print(f"  Objet produit (SAP)      : {fmt(sum(r['baml_ok'] for r in results), n)}")
print(f"  Schéma respecté          : {fmt(sum(r['valid_schema'] for r in results), n)}")
print("  --- exactitude vs vérité terrain ---")
for c in COLUMNS:
    scored = [r["ok"][c] for r in results if r["ok"][c] is not None]
    print(f"  {'date (tolérant)' if c == 'date_lenient' else c:<24} : {fmt(sum(scored), len(scored))}")
print(
    "\nLecture : BAML produit presque toujours un objet (c'est le but du SAP), "
    "\nmais — comme outlines — il ne corrige pas la PERCEPTION : surveiller date et total."
)
