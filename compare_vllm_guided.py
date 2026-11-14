# vLLM guided-decoding arm: the SAME model (LFM2.5-VL-1.6B, bf16) served by vLLM, once free
# (prompt only) and once with the JSON schema enforced by vLLM's structured-output backend.
# Why: the HF+outlines arm (Family A) confounds "token masking" with the HF stack, and the
# ollama-vs-vLLM BAML runs showed the engine changes results. This arm gives, on ONE engine:
#   free vs guided (clean enforcement effect), guided vs BAML-on-vLLM (Family A vs B), and
#   guided vs HF+outlines (same masking idea, other engine).
# Same prompt/schema/decoding as compare_outlines.py (English, greedy, max 256 tokens);
# image sent as lossless PNG (same pixels as the HF arm). Resumes from the JSONL.
# Needs a vLLM server (see CLAUDE.md); BACKEND only labels which structured-output backend
# the server was started with (`--structured-outputs-config.backend xgrammar|outlines|guidance`).
#   BACKEND=xgrammar uv run compare_vllm_guided.py   (FULL=1 for all 361 receipts)

import base64
import io
import json
import os
import time
from itertools import islice

import requests
from PIL import Image
from pydantic import BaseModel, ValidationError

from energy import gpu_energy_j
from sroie import COLUMNS, FIELDS, iter_samples, load_split, score_fields
from stats import fmt, mcnemar

# %% Config (keep identical to compare_outlines.py: prompt, schema, decoding)
SPLIT = "test"
N_IMAGES = None if os.environ.get("FULL") else 5
BACKEND = os.environ.get("BACKEND", "xgrammar")
# ENGINE=ollama: native /api/chat with `format` = the Pydantic JSON schema (what ollama.chat(...,
# format=Receipt.model_json_schema()) sends), enforced by a llama.cpp grammar. Guided only (the
# prompt-only reference there is ollama+BAML v2); ~25 s/receipt => ~2.5 h for 361.
ENGINE = os.environ.get("ENGINE", "vllm")  # or: ENGINE=ollama uv run compare_vllm_guided.py
if ENGINE == "ollama":
    URL = "http://localhost:11434/api/chat"
    SERVED = "hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16"
    MODEL = "LFM2.5-VL-1.6B (ollama BF16 GGUF)"  # same label as compare_baml.py's ollama records
    METHODS = ("guided-ollama",)
    RESULTS_PREFIX, MODEL_SUB = "results_ollama_format", "ollama"
    assert os.environ.get("PROMPT_VARIANT", "B") == "B", "ollama arm uses prompt B only"
else:
    URL = "http://localhost:8000/v1/chat/completions"
    SERVED = "LiquidAI/LFM2.5-VL-1.6B"
    MODEL = "LFM2.5-VL-1.6B (vLLM bf16)"  # same label as compare_baml.py's vLLM records
    METHODS = ("free", f"guided-{BACKEND}")  # "free" does not depend on the backend: run once
    RESULTS_PREFIX, MODEL_SUB = "results_vllm_guided", "vLLM"
# PROMPT_VARIANT (vLLM, free mode only): why does BAML's rendered prompt beat ours? A = BAML's exact
# prompt, B = our wording + BAML's commented schema block, C = our wording/inline schema in BAML's
# order (instruction, image, schema). Default "ours" = the common prompt of every other arm.
# Default is now B (v2 files); A/C are the legacy v1 experiments, "ours" reproduces the v1 prompt.
VARIANT = os.environ.get("PROMPT_VARIANT", "B")
if VARIANT in ("A", "C"):
    assert ENGINE == "vllm", "variants A/C: vLLM only"
    METHODS = (f"free-{VARIANT}",)
# RES=<px>|native (vLLM, free, prompt B): resolution study. Long side is downscaled to <px> (never upscaled)
# BEFORE sending; "native" = untouched (the processor still tiles/caps it). Own results file.
RES = os.environ.get("RES")
if RES:
    assert ENGINE == "vllm" and VARIANT == "B", "RES: vLLM + prompt B only"
    METHODS = (f"free-B-res{RES}",)
    RESULTS_PREFIX = "results_resolution"
GUIDED = METHODS[-1]
TAG = "v1" if RES else "v2" if VARIANT == "B" else "v3"  # v1 of free-A/B/C ran guided by mistake: truly free reruns go to v3
RESULTS = f"{RESULTS_PREFIX}_{TAG}.jsonl"
KEYS = load_split(SPLIT)["key"][:N_IMAGES]


class Receipt(BaseModel):
    company: str
    date: str
    address: str
    total: float


schema_desc = (
    '{"company": string (name of the store), '
    '"date": string (purchase date as printed), '
    '"address": string (full address of the store), '
    '"total": number (total amount paid, without currency symbol)}'
)
prompt = (
    "Extract the information from this receipt. "
    "Reply ONLY with valid JSON following exactly this schema, "
    "with no additional text and no markdown:\n" + schema_desc
)


# BAML's rendered schema block (copied from the request BAML builds for baml_src/receipt.baml)
BAML_SCHEMA = (
    "Answer in JSON using this schema:\n{\n  // name of the store\n  company: string,\n"
    "  // purchase date as printed\n  date: string,\n  // full address of the store\n  address: string,\n"
    "  // total amount paid, without currency symbol\n  total: float,\n}"
)
OUR_HEAD = prompt.split("\n")[0]  # "Extract ... receipt. Reply ONLY with valid JSON ... no markdown:"


PROMPT_B = OUR_HEAD + "\n" + BAML_SCHEMA.split("\n", 1)[1]  # prompt B text


def content_parts(image_url: str) -> list[dict]:
    img, txt = {"type": "image_url", "image_url": {"url": image_url}}, lambda t: {"type": "text", "text": t}
    return {
        "ours": [img, txt(prompt)],
        "A": [txt("Extract the information from this receipt."), img, txt(BAML_SCHEMA)],
        "B": [img, txt(PROMPT_B)],  # our head + BAML's block only
        "C": [txt("Extract the information from this receipt."), img,
              txt("Reply ONLY with valid JSON following exactly this schema, with no additional text "
                  "and no markdown:\n" + schema_desc)],
    }[VARIANT]


def lenient_json(text: str) -> dict:
    i, j = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[i : j + 1]) if 0 <= i < j else {}
    except json.JSONDecodeError:
        return {}


def evaluate(raw_text: str, gt: dict) -> dict:
    """Same two axes as compare_outlines.py: strict structure on the raw text, accuracy on the payload."""
    result = {"raw": raw_text, "valid_json": False, "valid_schema": False, "pred": {}}
    try:
        data = json.loads(raw_text)
        result["valid_json"] = True
        try:
            Receipt.model_validate(data)
            result["valid_schema"] = True
        except ValidationError:
            pass
    except json.JSONDecodeError:
        data = lenient_json(raw_text)
    if isinstance(data, dict):
        result["pred"] = {f: data.get(f) for f in FIELDS}
    result["ok"] = score_fields(result["pred"], gt)
    return result


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
    with open(RESULTS, "a") as f:  # leading "\n": a torn last line never fuses with the next record
        f.write("\n" + json.dumps(rec, ensure_ascii=False, default=str))


def png_data_url(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate(image, guided: bool) -> tuple[str, int, int]:
    """(text, input_tokens, output_tokens) from one greedy chat completion."""
    if RES and RES != "native" and max(image.size) > int(RES):
        r = int(RES) / max(image.size)
        image = image.resize((max(1, round(image.width * r)), max(1, round(image.height * r))), Image.LANCZOS)
    if ENGINE == "ollama":  # native /api/chat, exactly what ollama.chat(..., format=schema) sends
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        r = requests.post(URL, json={
            "model": SERVED, "stream": False,
            "messages": [{"role": "user", "content": PROMPT_B,
                          "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "format": Receipt.model_json_schema(),
            "options": {"temperature": 0, "num_predict": 256},  # num_ctx left at ollama's default, as in the BAML arm
        }, timeout=600)
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"], j.get("prompt_eval_count", -1), j.get("eval_count", -1)
    body = {
        "model": SERVED,
        "messages": [{"role": "user", "content": content_parts(png_data_url(image))}],
        "temperature": 0,
        "max_tokens": 256,
    }
    if guided:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "Receipt", "schema": Receipt.model_json_schema(), "strict": True}}
    r = requests.post(URL, json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"] or "", j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]


# %% Run (resumes from RESULTS; server/connection failures are recorded as "error" and retried on rerun)
done = {(r["method"], r["key"]) for r in load_results() if r["model"] == MODEL and "error" not in r}
for n, (key, image, gt) in enumerate(islice(iter_samples(SPLIT), N_IMAGES), 1):
    for method in METHODS:
        if (method, key) in done:
            continue
        e0, t0 = gpu_energy_j(), time.perf_counter()
        error, text, n_in, n_out = None, "", None, None
        try:
            text, n_in, n_out = generate(image, guided=method.startswith("guided"))
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:400]}"
        elapsed, energy_j = time.perf_counter() - t0, gpu_energy_j() - e0
        r = evaluate(text, gt)
        r.update(model=MODEL, method=method, key=key, time=elapsed, gt=gt, energy_j=energy_j,
                 input_tokens=n_in, output_tokens=n_out)
        if error:
            r["error"] = error
        save_result(r)
        ok = "".join("✅" if r["ok"][c] else "·" if r["ok"][c] is None else "❌" for c in COLUMNS)
        print(f"[{method} {n} {key}] {elapsed:.1f}s  JSON {'✅' if r['valid_json'] else '❌'}  {ok}"
              + (f"  ERREUR {error[:120]}" if error else ""))


# %% Summary + paired comparisons (counts with Wilson 95% CI; McNemar on the same receipts)
def latest(path, model_sub, method=None) -> dict:
    """key -> last non-error record of a results file (model label contains model_sub)."""
    out = {}
    try:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model_sub in r["model"] and "error" not in r and (method is None or r.get("method") == method):
                out[r["key"]] = r
    except FileNotFoundError:
        pass
    return {k: v for k, v in out.items() if k in set(KEYS)}


arms = {m: latest(RESULTS, MODEL_SUB, m) for m in sorted({r["method"] for r in load_results()})}
arms[f"BAML ({MODEL_SUB}, v2)"] = latest("results_baml_v2.jsonl", MODEL_SUB)
arms["HF + outlines (v2)"] = latest("results_outlines_v2.jsonl", "LFM2.5-VL-1.6B", "outlines")
arms["HF baseline (v2)"] = latest("results_outlines_v2.jsonl", "LFM2.5-VL-1.6B", "baseline")
arms = {m: a for m, a in arms.items() if a}
print("=" * 78)
print(f"vLLM guidé — {SPLIT}, {len(KEYS)} images, glouton, même modèle (LFM2.5-VL-1.6B bf16)")
print("=" * 78)
for m, a in arms.items():
    rs = list(a.values())
    n = len(rs)
    print(f"\n{m}  (n={n}, {sum(r['time'] for r in rs)/n:.1f}s, {sum(r['energy_j'] for r in rs)/n:.0f} J GPU/image)")
    if "valid_json" in rs[0]:
        print(f"  JSON valide {fmt(sum(r['valid_json'] for r in rs), n)}   schéma {fmt(sum(r['valid_schema'] for r in rs), n)}")
    for c in COLUMNS:
        s = [r["ok"][c] for r in rs if r["ok"][c] is not None]
        print(f"  {'date (tolérant)' if c == 'date_lenient' else c:<16}: {fmt(sum(s), len(s))}")
print("\nComparaisons appariées (A seul juste / B seul juste, p) :")
for a_name, b_name in [(GUIDED, "free"), (GUIDED, "HF + outlines (v2)"),
                       (GUIDED, f"BAML ({MODEL_SUB}, v2)")]:
    if a_name in arms and b_name in arms:
        ks = sorted(arms[a_name].keys() & arms[b_name].keys())
        print(f"  {a_name} vs {b_name}:")
        for c in COLUMNS:
            print(f"    {c:<13}", mcnemar([arms[a_name][k]["ok"][c] for k in ks], [arms[b_name][k]["ok"][c] for k in ks]))
