# Compare small-VLM structured output on SROIE receipts: plain prompting vs
# outlines-constrained generation, across four models — LFM2.5-VL-1.6B base and its
# -Extract variant (tuned for JSON extraction), Qwen3-VL-2B and SmolVLM2-2.2B.
# Scaled design: the unit of analysis is the IMAGE (one greedy generation per receipt
# per method), scored field by field vs the SROIE ground truth (sroie.score_fields:
# normalized date/total, fuzzy company/address, strict + lenient date), with Wilson CIs
# and a paired McNemar test between methods (stats.py). The original single-ticket
# experiment (n=20, temp 0.4, French Intermarché receipt) is in git history / README.
# Every result is appended to a JSONL file the moment it exists and a rerun resumes
# where it stopped (a crash loses at most the generation in flight).
# Models are loaded one at a time (8 GiB GPU can't hold two) and freed between.
# Run cell by cell in the VS Code Interactive Window (Shift+Enter on each # %% cell).

import gc
import json
import os
import time
from itertools import islice

import outlines
import torch
from outlines.inputs import Chat, Image
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForImageTextToText, AutoProcessor

from energy import gpu_energy_j
from sroie import COLUMNS, FIELDS, iter_samples, load_split, score_fields
from stats import fmt, mcnemar

# %% Config: models to compare, data, generation settings (run once)
MODEL_IDS = [
    "LiquidAI/LFM2.5-VL-1.6B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    # Extract is not in the v2 (English) run: its native English prompt was run in v1 as
    # "LiquidAI/LFM2.5-VL-1.6B-Extract#native" (system-turn YAML field list, image-only user turn);
    # re-add that label here to rerun it.
]

SPLIT = "test"  # 361 receipts
N_IMAGES = None if os.environ.get("FULL") else 5  # smoke test; FULL=1 = the whole split (~45 min, 4 models)
# Greedy, one generation per image: the image is the statistical unit. Sampling noise is
# studied separately (small subset, repeated runs), not mixed into the main table.
GEN_KWARGS = dict(max_new_tokens=256, do_sample=False)
METHODS = ("baseline", "outlines")
# Qwen3-VL's processor has no token cap (LFM's stops at 2856): the 55 A4 scans (32-35 Mpx)
# OOM on 8 GiB. 9 Mpx leaves every other receipt (max 8.7 Mpx) untouched and only shrinks those scans.
QWEN_MAX_PIXELS = 9_000_000

# Bump TAG when the prompt / schema / GEN_KWARGS change: results go to a new file and are
# never mixed with (or resumed from) an older protocol.
TAG = "v3"  # v1 = French prompt, v2 = English inline-schema prompt (both kept), v3 = prompt B (commented schema block)
RESULTS = f"results_outlines_{TAG}.jsonl"
KEYS = load_split(SPLIT)["key"][:N_IMAGES]  # column access only: no image decoding


# %% Target structure (Pydantic, = SROIE ground-truth keys) and the shared prompt
class Receipt(BaseModel):
    # Loose on value format on purpose: normalization (dates, currency...) lives in the
    # scorer, otherwise we'd measure format compliance instead of reading.
    company: str
    date: str
    address: str
    # A number, not a string: the models naturally emit 193.0, and forcing `str` made
    # outlines mask them into junk (', ') on the -Extract model — a typing artifact.
    total: float


# Same hints as baml_src/receipt.baml (@description) — keep both in sync.
schema_desc = (
    '{"company": string (name of the store), '
    '"date": string (purchase date as printed), '
    '"address": string (full address of the store), '
    '"total": number (total amount paid, without currency symbol)}'
)
# Prompt B (best of the vLLM prompt-variant test): our instruction + BAML's commented schema block.
# schema_desc (inline schema, v2) is kept for reference only.
prompt = (
    "Extract the information from this receipt. "
    "Reply ONLY with valid JSON following exactly this schema, with no additional text and no markdown:\n"
    "{\n  // name of the store\n  company: string,\n  // purchase date as printed\n  date: string,\n"
    "  // full address of the store\n  address: string,\n"
    "  // total amount paid, without currency symbol\n  total: float,\n}"
)


# Extract's native format (model card): YAML "field: description" list in the system prompt.
NATIVE_SYSTEM = (
    "Extract the following from the image:\n"
    "company: Name of the store or brand\n"
    "date: Purchase date as printed on the receipt\n"
    "address: Full address of the store\n"
    "total: Total amount paid, as a number without currency symbol\n\n"
    "Respond with only a JSON object. Do not include any text outside the JSON."
)


def build_messages(image, native: bool) -> list[dict]:
    if native:
        return [
            {"role": "system", "content": NATIVE_SYSTEM},
            {"role": "user", "content": [{"type": "image", "image": image}]},
        ]
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def lenient_json(text: str) -> dict:
    """Payload of a fenced / chatty reply ('```json {...} ```'), or {}. Used for the ACCURACY
    axis only, so format noise doesn't zero a correct read; valid_json stays strict."""
    i, j = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[i : j + 1]) if 0 <= i < j else {}
    except json.JSONDecodeError:
        return {}


def evaluate(raw_text: str, gt: dict) -> dict:
    """Structure (JSON + schema, strict on the raw text) AND per-column correctness vs GT.
    ok[c] is True/False, or None when the GT is missing (excluded from the denominator).
    No parsable value = wrong on every field."""
    result = {"raw": raw_text, "valid_json": False, "valid_schema": False, "pred": {}}
    try:
        data = json.loads(raw_text)
        result["valid_json"] = True
        try:
            Receipt.model_validate(data)
            result["valid_schema"] = True
        except ValidationError as e:
            result["schema_error"] = str(e)
    except json.JSONDecodeError as e:
        result["json_error"] = str(e)
        data = lenient_json(raw_text)
    if isinstance(data, dict):
        result["pred"] = {f: data.get(f) for f in FIELDS}
    result["ok"] = score_fields(result["pred"], gt)
    return result


def status(r: dict) -> str:
    ok = "".join("✅" if r["ok"][c] else "·" if r["ok"][c] is None else "❌" for c in COLUMNS)
    return f"JSON {'✅' if r['valid_json'] else '❌'}  schéma {'✅' if r['valid_schema'] else '❌'}  {ok}"


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


def run_model(model_id: str) -> None:
    """Load a model, run baseline + outlines once per image (skipping what is already in
    RESULTS), free the GPU."""
    hf_id, _, variant = model_id.partition("#")  # model_id is the stored label; hf_id is what we load
    native = variant == "native"
    short = model_id.split("/")[-1]
    # generations that crashed ("error": e.g. CUDA OOM) are retried on a rerun
    done = {(r["model"], r["method"], r["key"]) for r in load_results() if "error" not in r}
    if all((model_id, m, k) in done for m in METHODS for k in KEYS):
        print(f"{short}: déjà fait ({RESULTS})")
        return
    print(f"\n########## {model_id} ##########")
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id, device_map="auto", dtype="bfloat16"
    )
    processor = AutoProcessor.from_pretrained(hf_id)
    if "Qwen" in model_id:
        processor.image_processor.size["longest_edge"] = QWEN_MAX_PIXELS
    ol_model = outlines.from_transformers(model, processor)

    def chat_inputs(image):  # prompt + image as the model's tokens (also gives the input-token count)
        return processor.apply_chat_template(
            build_messages(image, native),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def baseline_text(image) -> str:  # plain generate(), JSON requested via prompt only
        inputs = chat_inputs(image).to(model.device)
        outputs = model.generate(**inputs, **GEN_KWARGS)
        return processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def outlines_text(image) -> str:  # same prompt/decoding, constrained by the schema
        # (Image(image) needs image.format — the HF dataset's JPEG receipts keep it)
        return ol_model(Chat(build_messages(Image(image), native)), Receipt, **GEN_KWARGS)

    generate = {"baseline": baseline_text, "outlines": outlines_text}
    # iter_samples decodes one receipt at a time — never hold the whole split in memory
    for n, (key, image, gt) in enumerate(islice(iter_samples(SPLIT), N_IMAGES), 1):
        n_in = None  # prompt+image length: same chat template for both methods, computed once
        for method in METHODS:
            if (model_id, method, key) in done:
                continue
            if n_in is None:
                try:
                    n_in = chat_inputs(image)["input_ids"].shape[1]
                except Exception:  # a processor failure resurfaces as the generation's own error below
                    n_in = -1
            e0, t0 = gpu_energy_j(), time.perf_counter()
            error = None
            try:
                text = generate[method](image)
            except Exception as e:  # e.g. CUDA OOM on a huge receipt: record it, keep the run alive
                torch.cuda.empty_cache()
                text, error = "", f"{type(e).__name__}: {str(e)[:400]}"
            elapsed = time.perf_counter() - t0
            energy_j = gpu_energy_j() - e0
            r = evaluate(text, gt)
            r.update(
                model=model_id, method=method, key=key, time=elapsed, gt=gt, energy_j=energy_j,
                input_tokens=n_in,  # output_tokens = re-tokenized `raw` (no EOS): a close approximation
                output_tokens=len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]),
            )
            if error:
                r["error"] = error
            save_result(r)
            print(f"[{short} {method} {n} {key}] {elapsed:.1f}s  {status(r)}" + (f"  ERREUR {error}" if error else ""))

    # free GPU before loading the next model
    del ol_model, model, processor
    gc.collect()
    torch.cuda.empty_cache()


# %% Run the experiment for every model (loads one at a time; resumes from RESULTS)
for model_id in MODEL_IDS:
    run_model(model_id)


# %% Comparison across models and methods (counts with Wilson 95% CI)
rows = [r for r in load_results() if r["key"] in set(KEYS)]


def pick(model_id: str, method: str) -> dict:
    return {r["key"]: r for r in rows if r["model"] == model_id and r["method"] == method}


def label(c: str) -> str:
    return "date (tolérant)" if c == "date_lenient" else c


def summarize(name: str, res: dict) -> None:
    rs = [r for r in res.values() if "error" not in r]
    errors = len(res) - len(rs)  # crashed generations (e.g. OOM): reported apart, not scored as wrong reads
    n = len(rs)
    if not rs:
        print(f"  {name:<28} aucune image évaluée ({errors} erreurs)")
        return
    avg_time = sum(r["time"] for r in rs) / n
    avg_energy = sum(r["energy_j"] for r in rs) / n
    print(
        f"  {name:<28} valide: JSON {fmt(sum(r['valid_json'] for r in rs), n)}"
        f"  schéma {fmt(sum(r['valid_schema'] for r in rs), n)}   ({avg_time:.1f}s, {avg_energy:.0f} J GPU/image"
        + (f", {errors} erreurs exclues" if errors else "")
        + ")"
    )
    for c in COLUMNS:
        scored = [r["ok"][c] for r in rs if r["ok"][c] is not None]
        print(f"  {'':<28} exact {label(c):<15}: {fmt(sum(scored), len(scored))}")


print("=" * 78)
print(f"RÉSULTATS — {SPLIT}, {len(KEYS)} images, glouton")
print("=" * 78)
for model_id in MODEL_IDS:
    base, ol = pick(model_id, "baseline"), pick(model_id, "outlines")
    if not base or not ol:
        continue
    print(f"\n{model_id}")
    summarize("sans outlines (prompt seul)", base)
    summarize("avec outlines (contraint)", ol)
    # same images, two methods -> paired test per column
    common = sorted(k for k in base.keys() & ol.keys() if "error" not in base[k] and "error" not in ol[k])
    for c in COLUMNS:
        only_ol, only_base, p = mcnemar([ol[k]["ok"][c] for k in common], [base[k]["ok"][c] for k in common])
        print(f"  McNemar {label(c):<15}: juste seulement avec outlines {only_ol}, seulement sans {only_base}, p={p:.3f}")

print("\nRappel : outlines garantit la STRUCTURE, pas l'exactitude des valeurs.")
