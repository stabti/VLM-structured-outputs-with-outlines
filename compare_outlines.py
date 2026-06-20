# Compare LFM2.5-VL structured output: plain prompting vs outlines-constrained generation,
# across two models — the base VL model and the -Extract variant tuned for JSON extraction.
# Experiment: sampling (do_sample=True) + strict calendar date pattern, N_RUNS per cell.
# Models are loaded one at a time (8 GiB GPU can't hold both) and freed between.
# Run cell by cell in the VS Code Interactive Window (Shift+Enter on each # %% cell).

import gc
import json
import time

import outlines
import torch
from outlines.inputs import Chat, Image
from PIL import Image as PILImage
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForImageTextToText, AutoProcessor

# %% Config: models to compare, image, generation settings (run once)
MODEL_IDS = [
    "LiquidAI/LFM2.5-VL-1.6B",
    "LiquidAI/LFM2.5-VL-1.6B-Extract",
]

# PILImage.open keeps image.format ("JPEG"), required by outlines.inputs.Image
image = PILImage.open("ticket_caisse.jpeg")
image.load()

N_RUNS = 20
GEN_KWARGS = dict(max_new_tokens=256, do_sample=True, temperature=0.4)

# Ground truth read off the receipt (Intermarché Montpellier) — same as compare_baml.py
GT = {"store": "intermarch", "date": "05/01/2026", "time": "13:06:09", "total": 27.49}


# %% Target structure (Pydantic) and the shared prompt
class Item(BaseModel):
    name: str
    price: float


class Receipt(BaseModel):
    store_name: str
    # Strict format AND plausible values: day<=31, month<=12, year 20xx
    date: str | None = Field(
        default=None, pattern=r"^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/20\d{2}$"
    )
    # Dedicated time field: the model kept stuffing the receipt's time (13:06:09)
    # into `date`. Giving it a home should stop the conflation.
    time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$")
    items: list[Item]
    total_ttc: float


schema_desc = (
    '{"store_name": string, "date": string au format JJ/MM/AAAA ou null, '
    '"time": string au format HH:MM:SS ou null, '
    '"items": [{"name": string, "price": number}], "total_ttc": number}'
)
prompt = (
    "Extrais les informations de ce ticket de caisse. "
    "Réponds UNIQUEMENT avec un JSON valide suivant exactement ce schéma, "
    "sans texte supplémentaire et sans markdown :\n" + schema_desc
)


def evaluate(raw_text: str) -> dict:
    """Validity (JSON + strict schema) AND field-level accuracy vs ground truth."""
    result = {
        "raw": raw_text, "valid_json": False, "valid_schema": False, "parsed": None,
        "store_ok": False, "date_ok": False, "time_ok": False, "total_ok": False,
        "date_val": None,
    }
    try:
        data = json.loads(raw_text)
        result["valid_json"] = True
    except json.JSONDecodeError as e:
        result["json_error"] = str(e)
        return result
    try:
        result["parsed"] = Receipt.model_validate(data)
        result["valid_schema"] = True
    except ValidationError as e:
        result["schema_error"] = str(e)
    # Accuracy is scored from the raw JSON values, independent of strict-schema validity
    if isinstance(data, dict):
        result["date_val"] = data.get("date")
        result["store_ok"] = GT["store"] in (data.get("store_name") or "").lower()
        result["date_ok"] = data.get("date") == GT["date"]
        result["time_ok"] = data.get("time") == GT["time"]
        tot = data.get("total_ttc")
        result["total_ok"] = isinstance(tot, (int, float)) and abs(tot - GT["total"]) < 0.01
    return result


def status(r: dict) -> str:
    return f"JSON {'✅' if r['valid_json'] else '❌'}  schéma {'✅' if r['valid_schema'] else '❌'}"


def run_model(model_id: str) -> dict:
    """Load a model, run baseline + outlines N_RUNS times each, free the GPU, return results."""
    print(f"\n########## {model_id} ##########")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map="auto", dtype="bfloat16"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # --- baseline: plain generate() with sampling, JSON requested via prompt only ---
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    baseline = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        outputs = model.generate(**inputs, **GEN_KWARGS)
        elapsed = time.perf_counter() - t0
        text = processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        r = evaluate(text)
        r["time"] = elapsed
        baseline.append(r)
        print(f"[{model_id.split('/')[-1]} baseline {i + 1}/{N_RUNS}] {elapsed:.1f}s  {status(r)}")

    # --- outlines: same model, same prompt/sampling, decoding constrained by the schema ---
    ol_model = outlines.from_transformers(model, processor)
    chat_input = Chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image(image)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
    )
    constrained = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        text = ol_model(chat_input, Receipt, **GEN_KWARGS)
        elapsed = time.perf_counter() - t0
        r = evaluate(text)
        r["time"] = elapsed
        constrained.append(r)
        print(f"[{model_id.split('/')[-1]} outlines {i + 1}/{N_RUNS}] {elapsed:.1f}s  {status(r)}")

    # free GPU before loading the next model
    del ol_model, model, processor, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return {"baseline": baseline, "outlines": constrained}


# %% Run the experiment for every model (loads one at a time)
all_results = {model_id: run_model(model_id) for model_id in MODEL_IDS}


# %% Comparison across models and methods
def summarize(name: str, results: list[dict]) -> None:
    n = len(results)
    g = lambda k: sum(r[k] for r in results)  # noqa: E731
    avg_time = sum(r["time"] for r in results) / n
    print(f"  {name:<28} valide: JSON {g('valid_json'):>2}/{n}  schéma {g('valid_schema'):>2}/{n}")
    print(
        f"  {'':<28} exact: magasin {g('store_ok'):>2}/{n}  date {g('date_ok'):>2}/{n}  "
        f"heure {g('time_ok'):>2}/{n}  total {g('total_ok'):>2}/{n}   ({avg_time:.1f}s/run)"
    )


print("=" * 78)
print(f"RÉSULTATS — {N_RUNS} runs, temp {GEN_KWARGS['temperature']}")
print("=" * 78)
for model_id, res in all_results.items():
    print(f"\n{model_id}")
    summarize("sans outlines (prompt seul)", res["baseline"])
    summarize("avec outlines (contraint)", res["outlines"])

print("\n" + "=" * 78)
print("Champs date / time extraits (le time devrait absorber le 13:06:09)")
print("=" * 78)
for model_id, res in all_results.items():
    short = model_id.split("/")[-1]
    for method in ("baseline", "outlines"):
        pairs = [
            f"{r['parsed'].date}|{r['parsed'].time}" if r["parsed"] else "—"
            for r in res[method]
        ]
        print(f"  {short} / {method}: {pairs}")

print("\nRappel : outlines garantit la STRUCTURE, pas l'exactitude des valeurs —")
print("le variant -Extract devrait surtout aider la baseline (sans contrainte).")
