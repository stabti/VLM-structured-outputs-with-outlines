# BAML arm of the structured-output comparison.
# Paradigm: Schema-Aligned Parsing (Family B) — the model (LFM2.5-VL via ollama,
# BF16 GGUF) generates freely, BAML coerces the output to the Receipt shape.
# Contrast with compare_outlines.py (Family A: token-level masking, HF + outlines).
#
# We judge BAML's output with the SAME yardstick as the outlines run: the strict
# Pydantic schema (date/time regexes) + field accuracy against the receipt's ground
# truth. Run cell by cell in the VS Code Interactive Window (# %% markers).

import base64
import time

from baml_client import b
from baml_py import Image
from pydantic import BaseModel, Field, ValidationError

# %% Setup: encode the image, ground truth, strict schema (run once)
with open("ticket_caisse.jpeg", "rb") as f:
    receipt_img = Image.from_base64("image/jpeg", base64.b64encode(f.read()).decode())

# Ground truth read off the receipt (Intermarché Montpellier)
GT = {"store": "intermarch", "date": "05/01/2026", "time": "13:06:09", "total": 27.49}

N_RUNS = 20


class Item(BaseModel):
    name: str
    price: float


class StrictReceipt(BaseModel):
    """Same strict schema used to judge the outlines run — the fair yardstick."""

    store_name: str
    date: str | None = Field(
        default=None, pattern=r"^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/20\d{2}$"
    )
    time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$")
    items: list[Item]
    total_ttc: float


def score(data: dict | None) -> dict:
    """data: BAML Receipt as a dict, or None if BAML raised."""
    r = {
        "baml_ok": data is not None,
        "strict_schema": False,
        "store_ok": False,
        "date_ok": False,
        "time_ok": False,
        "total_ok": False,
        "date": None,
    }
    if data is None:
        return r
    try:
        StrictReceipt.model_validate(data)
        r["strict_schema"] = True
    except ValidationError:
        pass  # SAP still produced values; we score their accuracy below
    r["date"] = data.get("date")
    r["store_ok"] = GT["store"] in (data.get("store_name") or "").lower()
    r["date_ok"] = data.get("date") == GT["date"]
    r["time_ok"] = data.get("time") == GT["time"]
    tot = data.get("total_ttc")
    r["total_ok"] = tot is not None and abs(tot - GT["total"]) < 0.01
    return r


# %% Run BAML N_RUNS times (free generation + schema-aligned parsing)
results = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    try:
        out = b.ExtractReceipt(receipt_img=receipt_img)
        data = out.model_dump()
    except Exception as e:  # parse failure / connection — counts as a miss
        data = None
        print(f"  [run {i + 1}] EXCEPTION: {type(e).__name__}: {e}")
    elapsed = time.perf_counter() - t0
    r = score(data)
    r["time"] = elapsed
    results.append(r)
    print(
        f"[baml {i + 1}/{N_RUNS}] {elapsed:.1f}s  "
        f"obj {'✅' if r['baml_ok'] else '❌'}  strict {'✅' if r['strict_schema'] else '❌'}  "
        f"date={r['date']}"
    )


# %% Summary + how it lines up against the outlines run
def rate(key: str) -> str:
    n = sum(r[key] for r in results)
    return f"{n}/{len(results)}"


avg_time = sum(r["time"] for r in results) / len(results)
print("=" * 70)
print(f"BAML (SAP) sur LFM2.5-VL-1.6B BF16 via ollama — {N_RUNS} runs, {avg_time:.1f}s/run")
print("=" * 70)
print(f"  Objet produit (SAP)      : {rate('baml_ok')}")
print(f"  Schéma strict respecté   : {rate('strict_schema')}   (regex date/heure)")
print("  --- exactitude vs vérité terrain ---")
print(f"  Magasin (Intermarché)    : {rate('store_ok')}")
print(f"  Date (05/01/2026)        : {rate('date_ok')}")
print(f"  Heure (13:06:09)         : {rate('time_ok')}")
print(f"  Total TTC (27.49)        : {rate('total_ok')}")
print(f"\n  Dates générées : {[r['date'] for r in results]}")
print(
    "\nLecture : BAML produit presque toujours un objet (c'est le but du SAP), "
    "\nmais — comme outlines — il ne corrige pas la PERCEPTION : surveiller date_ok."
)
