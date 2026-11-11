"""Smoke test: LiquidAI/d1-2B (LFM3 decision model) — what it does and doesn't do.

FINDING (run 2026-11-09, French receipt): d1-2B's text/number/date extraction reads a
span **from the text `state`**, not from an image. It is NOT an image OCR extractor.
On an image it does `box` localization (and choice/noul/score). So three demos below:

  A. image-only typed extraction -> found 0.0 everywhere (nothing to read: no text state)
  B. extraction over receipt TEXT   -> works: value + span + unit
  C. box localization over the IMAGE -> works: boxes for the total / date regions

No free-text generation; one forward pass via `system_one`. First run downloads ~4 GB.
Run: `uv run smoke_d1.py` (or cell-by-cell in the Interactive Window). Needs transformers>=5.19.
"""
import json

import torch
from transformers import AutoModel
from transformers.image_utils import load_image

# %% load once (keeps d1-2B warm across re-runs of the demo cells)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModel.from_pretrained(
    "LiquidAI/d1-2B", trust_remote_code=True, dtype=dtype
).to(device)

image = load_image("ticket_caisse.jpeg")
GT = {"magasin": "Intermarché", "date": "2026-01-05", "heure": "13:06:09", "total": 27.49}


def show(title, answers):
    print(f"\n=== {title} ===")
    print(json.dumps(answers, indent=1, ensure_ascii=False, default=str))


# %% A — typed extraction straight off the IMAGE (shows the limitation: found 0.0)
extract_q = {
    "magasin": {"type": "text", "instructions": "Nom du magasin / enseigne du ticket."},
    "date": {"type": "date", "instructions": "Date d'achat figurant sur le ticket."},
    "total": {"type": "number", "instructions": "Montant total payé (TTC), en euros."},
}
show("A. image-only extraction (expect found 0.0 — not an image OCR)",
     model.system_one(None, extract_q, media=[image])["answers"])

# %% B — same extraction over the receipt as TEXT (this is what the Read types are for)
state = ("INTERMARCHE MONTPELLIER\nDATE 05/01/2026 13:06:09\nARTICLES ...\n"
         "TOTAL A PAYER 27,49 EUR\nTEL 04 67 06 19 00")
show("B. extraction over TEXT state (works: value + span + unit)",
     model.system_one(state, extract_q)["answers"])

# %% C — box localization on the IMAGE (works: the image path d1 is built for)
box_q = {
    "total_box": {"type": "box", "instructions": "Localise le montant total sur le ticket."},
    "date_box": {"type": "box", "instructions": "Localise la date sur le ticket."},
}
show("C. box localization over IMAGE (works: boxes + confidence)",
     model.system_one(None, box_q, media=[image])["answers"])

print("\n=== ground truth ===")
print(json.dumps(GT, indent=1, ensure_ascii=False))
