# VLM_live

Receipt → JSON extraction with [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B),
comparing plain prompting against [`outlines`](https://github.com/dottxt-ai/outlines)
constrained decoding.

## Setup

Uses a uv-managed virtualenv (Python 3.13). Key deps: `transformers` 5.x, `torch`,
`outlines`, `accelerate`.

```bash
uv sync
uv run main.py                  # single end-to-end generation
```

`uv sync` creates the `.venv` and installs the exact versions pinned in `uv.lock`
(it replaces a manual `pip install` of the `pyproject.toml` dependencies). On a Linux
x86_64 machine with an NVIDIA driver this is enough to get a CUDA-enabled `torch`, since
the PyPI `torch` wheels ship bundled CUDA. Caveats: the model weights are downloaded from
the Hugging Face Hub on first run (not vendored in the repo), and a non-Linux/non-CUDA
machine will resolve to a different `torch` build.

For interactive work, open `compare_outlines.py` in the VS Code Interactive Window
and run it cell by cell (`# %%` markers) on the `.venv` kernel — the model loads once
and stays in GPU memory across cells.

## Files

- `main.py` — minimal single-image generation.
- `compare_outlines.py` — the experiment: baseline vs. `outlines`, across the base and
  `-Extract` models, with sampling and a strict Pydantic schema. Models load one at a
  time (8 GiB GPU can't hold both).
- `ticket_caisse.jpeg` — test receipt (Intermarché Montpellier; ground truth: date
  `05/01/2026`, time `13:06:09`, total `27.49 €`).

## Experiment: prompting vs. constrained decoding

### Journey

- **v1 — greedy, loose schema:** both methods valid JSON, no difference (greedy is deterministic).
- **v2 — sampling (temp 0.8) + strict date regex `\d{2}/\d{2}/\d{4}`:** baseline 3/5 schema-valid,
  outlines 5/5 — but outlines produced absurd dates (`50/01/2026`, year `0950`).
- **v3 — calendar regex + temp 0.4:** baseline 5/5 (small sample), outlines 5/5 with valid-looking dates.
- **v4 — n=20:** baseline 18/20, outlines 20/20. Found both misreading the receipt's **time**
  `13:06:09` as the date.
- **v5 — year forced to `20\d{2}`:** killed the absurd years, but the time-hallucination *adapted*
  into `13/06/2023` — now indistinguishable from a real date.
- **v6 — added `-Extract` model:** baseline **0/20** schema-valid (confidently misreads time as date,
  adds a `currency` field); outlines 20/20.
- **v7 — added dedicated `time` field:** `time` now correctly captures `13:06:09`, but Extract
  *duplicated* it into `date` too — the dedicated field didn't stop the conflation.

### Final numbers (n=20, temp 0.4)

| Model            | Method         | Valid JSON | Valid schema |
|------------------|----------------|------------|--------------|
| VL-1.6B base     | without outlines | 20/20    | 18/20        |
| VL-1.6B base     | with outlines    | 20/20    | **20/20**    |
| VL-1.6B-Extract  | without outlines | 16/20    | **0/20**     |
| VL-1.6B-Extract  | with outlines    | 19/20    | **19/20**    |

### Conclusions

- **`outlines` guarantees structure, never correctness** — 100% parseable/schema-valid in every
  config; baseline plateaus around 90%.
- **Constraints reshape errors, they don't remove them** — each tighter regex made hallucinations
  rarer but *more plausible-looking*, and so harder to catch.
- **Validation-as-detector vs. constraint-as-mask** — a baseline schema failure is useful signal
  ("the model misread"); `outlines` silences that signal by reformatting the error into compliance.
- **"Good at JSON" (Extract) means low variance, not accuracy** — it reliably emits clean JSON in
  its own shape, but its consistency turned a systematic date misread into a 100% failure.
- **Schema design routes and formats perception; it can't fix it** — the `time` field captured the
  time correctly but couldn't make any model read the true date. The remaining levers are on the
  *input*: prompt disambiguation, a tighter image crop, or a bigger model.
- **Base model > Extract on this receipt** — only the base model ever read the real date
  (`05/01/2026`); Extract was confidently, consistently wrong.
