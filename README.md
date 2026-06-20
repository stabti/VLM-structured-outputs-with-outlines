# Enforcing structured outputs from VLMs

Receipt → JSON extraction with [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B),
exploring **how to enforce structured outputs from a vision model across different stacks**.

There are really only two families of enforcement, distinguished by *where* it happens:

- **Family A — enforce in the decoder (token masking).** The engine masks logits so only
  schema-valid tokens can be sampled. Hard guarantee; lives inside the engine.
  Here: HF + [`outlines`](https://github.com/dottxt-ai/outlines). (Same idea powers vLLM
  guided decoding, llama.cpp GBNF grammars, and ollama's `format`.)
- **Family B — enforce at the client (parse / coerce).** The model generates freely, a
  client library coerces the output to the schema afterward. No hard guarantee, but
  provider-agnostic and often more accurate. Here: [BAML](https://github.com/BoundaryML/baml)
  (Schema-Aligned Parsing). (Same family as `instructor`.)

The headline finding: **neither family fixes *perception*** — they only decide what happens
to a value the model already misread.

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

### BAML arm (Family B)

The BAML arm talks to LFM2.5-VL served locally by **ollama**, over its OpenAI-compatible
endpoint — so BAML needs no torch/transformers and no `ollama` Python package, just HTTP.

```bash
ollama pull hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16   # BF16 = no quantization loss
uv run baml-cli generate                              # (re)generate baml_client/ from baml_src/
uv run compare_baml.py
```

`baml-cli` ships inside the `baml-py` package (no separate install). You only ever need
two commands: `baml-cli init` (done once — created `baml_src/`) and `baml-cli generate`
(re-run after editing any `.baml` file; it rewrites the generated `baml_client/`, which is
gitignored).

## Files

- `main.py` — minimal single-image generation.
- `compare_outlines.py` — Family A: baseline vs. `outlines`, across the base and
  `-Extract` models, with sampling and a strict Pydantic schema. Models load one at a
  time (8 GiB GPU can't hold both).
- `compare_baml.py` — Family B: BAML (Schema-Aligned Parsing) on LFM2.5-VL via ollama,
  scored against the *same* strict schema and ground truth as the outlines run.
- `baml_src/` — BAML source: the `Receipt` schema, the `ExtractReceipt` function, and the
  ollama client config. `baml_client/` is generated from this and gitignored.
- `ticket_caisse.jpeg` — test receipt (Intermarché Montpellier; ground truth: date
  `05/01/2026`, time `13:06:09`, total `27.49 €`).

## Experiment: outlines / constrained decoding (Family A)

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

- **`outlines` guarantees structure, never correctness** — schema-valid by construction (the lone
  19/20 was output truncated at the token cap, not a constraint failure); the unconstrained
  baseline plateaus around 90%.
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

## Experiment: BAML / Schema-Aligned Parsing (Family B)

Same model, same receipt, same strict schema and ground truth — but the model now runs as a
**BF16 GGUF served by ollama** (vs. bf16 HF weights for the outlines arm), and enforcement is
**client-side coercion** instead of token masking.

### Numbers (n=20, temp 0.4)

| Metric                         | Result    |
|--------------------------------|-----------|
| Object produced (SAP)          | **20/20** |
| Strict schema valid (date/time regex) | **4/20** |
| Store = Intermarché            | 20/20     |
| Total = 27.49                  | 19/20     |
| Time = 13:06:09                | 15/20     |
| **Date = 05/01/2026**          | **0/20**  |

### What it shows

- **SAP never fails to produce an object (20/20)** — it even coerced pure garbage like
  `"04 06 19 00"` into the `Receipt` rather than throwing. That robustness is the whole point
  of Family B.
- **But "object produced" ≠ "valid by the schema" (only 4/20 strict)** — with no token masking,
  malformed dates flow straight into the object as strings. The exact inverse of `outlines`,
  which is 20/20 valid *by construction*.
- **BAML is more *diagnostic* precisely because it doesn't mask** — it leaves `04/06/19` visibly
  malformed, where the model *appears* to have misread the receipt's phone number
  **`TEL 04 67 06 19 00`** as the date. Token masking would have laundered that into a plausible
  fake. (And the 4 regex-passing dates were all `06/05/2026` — well-formed and still wrong.)
- **The engine swap made date perception *worse* (0/20)** — same model, same BF16 precision, so
  this is the **llama.cpp vision pipeline**, not quantization and not BAML. The easy fields
  (store, total) stayed robust; only the hardest glyph degraded.

## Bottom lines

- **`outlines` (Family A, mask):** turns "will it parse?" from a probability into a certainty —
  100% structurally valid. The cost: it *hides* errors, reshaping a misread value into a
  well-formed, plausible-looking fake. Tighter schemas make fakes rarer **and harder to spot**.
- **BAML (Family B, parse):** turns "did we get an object?" into a near-certainty (SAP coerces
  almost anything) — but it does *not* lie about format, so malformed reads stay **visible and
  diagnosable**. Lower strict-validity is a feature here, not a bug: it's the model's failure
  shown honestly rather than masked.
- **Perception is the real bottleneck, and neither family fixes it.** Every method nailed the
  easy fields (store, total) and failed the hard one (the date) — outlines by hiding the error,
  BAML by surfacing it. The levers that *would* help are upstream of enforcement: prompt
  disambiguation, a tighter/higher-res image crop, a better vision frontend, or a bigger model.
  Structured-output tooling decides *what happens to a wrong value*; it never makes the value
  right.
