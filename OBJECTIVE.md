# Objective: Enforcing structured outputs from small VLMs (receipt → JSON)

## Goal
Answer one question with a controlled experiment: **when you force a small,
open-weights VLM to emit structured JSON, does the enforcement buy you
*structure*, or *correctness*?** Concretely — extract typed fields (store, date,
time, total, line items) from a receipt image with LFM2.5-VL, survey the ways to
enforce a schema across stacks, and measure on an identical testbed what each
approach actually delivers. "Done" = a fair head-to-head across enforcement
families on one receipt with a clear, defensible conclusion.

**Why it matters (framing for the talk):** when the model *has* to be small and
open-weight — private data stays on-device, no per-call cost, offline, full
control — structured-output reliability stops being a convenience and becomes the
whole game. Small models drift on format *and* misread content; the study asks
whether enforcement tooling makes the output trustworthy or only well-shaped.

## Brainstorm / decisions
- **Two enforcement families, split by *where* enforcement happens** — this is the
  organizing idea of the whole study:
  - **Family A — mask in the decoder (token masking).** The engine masks logits so
    only schema-valid tokens can be sampled. Hard guarantee, lives inside the
    engine. Studied here via **HF + `outlines`**. (Same mechanism as vLLM guided
    decoding, llama.cpp GBNF, ollama `format`.)
  - **Family B — parse at the client (coerce).** The model generates freely; a
    client library coerces the output to the schema afterward. No hard guarantee,
    but provider-agnostic and honest. Studied here via **BAML (Schema-Aligned
    Parsing)**. (Same family as `instructor`.)
- **Fair-comparison discipline:** identical receipt, identical strict Pydantic
  schema (date/time regexes), identical ground truth, n=20, temperature 0.4 across
  every arm. BAML is judged with the *same* yardstick as outlines even though it
  doesn't mask.
- **Two axes of measurement, deliberately separated:** (1) structural validity —
  does it parse and pass the strict schema; (2) field accuracy vs. ground truth —
  is each value actually right. Keeping these apart is the point.
- **Model choice:** compare the base `LFM2.5-VL-1.6B` against the `-Extract`
  variant (tuned for JSON extraction) to test whether "good at JSON" means
  accurate. Models load one at a time (8 GiB GPU can't hold both).
- **Engine choice per arm (a confound made explicit):** outlines runs on bf16 **HF**
  weights; BAML talks to a **BF16 GGUF served by ollama** (llama.cpp) over its
  OpenAI-compatible HTTP endpoint. BF16 both sides, so a perception delta between
  arms is the *vision pipeline*, not quantization.

## Experiments / benchmarks
Testbed: one receipt (`ticket_caisse.jpeg`, Intermarché Montpellier). Ground
truth — store `Intermarché`, date `05/01/2026`, time `13:06:09`, total `27.49 €`.
n=20, temp 0.4.

**Family A — outlines / constrained decoding.** Iterated across 7 versions
("seven turns of the screw"): greedy loose schema → sampling + strict date regex →
calendar regex → n=20 → year forced `20\d{2}` → added the `-Extract` model → added
a dedicated `time` field. Each tighter constraint made hallucinations rarer but
*more plausible-looking*.

Structural validity:

| Model            | Method           | Valid JSON | Valid schema |
|------------------|------------------|------------|--------------|
| VL-1.6B base     | without outlines | 20/20      | 19/20        |
| VL-1.6B base     | with outlines    | 20/20      | **20/20**    |
| VL-1.6B-Extract  | without outlines | 13/20      | **0/20**     |
| VL-1.6B-Extract  | with outlines    | 20/20      | **20/20**    |

Field accuracy vs. ground truth:

| Model            | Method           | Store | Total | Time  | Date     |
|------------------|------------------|-------|-------|-------|----------|
| VL-1.6B base     | without outlines | 20/20 | 19/20 | 1/20  | 3/20     |
| VL-1.6B base     | with outlines    | 20/20 | 20/20 | 4/20  | **5/20** |
| VL-1.6B-Extract  | without outlines | 13/20 | 11/20 | 6/20  | **0/20** |
| VL-1.6B-Extract  | with outlines    | 20/20 | 19/20 | 20/20 | **0/20** |

Structure goes to 20/20 by construction; the date is right at most 5/20.
Structure says nothing about correctness. Base model was the *only* one to ever
read the true date; Extract was confidently, consistently wrong.

**Family B — BAML / Schema-Aligned Parsing** (same model, BF16 GGUF via ollama):

| Metric                                  | Result    |
|-----------------------------------------|-----------|
| Object produced (SAP)                   | **20/20** |
| Strict schema valid (date/time regex)   | **4/20**  |
| Store = Intermarché                     | 20/20     |
| Total = 27.49                           | 19/20     |
| Time = 13:06:09                         | 15/20     |
| **Date = 05/01/2026**                   | **0/20**  |

SAP always produces an object (it even coerced garbage like `"04 06 19 00"` rather
than throwing) but leaves malformed dates *visible* — the model appears to misread
the receipt's phone number `TEL 04 67 06 19 00` as the date. The exact inverse of
outlines: honest about format instead of masking. Date perception was *worse* here
(0/20) than the HF/outlines arm — same model, same BF16 — pinning the degradation
on the llama.cpp vision pipeline, not quantization or BAML.

## Code
Repo: **`~/Documents/VLM_live`** (public: `github.com/stabti/VLM-structured-outputs-with-outlines`).
uv-managed venv, Python 3.13; key deps `transformers` 5.x, `torch`, `outlines`,
`accelerate`, `baml-py`.
- `main.py` — minimal single-image generation (smoke test).
- `compare_outlines.py` — Family A: baseline vs. outlines, base vs. `-Extract`,
  strict Pydantic `Receipt` schema, structural + field-accuracy scoring. Run cell
  by cell in the VS Code Interactive Window (`# %%`).
- `compare_baml.py` — Family B: BAML SAP on LFM2.5-VL via ollama, scored against
  the *same* strict schema + ground truth.
- `baml_src/` — BAML source (`Receipt` schema, `ExtractReceipt` function,
  `CustomOllama` client → `hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16`). `baml_client/`
  is generated (`uv run baml-cli generate`) and gitignored.
- `utils.py` — `draw_bboxes` helper (Qwen-VL-style box overlay; adjacent tooling).
- `ticket_caisse.jpeg` — the test receipt. `README.md` — full writeup.
- `slides.html` — the talk (intentionally **not** pushed).

## Conclusion
- **outlines (Family A, mask):** turns "will it parse?" into a certainty (100%
  structurally valid) — but *hides* errors, reshaping a misread value into a
  well-formed, plausible fake. Tighter schemas make fakes rarer **and harder to
  spot**. A baseline schema *failure* is useful signal ("the model misread");
  masking silences that signal.
- **BAML (Family B, parse):** turns "did we get an object?" into a near-certainty
  (SAP coerces almost anything) but does **not** lie about format — malformed reads
  stay visible and diagnosable. Lower strict-validity is a *feature*: the model's
  failure shown honestly rather than masked.
- **"Good at JSON" (Extract) means low variance, not accuracy** — it reliably emits
  clean JSON in its own shape, which turned a systematic date misread into a 100%
  failure. On this receipt the base model beat Extract.
- **Perception is the real bottleneck, and neither family fixes it.** Every method
  nailed the easy fields (store, total) and failed the hard glyph (the date) —
  outlines by hiding the error, BAML by surfacing it. **The one line:**
  structured-output tooling decides *what happens to a wrong value*; it never makes
  the value right. The levers that would help are upstream of enforcement — prompt
  disambiguation, a tighter/higher-res crop, a better vision frontend, or a bigger
  model.

## Remaining steps / actions
Directions to scale this from a demo into a study (from the closing slide):
1. **More images** — many receipts and document types; real accuracy with
   confidence intervals, not a single ticket.
2. **More lightweight VLMs** — Qwen-VL small, Ovis, SmolVLM… same testbed, head to
   head.
3. **A bigger VLM as judge (bonus)** — errors here are systematic, so only better
   *perception* helps; test whether a cost-aware cascade with a vision-capable
   verifier catches them. A vision verifier is the one lever that can touch the
   perception wall enforcement can't cross.

## Related / open questions
- Does the date miss survive a **tighter/higher-res crop** or an **image
  preprocessing** step (contrast, deskew), isolating perception from formatting?
- How much of the BAML-arm degradation is specifically the **llama.cpp vision
  encoder** vs. HF? Worth an ollama-vs-HF same-precision probe on the raw (no
  enforcement) reads.
- Would **prompt disambiguation** (explicitly telling the model the phone number
  and time are not the date) move the date accuracy at all, or is it a hard
  perception limit at 1.6B?
- Where does a **vLLM guided-decoding** arm land vs. HF+outlines — same Family A,
  different engine?
- Repo: `github.com/stabti/VLM-structured-outputs-with-outlines`.
  Model: [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B).
  Tools: [outlines](https://github.com/dottxt-ai/outlines),
  [BAML](https://github.com/BoundaryML/baml).

---

## Scale-up plan — WIP (2026-11-08, for a ~20-min Generation AI conference talk)

Planning session in progress. Goal: turn the n=1-receipt demo into a statistically
defensible study + add newer small VLMs. **User stepped away mid-plan; several
decisions still open (see below). Nothing built yet.**

### Locked so far
- **Dataset → SROIE** (public receipt KIE benchmark, 361 test imgs, field GT for
  `company/date/total`). Matches the current schema; citable; zero annotation.
  CORD optional later for line-items. **No `time` label in SROIE** → the time/
  phone-number confusion finding stays a vignette on our own `ticket_caisse.jpeg`
  (which has time GT), not a column in the scaled table.
- **Statistical design:** unit of analysis becomes the **image, not the
  generation**. Main accuracy table = run each method **once per image (greedy)**
  over the full 361 test → Wilson CI ≈ ±0.05. Compare two methods with **McNemar's
  paired test** (same images), not overlapping CIs. Keep a *separate* small
  sampling-variance probe: n≈10 generations on a 30-image subset. Preserves the
  two-axes discipline (structural validity vs field accuracy) now with CIs.
- **Enforcement families:** A (outlines / mask) and B (BAML / parse) retained.
- **Decision-model-as-OCR-judge → dropped from benchmarking.** Reasons:
  (1) circularity — to judge a date read, the judge must itself read the date, the
  same perception task in a smaller model; (2) "outputs a probability" ≠
  calibrated — would need its own ECE study. **Kept only as a spoken *perspective*
  with a *bigger* backbone** (user's call: don't exclude entirely, float as future
  work).
- **Object-detection decision-model arm = bonus/perspective slide only**, not a
  core track. It is a *different task* (localization, IoU/mAP) and — correcting an
  earlier overclaim — **SROIE boxes do NOT give it free GT**: SROIE boxes are
  *text-region* boxes (OCR localization), not classical object detection on natural
  images. Real object-detection demo needs its own COCO/RefCOCO slice = separate
  experiment. Too much for 20 min → mention as a conclusion/perspective.

### Model roster — LOCKED (4)
- LFM2.5-VL-1.6B **base** + **-Extract** — the *controlled* Liquid pair; retests
  "extraction-tuned = reliably wrong, not more accurate" at N=361 with a CI.
- Qwen3-VL-2B-Instruct (Dec 2025; strong OCR/doc parsing, native grounding, ~4 GB bf16).
- SmolVLM2-2.2B-Instruct (tiny baseline) — **stays** (the smoke test showed d1-2B
  can't replace it as an image extractor; see the d1-2B block).
- Florence-2-large: only if the detection bonus is revived (tied to that, not core).

### d1-2B (LiquidAI decision model, LFM3-VL-2B backbone) — candidate, SMOKE-TEST FIRST
`LiquidAI/d1-2B`. 1.92B, vision, **not a chat model — no token generation**, one
forward pass. Typed questions over state+image via
`model.system_one(state, questions, media=...)` (`AutoModel`, `trust_remote_code`,
`transformers>=5.19`). Returns structured answers, no free text. Types: yes/no→P,
choice, score, **text/number/date extraction (value + source SPAN + `found` flag;
dates ISO 8601)**, quantity w/ p10–p90 interval, box/boxes localization. ~4 GB bf16
→ fits 8 GiB. License LFM Open v1.0. Invoice example on the card, **no OCR/doc
benchmark** → our receipt test is novel signal.

**SMOKE-TEST RESULT (2026-11-09, `smoke_d1.py` on `ticket_caisse.jpeg`) — key
correction:** d1's `text/number/date` extraction reads a **span from the text
`state`, NOT from an image**. It is *not* an image OCR extractor.
- A. Typed extraction straight off the image → `found: 0.0` on every field (nothing
  to read — no text state). So **d1-2B canNOT replace SmolVLM as an image→JSON
  extractor.** SmolVLM stays in the roster.
- B. Same extraction over the receipt as **text** → works: `total=27.49` span
  `[76,85]` unit `EUR`, `date=05/01/2026`.
- C. **`box` localization over the image** → works: located the total region
  (`found 0.99`, conf 0.18) and the **date** region (`found 1.0`, conf 0.87).

*Revised role (what d1 actually unlocks here):*
1. **Localizer (`box`) → "localize-then-read" perception attack, now LOCAL.** d1 box
   → crop the field → a VLM reads the clean crop. This is the one lever that can
   touch the perception wall, and it runs in 8 GiB (I'd earlier said it needed the
   27B — wrong). Directly tests whether isolating the date glyph fixes the misread.
   Box also gives a confidence to study (date conf 0.87 vs total 0.18).
2. **Text-side extractor/judge** over another model's OCR dump: given OCR text +
   a candidate, extract/verify the span ("is `27,49` the total, and where?").
   Honest judge, but it judges *text*, so it inherits the upstream OCR's errors —
   it does not re-perceive the image.

*Not doing:* d1 as an image→fields extractor (demo A), or an image-reading judge.
STEP 0 done.

*Judge design (proposed — pending user confirm of Goal A framing).* Use case = the
practitioner takeaway: enforcement hides errors; a decision-model judge flags
low-confidence/likely-wrong fields for **human review** (agent pipelines,
pre-labeling). Mechanism: text-side `noul` over the model-under-test's output /
OCR text — "is the extracted <field> correct & coherent?" → P(yes); flag if
P<τ. (Variant: d1 re-extracts the field from the text; low `found` or a span landing
on the phone number = the flag.) **Start with ONE field (`date`)** as the proof it
catches the known masked error, then roll out to all four for a review-rate ×
errors-caught table. Honest limit to state: text-side judge inherits upstream OCR
errors — it does not re-perceive the image.
*Evaluate the judge itself vs GT* (we know each field's truth): 2×2 of {field
correct?} × {d1 flags?} → **error-catch recall** (of wrong fields, how many flagged),
**review precision** (of flagged, how many wrong), + **calibration** (ECE/reliability
of P). User's example "VLM wrong + d1 confident-correct" = a **false negative** (missed
error) — the costliest cell, the number that says if the judge is trustworthy.

*Goal A (resolved): sense (a), on text.* User's success metric = correct values in
the right JSON fields. Plan: feed d1 the SROIE **`words`** (the OCR transcription the
dataset ships) → d1 typed extraction → JSON, scored vs `entities`. Isolates
*structuring* ability. Honest caveat to state on stage: d1 gets text while the VLM
arms get the image → not apples-to-apples on perception; it answers "given OCR text,
does d1 fill the fields right?". Don't imply d1 reads the image.

### Measurements & side quests (2026-11-09)
- **Inference time (locked):** record per model × arm in the SROIE sweep (extend the
  per-run timing already in `compare_outlines.py`). A core result column.
- **Ecological impact (side quest, maybe not in talk):** wrap the *local* arms with
  **codecarbon** (GPU energy + gCO₂ per run) to compare framework sustainability
  (HF/outlines vs ollama/BAML vs d1). Tool honesty: **ecologits estimates *API*-model
  impact** → only if we add a closed-API baseline (ties to the agent idea: local vs
  API on accuracy *and* gCO₂). Caveats: needs RAPL/nvidia-smi, estimates, single-GPU.
- **Closing dummy on-device agent (stretch, if time):** tiny local agent (receipt
  image → best small-VLM arm → categorize/tally/answer), zero API. Reuses the winning
  sweep arm; lands "small open VLMs are enough, don't default to big closed APIs."

### Env note (2026-11-09)
Bumped `transformers` 5.10.2→**5.22.2** (d1 needs ≥5.19); hub 1.18→2.4, tokenizers/
safetensors bumped too; **torch untouched** (2.12.0+cu130). Verified: outlines arm
runs end-to-end on 5.22 (constrained gen parses + schema-valid); BAML arm unaffected
(HTTP to ollama). Optional: `causal_conv1d` not installed → slower LFM fallback kernel.

### OPEN DECISIONS (for the user, on return)
1. **Talk spine:** "evolve current story" (two enforcement families, neither fixes
   perception) vs "practitioner decision guide" (how to pick a stack). Affects
   framing/slide order only, not what we build. Recommendation: build toward the
   first; the guide is one summary slide on top. *Can be decided late.*
2. `time` column dropped from the SROIE table (SROIE has no time GT) — kept as an
   own-ticket vignette. *Explained; assume yes unless user objects.*

### Parked (future work, maybe revisit by talk length)
- **Detection decision-model arm** — different task (IoU/mAP), needs its own
  COCO/RefCOCO slice. Revisit only if the slot is >20 min. (Note: d1-2B does `box`
  localization in 8 GiB, so a local detection demo is now feasible if revived.)
- Decision-model-as-judge is **no longer parked** — now an active test via d1-2B
  (see the d1-2B block above).

### To verify before committing (unverified)
- LFM3-VL-2B extraction ability; whether a Liquid decision model does `Box` in 8 GiB.
- LocateAnything-3B's "runs on 8 GB" claim (one blog).
- `transformers` 5.x loader compat for Qwen3-VL-2B / Florence-2.
- SROIE license terms for conference use.

### Schema alignment (locked) — enables automated eval
Prompt + Pydantic `Receipt` + `baml_src/receipt.baml` all align to SROIE GT keys so
scoring is key-by-key, no human in the loop. **Aligned fields: `company` (was
`store`), `date`, `address` (NEW — in GT, hard long-field), `total` (bare number).**
`time` and `line_items` dropped for SROIE (no GT). Mirror across all 3 files
(existing consistency invariant). Two eval layers: **structural** = parses into the
class (schema's job); **field-correct** = `normalize(pred)==normalize(gt)` per field
(scorer's job). Keep the schema **loose on value format** — normalization lives in
the scorer, not a strict regex, or we'd score format compliance not reading (GT
dates vary: `25/12/2018` vs `12-01-19`; `total` is the bare rounded amount).

### Dataset (locked): `jsdnrs/ICDAR2019-SROIE` on HF
Clean mirror, verified via datasets-server preview. Fields: `image`, `key`,
`image_size`, **`entities{company,date,address,total}`** (GT dict, already
structured — no BIO collapse), plus bonus `words`+`bboxes` (text-region boxes, for a
possible detection revival). Splits: `train` (626) + **`test` (361)** = 987 receipts
(the competition literature quotes 347 test; this mirror has 361 — reason not verified,
always quote *our* 361). The `"TAN WOON YANN"` top text is original receipt content
(`words[0]`), constant, not an artifact — ignore. Load via
`datasets.load_dataset("jsdnrs/ICDAR2019-SROIE")` (HF cache, no manual `data/` dir);
GT = `row["entities"]`. `datasets` 5.1.0 added to the project (2026-11-09).
**Memory lesson:** a list of all decoded rows (~1000 PIL images) OOM-killed the
machine (exit 137, crashed VS Code). Never materialize rows — iterate lazily, and use
`ds.select_columns([...])` for GT-only work (no image decoding).

### Dataset analysis (for the talk intro) — from a full pass over all 987 GT rows
SROIE = Malaysian shop receipts (English, some Chinese), scanned, KIE task: 4 string
fields per receipt, no boxes/tokens in the label (the mirror's `words`/`bboxes` are
OCR text lines, a separate thing). The point to make: **the labels are human-curated
but wildly inconsistent in *format*, so evaluation must normalize, not string-match.**
- **Dates: 17 distinct format shapes.** Only 522/987 (~53%) use `DD/MM/YYYY` — the one
  format a strict regex like our earlier Pydantic one would accept. Others:
  `DD-MM-YY` 114, `DD-MM-YYYY` 110, `DD/MM/YY` 97, `05 MAR 2018` 84, `DD/MON/YYYY` 10,
  ISO `2018-12-25` 6, compact `20180304` / `25032018` 3, `11.02.18` 2, plus one-offs
  `(06/12/2016)` and `OCT 3, 2016`. 3 are **US month-first** (`12/28/2017`) —
  recoverable only because day>12 (a US date with both ≤12 would be silently misread;
  documented shortcut in `sroie.py`).
- **Totals: currency prefixes + sign quirks.** 99 `RM…` (`RM41.81`, `RM 3.90`), 50 `$…`
  (`$8.20`), 3 negative (`-1.73`, credit notes), 1 thousands comma (`1,007.50`),
  **1 empty** (excluded from that field's denominator). GT total = the final *rounded
  payable* (e.g. printed `TOTAL 33.92` → `TOTAL ROUNDED 33.90`, label `33.90`).
- **Missing GT:** 1 empty total (train), 1 `None` address (train); 0 in test.
- **Handwriting & damage:** some receipts carry a handwritten circled total
  (train[0], train[2]); train[0]'s address is smudged on the image (`NO.5:`) yet the GT
  reads `NO.53` — GT was curated, not a raw OCR dump. GT company keeps OCR noise
  (`BOOK TA .K`) → company/address scored fuzzy (difflib ratio ≥ 0.9), date/total exact
  after normalization.
- **No `time` field** (see earlier decision) — and no line items.

**Illustration picks (file `key` = original SROIE id; `split[index]` = HF row):**

| Phenomenon | split[idx] | key | GT |
|---|---|---|---|
| Canonical, handwritten circled total, smudged address | train[0] | X00016469612 | `25/12/2018`, `9.00` |
| `DD-MM-YY`, rounding (33.92→33.90), circled total, longest address | train[2] | X00016469620 | `12-01-19`, `33.90` |
| Month name + `$` | train[30] | X51005433492 | `05 MAR 2018`, `$8.20` |
| Compact date | train[68] / train[152] | X51005447850 / X51005715010 | `20180304` / `25032018` (`RM41.45`) |
| Month-first with comma | train[414] | X51006733494 | `OCT 3, 2016`, `RM33.90` |
| Parenthesised date | train[381] | X51006619545 | `(06/12/2016)`, `RM111.90` |
| US month-first | train[13]; **test[194]** | X51005268400; X51006414715 | `12/28/2017`; `4/22/2018` (`RM41.81`) |
| Negative total | train[347]; **test[209]** | X51006556824; X51006556654 | `-1.73`; `-6.42` |
| Thousands comma | train[350] | X51006556830 | `1,007.50` |
| Empty total | train[33] | X51005433522 | `''` |
| `None` address | train[104] | X51005663280 | `30 DEC 17` |

To export an image / crop: `load_dataset("jsdnrs/ICDAR2019-SROIE", split="train")[2]["image"].save("x.png")`.
For precise crops use the row's `words`+`bboxes` (`[x0,y0,x1,y1]` per OCR line, e.g.
the word containing the date) instead of guessing pixel coordinates.
Statistical note: at N=361, Wilson 95% half-width ≈ ±0.05 (worst case p=0.5).

### Build status
- **DONE — `sroie.py`** (2026-11-09): lazy loader `iter_samples(split)`, `norm_date`
  (→ ISO; 987/987 GT dates parse), `norm_total` (→ float; handles `RM`/`$`/`1,007.50`/
  French `27,49`), `norm_text`, `field_ok(field, pred, gt)` (True/False, or `None` when
  GT missing → drop from denominator). `uv run sroie.py` = assert self-checks + GT-only
  smoke print. FUZZY threshold 0.9 for company/address is a **choice to revisit** once
  we see real model outputs.
- **DONE — fuzzy metric switched to Levenshtein** (`rapidfuzz` 3.14.6, MIT, active):
  `difflib` is Ratcliff/Obershelp (not edit distance) and its autojunk made 3 typos in a
  244-char address score 0.42 vs 0.99 Levenshtein. Levenshtein is also the field
  standard (ANLS / CER). Threshold 0.9 on `normalized_similarity`.
- **DONE — `stats.py`**: `wilson(k,n)`, `mcnemar(a_ok,b_ok)` (exact, paired, skips None),
  `fmt(k,n)` → `19/20 = 95% [76-99]`. Stdlib only; self-checks vs textbook values.
- **DONE — schema aligned in all 3 files** (`compare_outlines.py`, `compare_baml.py`,
  `baml_src/receipt.baml`): `company str, date str, address str, total float` + shared
  French `@description` hints; greedy (BAML client `temperature 0`); dataset loop over
  `iter_samples("test")`, one generation per image, `N_IMAGES=5` smoke default (`None` =
  all 361); raw results dumped to `results_outlines.json` / `results_baml.json`;
  `CLAUDE.md` invariants updated. BAML images sent as lossless PNG (same pixels as HF).
- **FINDING (smoke, 5 images — not conclusive): `total: str` was a schema mistake of
  mine.** The -Extract model emits numbers (`193.0`); with `str` its baseline failed the
  schema (2/5) and outlines masked it into junk (`', '`). Switching to `float` fixed it
  (5/5, outlines == baseline values; remaining wrong totals are real misreads
  `411.5`/`4.62`/`2.2`). Nice talk anecdote: *the constraint's type decided the
  outcome* — a mask forces whatever shape the schema says, even against the model.
- **RESOLVED — date field holding date+time** (`'15/01/2019 11:05:16 AM'`, base LFM and
  SmolVLM2 do it): **strict `date` is the main number** (the field must hold only the
  date — the original date/time conflation, "right info in the right field"), plus a
  **`date_lenient` second column** (GT date appears anywhere in the string;
  `sroie.date_lenient`, shared via `score_fields`/`COLUMNS`). Smoke: base LFM strict 1/5
  vs lenient 5/5 → its date misses are placement, not reading.
- **DONE — 4 HF models verified on both paths (baseline + outlines):** LFM2.5-VL-1.6B,
  -Extract, **Qwen3-VL-2B-Instruct** (peak 4.1 GiB), **SmolVLM2-2.2B-Instruct** (4.5 GiB;
  needed `uv add num2words`). All run in `compare_outlines.py` (`MODEL_IDS`).
- **Two-axes detail:** Qwen3-VL wraps JSON in ```json fences despite "sans markdown" →
  prompt-only **valid JSON 0/5** yet date/address/total 5/5. So `valid_json`/`valid_schema`
  are strict on the raw text, but the *accuracy* axis reads the payload (`lenient_json`)
  — otherwise formatting noise would zero a correct read and collapse the two axes.
- **DONE — crash safety:** every result is appended to `results_<arm>_<TAG>.jsonl` the
  moment it exists; a rerun skips finished (model, method, key) and a torn last line is
  ignored (tested: deleted 3 records + a half-written line → exactly those 3 regenerated,
  identical to the first run = greedy is deterministic). A CUDA-OOM/exception on one
  image is recorded (`error`) and the run continues. Bump `TAG` when prompt/schema/
  GEN_KWARGS change (new file, never mixed). Old `results_*.json` smoke files deleted.
- **Timing (5-image smoke):** HF ≈ 1.0–2.4 s/image (Liquid 1.0–1.6, Qwen 2.0–2.3,
  SmolVLM2 2.0–2.4); BAML via ollama ≈ 13 s/image (4–25 s, scales with image size).
  `ollama ps`: 100% GPU, 2.5 GB, ctx 4096 → NOT CPU offload; gap vs HF (~10×) likely
  llama.cpp's image path (unverified). Full run ≈ 40–50 min (4 models, outlines arm) and
  ≈ 80 min (BAML arm). Set `BAML_LOG=WARN`.
- **RESULT — BAML on vLLM vs ollama (5-receipt smoke, same BAML code, `ENGINE=` switch):**
  vLLM **1.1 s/image vs ollama 13.3 s (≈12× faster)**. Not a resolution trick: input
  tokens are identical (1885 vs 1895 on the biggest receipt) and output length equal
  (~82). Ollama's time grows with input tokens (~15 ms/token: 4.5 s @ 345 tok → 28.6 s
  @ 1895 tok) while vLLM stays ~1.1 s → llama.cpp's image/prefill path is the bottleneck
  (cause not isolated). **Accuracy is NOT the same across engines even with "same bf16
  weights":** on the big receipt X51005200931 vLLM read total 411.5 (HF baseline also
  411.5) while ollama read the correct 436.2 — so on that receipt llama.cpp read BETTER,
  the opposite of the n=1 claim "llama.cpp degrades date perception". 5 images prove
  nothing; the engine is a real confound to measure (full-run McNemar between engines).
- **vLLM setup (separate env, project pins untouched):** `~/envs/vllm` (uv venv py3.12,
  vLLM 0.33.0, torch 2.14+cu130, 8.9 GB). Start (needs the FlashInfer sampler OFF — no
  `nvcc` on this machine, otherwise EngineCore dies compiling the sampling kernel):
  `VLLM_USE_FLASHINFER_SAMPLER=0 ~/envs/vllm/bin/vllm serve LiquidAI/LFM2.5-VL-1.6B
  --max-model-len 8192 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.6
  --port 8000` (the 4096 is REQUIRED: with the default 2048 image-encoder cache vLLM
  rejects every receipt above 2048 image tokens); then
  `ENGINE=vllm uv run compare_baml.py`. It holds ~4.7 GB of the 8 GiB GPU → **stop it
  before any HF run** (outlines arm, d1). Client `CustomVllm` in `baml_src/clients.baml`.
  Beware: never `pkill -f "vllm serve"` from a command that contains that string (kills
  its own shell); redirect the server log to a file (a `| tail` hides the root cause).
- **RESULT — full BAML run on vLLM, 361/361 test receipts (greedy; `results_baml_v1.jsonl`):**
  object produced 361/361 [99-100]; company 330/361 = 91% [88-94]; **date (strict)
  273/361 = 76% [71-80]**; address 315/361 = 87% [83-90]; total 334/361 = 93% [89-95];
  **date (lenient) 349/361 = 97% [94-98]**. The 21-point strict/lenient gap is date+time
  stuffing, not misreading. 0 server failures, 0 parse failures. Input tokens median 1885
  / max 2916, output ~83.
- **Cost on vLLM (full run): 1.50 s and 118 J of GPU per receipt on average; 11.8 Wh in
  total over 9 min.** By receipt size: <1.5 Mpx (181 receipts) 1.33 s / 112 J; 1.5–20 Mpx
  (125) 1.46 s / 120 J; >20 Mpx scans (55, 4961×7016 px) 2.18 s / 133 J. Scans cost
  ~0.85 s more: ~0.6 s client PNG encoding (we send lossless PNG) + server decode of a
  35-Mpx image. Energy correlates with input tokens (r = 0.62). The 5-receipt smoke
  (≈0.9 s, 75 J, same ~84 W) was optimistic — cause not isolated (laptop GPU under
  sustained load?). **So the "≈12× faster than ollama" above is PROVISIONAL**: it comes
  from 5 mid-size receipts; the ollama full run gives the real ratio.
- **THREE BUGS FOUND ON THE FIRST FULL RUN — the numbers it first printed (~50% accuracy,
  55% objects) were WRONG and must never be quoted:**
  1. vLLM rejected **161/361 receipts with HTTP 400**: its image-encoder cache defaults to
     2048 tokens and 161 receipts need more. The LFM processor caps an image at 2856 tokens
     (median prompt 1825, p90 2846 — counted on all 361 with the processor alone, no
     model); exactly 161 are above 2048. Fix: `--max-num-batched-tokens 4096` (the cache
     size follows it). The script had scored these failures as all-wrong.
  2. **Some SROIE receipts are grayscale** and the LFM processor crashes on 1-channel
     images ("output with shape [1,1,512,512] doesn't match the broadcast shape
     [1,3,512,512]"). Fix in the loader: `iter_samples` converts to RGB and restores
     `.format` (outlines' `Image()` needs it); asserted RGB+JPEG on all 361. It would have
     crashed or mis-scored the outlines arm.
  3. **Failure handling:** infrastructure failures (`error`: server rejection, OOM) are
     now retried on a rerun and reported apart (excluded from accuracy); only a model text
     BAML cannot parse (`parse_error`) counts as wrong. `results_baml_v1.jsonl` still holds
     the 161 superseded error rows; readers keep the last record per receipt.
  Also learned: ollama's 4096-token context is enough (max prompt 2856 + ≤512 output), so
  the ollama arm is not silently truncated.
- **Outlines arm smoke re-verified (5 receipts × 4 models):** 40 clean records with
  `energy_j`, `input_tokens`, `output_tokens`; per-generation GPU energy ≈ 60–190 J
  (Extract lowest ≈ 60–80, Qwen3-VL ≈ 135–175, SmolVLM2 ≈ 145–190; same first receipt:
  348 prompt tokens for LFM, 497 Qwen, 1210 SmolVLM2). Full run not done yet.
- **OLLAMA FULL RUN DONE (361/361, 0 server failures, greedy BAML, same receipts as vLLM):**
  ollama 25.3 s & 950 J GPU/receipt vs vLLM 1.50 s & 118 J → **vLLM ≈17× faster, ≈8× less
  GPU energy** (replaces the provisional "12×"). Object 358/361 (99%). Accuracy ollama vs
  vLLM: company 87 vs 91, date strict 76 vs 76, address 83 vs 87, total 89 vs 93, date
  lenient 92 vs 97. Paired McNemar (vLLM vs ollama): company p=0.003, address p=0.02,
  total p=0.02, lenient date p=0.0003, strict date p=0.89 (tie). → **the ENGINE changes
  accuracy, not only speed**; cause not isolated (GGUF/llama.cpp image preprocessing vs
  HF bf16? quantization path?). 4 tests → mild multiple-comparison caveat. Always quote
  the engine next to an accuracy number.
- **IDEA (decided worth doing, 2026-11-09): vLLM guided decoding arm.** Since the engine
  matters, Family A (token masking) is currently confounded with the HF stack. Run the
  SAME model (LFM2.5-VL-1.6B, bf16) on vLLM with structured outputs (`response_format`
  json_schema / guided_json, xgrammar backend) and compare: (a) vLLM free vs vLLM guided
  (clean enforcement effect on a fast engine), (b) vLLM guided vs HF+outlines (same
  mask idea, different engine/implementation), (c) vLLM+BAML vs vLLM guided (Family B vs
  A on ONE engine — the cleanest A-vs-B comparison, removes the engine confound).
  Cheap: ~9 min per arm on 361. Check vLLM 0.33 structured-output flags (no nvcc
  → xgrammar must not need JIT compile; verify on 5 receipts first). Only the LFM base
  model unless time allows (Extract/Qwen/SmolVLM2 also run on vLLM if supported).
- **OUTLINES ARM FULL RUN DONE (2026-11-10; 4 models × {baseline, outlines} × 361, greedy, HF
  transformers; `results_outlines_v1.jsonl`).** Qwen3-VL OOM'd on the 55 A4 scans (32–35 Mpx,
  no token cap in its processor) → rerun with `QWEN_MAX_PIXELS = 9_000_000` (every other
  receipt ≤ 8.7 Mpx, so unchanged; only the 55 scans are downscaled — a Qwen-only protocol
  note). Final 361/361, 0 errors. Accuracy baseline → outlines (strict date; lenient in []):
  - LFM2.5-VL-1.6B: JSON valid 100% both; company 92/92, date 64/64 [97/97], address 65/65,
    total 91/91 — outlines output IDENTICAL to baseline (0 discordant pairs): the base model
    already obeys the prompt, so masking changes nothing. 1.4 s/120 J vs 1.7 s/127 J.
  - LFM2.5-VL-1.6B-Extract: JSON valid 86% → 100%, schema 70% → 100%; company 56 → 81,
    date 61 → 73 [82 → 98], address 60 → 71, total 57 → 69 (McNemar p<0.001 all; nearly
    all discordant pairs in outlines' favour). Biggest enforcement win — but still below
    the base model on every field: enforcement rescues structure/format, not reading.
  - Qwen3-VL-2B: JSON valid 0% → 100% (baseline wraps JSON in ```fences```), but the values
    are the same (company 90/90, date 97/97, address 93/92, total 93/93; all p>0.3).
    Best small model overall; 3.2 s/262 J.
  - SmolVLM2-2.2B: JSON valid 92% → 99%; company 61 → 65 (p=0.004), date lenient 85 → 90
    (p=0.002), date/address/total no significant change; worst reader (54–71%).
  → Headline holds at scale: enforcement fixes structure (0–92% → 99–100% valid) and can
  rescue a model whose raw output is malformed (Extract), but never lifts accuracy above
  what the model perceives (Qwen unchanged, LFM base unchanged).
  **CAVEAT to chase:** HF LFM base (address 65, date 64) vs vLLM+BAML (address 87, date 76)
  on the SAME weights and receipts: a big gap, and it's not only the engine — BAML builds
  its own prompt (schema rendering) and the images differ (HF: original JPEG, BAML: PNG).
  The vLLM guided arm must reuse the HF arm's exact prompt to separate prompt vs engine.
- **PROMPT-LANGUAGE LIMITATION (noted 2026-11-10, NOT fixed for the main runs):** every
  prompt in the study is in FRENCH (prompt + schema hints in compare_outlines.py,
  compare_baml.py, receipt.baml) while SROIE is English/Malay and the models were trained
  mostly on English instructions. We did not test this before the full runs. A French
  prompt may cost accuracy for all models (esp. small ones) and is a confound for any
  "model X reads badly" claim. To state in the talk as a limitation; an English-prompt rerun
  of the arms (outlines 4 models ~45 min, BAML vLLM ~9 min) is the way to measure it — not
  done yet, decision deferred.
- **EXTRACT PROMPT-FORMAT FINDING (2026-11-10):** the model card of LFM2.5-VL-1.6B-Extract
  prescribes a native format: SYSTEM turn = English YAML field list ("field: description")
  + "Respond with only a JSON object. Do not include any text outside the JSON."; USER
  turn = the image only. Our French user-turn prompt is out of its training distribution,
  and the baseline Extract numbers above (56/61/60/57, 14% non-JSON) are therefore NOT a fair
  "Extract without outlines" — raw outputs show broken JSON (`{"{"company"…`, key loops) and
  invented keys (`bien`, `gst_reg_no`); among its 253 schema-valid outputs it reads like the
  base model (company 80, date 71, address 72, total 67). Expected if prompted natively:
  Extract ≥ base, and Extract baseline ≈ Extract+outlines. FIX (done, running): variant
  `LiquidAI/LFM2.5-VL-1.6B-Extract#native` in compare_outlines.py (`NATIVE_SYSTEM`,
  `build_messages`), stored under its own label in results_outlines_v1.jsonl (French-prompt
  Extract records kept for the record, never mixed). Other models keep the common prompt.
- **ENGLISH RERUN PLAN (decided 2026-11-10; v2 = English, v1 = French kept):** pilots on 25
  receipts (baseline, EN vs FR): LFM base unchanged, Qwen unchanged (still fenced), SmolVLM2
  clearly better in English (company 16→21, date 19→23, address 11→14, total 16→20 /25) → its
  French numbers understate it. Extract total-wording test (40 receipts: current 22, "grand
  total incl. tax and rounding" 26, "line labelled TOTAL" 21): wording is not the lever — misses
  are misreads or GT rounding (4.90 vs 4.62); wording left alone, no per-model tuning.
  Faithful translation only (language is the single changed variable); date-only spec NOT
  added (strict date still penalizes "date + time"). Extract excluded (native English prompt
  already; reuse its v1 `#native` records). Status: BAML/ollama v2 launched 2026-11-10 evening
  (`results_baml_v2.jsonl`, receipt.baml translated + regenerated, TAG="v2" in compare_baml.py,
  ~2.5 h — NOT 80 min: 25 s/receipt). TODO next day: outlines 3 models (TAG v2 + English
  prompt/schema_desc in compare_outlines.py, ~85 min), BAML vLLM v2 (~9 min).
- **BAML/OLLAMA ENGLISH RUN DONE (v2, 361/361, 0 server errors) — WORSE than French:** object
  produced 91% [87-93] (FR 99%), company 80 (FR 87), date 69 (FR 76),
  address 77 (83), total 82 (89), date lenient 86 (92); paired McNemar FR vs EN all p<0.002,
  FR-only-right ≈ 30 vs EN-only-right ≈ 5 per field. Same cost (25.5 s, 958 J). CAUSE (raw
  outputs): with English `@description`s the model renames KEYS after the descriptions —
  `store_name` ×46, `purchase_date` ×34, `total_amount` ×33 (FR: `nom_du_enseigne` ×2 → the 3
  failures of v1) — and BAML's SAP cannot map them to company/date/total (33 receipts fail to
  parse; others salvaged but misaligned). Outlines is immune (keys forced); the HF prompt shows
  the JSON schema inline and had 0 wrong key sets in the pilot. LESSON (talk-worthy): with
  schema-aligned parsing, description wording that looks like an identifier ("name of the
  store", "total amount paid") leaks into the key names of a small model, and SAP can't
  recover — a robustness weakness of Family B, invisible to Family A. UNKNOWN: does vLLM show the
  same (tomorrow's vLLM v2 run answers it in ~9 min). Decide on description wording ONCE,
  after that run — no ollama rerun before then. Both v1 (FR) and v2 (EN) files kept.
- **BAML/vLLM ENGLISH RUN DONE (v2, 361/361, 0 errors, 2026-11-11) — NO key renaming:** object
  100%, 0 renamed keys (vs ollama-EN 46 `store_name`…). Accuracy EN vs FR: company 92 vs 91
  (p=0.63), address 88 vs 87 (p=1), date lenient 96 vs 97 (p=0.63), total 91 vs 93 (p=0.04, 1 vs
  8), date strict 71 vs 76 (p=0.0002, 2 vs 19 — likely date+time stuffing, not misreads). Same
  cost (1.5 s, 120 J). → the key-renaming failure is NOT a language effect on BAML in general: it is
  specific to the ollama/GGUF path (or its chat template/prompt rendering); vLLM+BAML is robust to
  the language change except for a small strict-date/total dip. Two engines, same weights, same
  BAML code → very different robustness: engine = part of the result. v1 (FR) and v2 (EN) both
  kept in results_baml_v*.jsonl (v2 file holds the ollama AND vLLM EN records, model label tells
  them apart). Decision on description wording: not needed for vLLM; ollama-EN stays as reported.
- **DIAGNOSIS from raw outputs (2026-11-11) — corrects the two bullets above:**
  (a) ollama-EN is NOT "worse in English in general": on the 315 receipts where it kept the
  right keys, EN ≈ FR (company 276 vs 275, date 239 vs 241, address 267 vs 263, total 285 vs
  286 /315). The entire drop comes from the 46 receipts with renamed keys, and 28 of those are
  ONE shop (UNIHAKKA INTERNATIONAL SDN BHD) → a single repeated receipt layout (SROIE shop
  clustering!) drives most of the "−8 pts". Why that shop triggers it on ollama only is NOT
  established (hypotheses: GGUF/llama.cpp image handling or chat template makes the model
  copy the description text as keys on that layout; untested — test = render the exact prompt
  ollama sends, try vLLM-vs-ollama on those 28 receipts, try other descriptions).
  (b) vLLM-EN strict-date dip (−19 net): 17/19 are "date + TIME" ('18/01/2018 17:09:21'):
  English "purchase date as printed" invites the time more than French "date d'achat telle
  qu'imprimée"; lenient date unchanged (96 vs 97) → a field-spec wording effect, not misreading.
  vLLM total dip (8 vs 1): mostly rounding/misreads (6.35→6.0, 31.80→30.0, 278.80→263.02).
  Statistical note: shop-level clustering means p-values on pooled receipts are optimistic.
- **OUTLINES ARM ENGLISH RERUN DONE (v2, 2026-11-11; `results_outlines_v2.jsonl`; 3 models ×
  {baseline, outlines} × 361, 0 errors; Extract not rerun — use its v1 `#native` records).**
  Baseline → outlines (company / date strict / address / total; date lenient):
  LFM base 91→91 / 59→59 / 65→65 / 91→91; 97→97 (outputs identical, as in FR). Qwen3-VL 91→91 /
  96→97 / 92→91 / 94→94; 98→98; JSON valid 0%→100% (still fenced). SmolVLM2 63→65 / 76→81 /
  58→60 / 75→74; lenient 89→93; JSON valid 97→99%. English vs French (baseline, paired): LFM
  unchanged except strict date 64→59 (20 vs 2, time-appended, lenient unchanged 97); Qwen
  unchanged (all p>0.2); SmolVLM2 better in English: date strict 63→76 (61 vs 13, p<0.001),
  company 61→63, address 54→58, total 71→75 (each p≈0.1, same direction) → its French numbers
  understated it. Conclusion unchanged: headline holds in English; the French-prompt limitation
  mostly mattered for the weakest model (SmolVLM2) and for date format (time appended).
  Caveat: shop clustering ⇒ p-values optimistic.
- **vLLM GUIDED ARM — GROUND PREPARED (2026-11-11), full run not started.** `nvcc` is now
  installed (CUDA 12.4) but the vLLM env's torch is cu130 → a version mismatch, and NOT needed:
  structured outputs use prebuilt wheels (xgrammar 0.2.8, outlines_core 0.2.14, llguidance);
  only FlashInfer's sampler needs nvcc and stays off (`VLLM_USE_FLASHINFER_SAMPLER=0`; greedy
  doesn't need it). New script `compare_vllm_guided.py` (English prompt/schema/decoding identical
  to the HF arm; image as lossless PNG; plain `requests`, no new dependency; JSON-schema
  `response_format`; records `time, energy_j, input/output_tokens, raw, ok`; resumes;
  results in `results_vllm_guided_v1.jsonl`; methods `free` (once) and `guided-<BACKEND>`;
  summary also pairs guided vs free, vs HF+outlines v2, vs BAML-vLLM v2). Smoke 5 receipts OK
  with xgrammar; enforcement verified with a contradicting schema (asked for a fruit → server
  returned `{"zebra_count": 1}`) on BOTH backends; free and guided outputs identical on the
  5 receipts (base model already valid, as in the HF arm). Two backends worth running: xgrammar
  (vLLM default) and `outlines` (same core as the HF arm → isolates engine from algorithm).
  Server flag: `--structured-outputs-config.backend xgrammar|outlines`. Full run ≈ 9 min per
  method (free once + guided per backend → ~27 min total incl. restarts).
- **OLLAMA + `format` (grammar-enforced) ARM — CODE READY, NOT SMOKE-TESTED (user idea,
  2026-11-11):** `ENGINE=ollama uv run compare_vllm_guided.py` (same script; calls ollama's NATIVE
  `/api/chat` with `format` = Pydantic JSON schema, `options={temperature 0, num_predict 256}`,
  base64 `images` — exactly what the documented `ollama.chat(..., format=Receipt.model_json_schema())`
  sends; enforced by a llama.cpp grammar). Guided only (prompt-only reference = ollama+BAML v2); results in
  `results_ollama_format_v1.jsonl`; ~25 s/receipt ⇒ ~2.5 h for 361, run overnight, vLLM server
  must be stopped. Hypotheses to test: (1) structure becomes 100% (BAML-ollama-EN lost the keys on
  46 receipts, 91% objects); (2) values stay at ollama's level (below vLLM's: engine/image
  handling, not structure); gives Family A vs Family B on ollama. Open check: smoke on 5
  receipts first (does the grammar work with this multimodal GGUF?). (First version used the
  OpenAI-compatible /v1 `response_format`; replaced by the native call after the user asked
  whether it matched ollama's docs — equivalence of /v1 was never verified.)
- **vLLM GUIDED ARM DONE (2026-11-11, 361 × {free, guided-xgrammar, guided-outlines}, 0
  errors; `results_vllm_guided_v1.jsonl`).** Same model/prompt/decoding as the HF arm.
  (1) guided == free on EVERY field for BOTH backends (0 discordant receipts of 361): company 91,
  date 58 [lenient 97], address 64, total 91, JSON 100% — the base model already emits valid
  JSON, so masking changes nothing (same as HF+outlines). (2) vLLM (free or guided) ≈ HF
  baseline/outlines (company 91/91, date 58–59, address 64–65, total 91/91): with the SAME prompt
  the engine does NOT change accuracy between HF and vLLM. (3) → this RESOLVES the earlier
  caveat: the big gap vs BAML-on-vLLM (address 88 vs 64, date strict 71 vs 58; McNemar
  p<1e-11) is a PROMPT effect, not an engine effect — BAML's rendered prompt (`ctx.output_format`)
  beats our hand-written one, and since parsing can't add correct values (guided==free) the
  gain is in how the prompt is built. Not yet tested directly (test = feed BAML's rendered
  prompt to the guided arm, ~10 min; or log `b.ExtractReceipt` prompt). So the engine effect
  that DOES exist is ollama (GGUF/llama.cpp) vs vLLM/HF. (4) CAVEAT on cost: guided-xgrammar shows
  1.1 s/88 J vs free 1.5 s/120 J — an ARTIFACT: in the xgrammar pass each receipt ran `free`
  then `guided` with the identical image+prompt, so the 2nd call hits vLLM's prefix cache.
  guided-outlines (no pairing) = 1.5 s/120 J = free. Do NOT quote xgrammar's 88 J; to compare
  backends' cost, rerun each guided method alone (~9 min) or disable prefix caching.
- **PERSPECTIVES / FUTURE WORK (user, 2026-11-11 — to list in the README; not all will be done
  or presented):** (a) receipt-level accuracy (all 4 fields right) — free from saved results;
  (b) data contamination: SROIE public since 2019, models may have seen it → small private set
  (20–30 own receipts incl. French) as a sanity check — also covers "no French docs in the
  benchmark"; (c) resolution/perception experiment (limited scope, fast: e.g. 2–3 image sizes
  or a zoom crop on the total, vLLM ~10 min per setting) to test "perception is the bottleneck"
  directly and find the real lever; (d) error-detection baseline: the extractor's OWN token
  logprobs (vLLM `logprobs`) as a free confidence signal that the d1 judge must beat
  (NB: d1 is a different model run via HF `system_one`, judging from TEXT — SROIE `words` =
  oracle OCR, an optimistic upper bound; it is not served by vLLM); (e) deployment realism:
  4-bit quantization (accuracy/energy), concurrency/throughput (vLLM's gap should widen
  under parallel requests), VRAM footprint and load time per setup; (f) reproducibility:
  store library versions/hardware/commit in each results file (a header line); (g) already
  planned: error taxonomy (misread / label convention / label noise), shop-clustered
  bootstrap CIs, d1 judge, sampling-variance probe, ecology cross-check (codecarbon/ecologits),
  larger reference model, French-ticket vignette, on-device agent stretch.
- **BEFORE COMMITTING ANYTHING (user, 2026-11-11):** `README.md` still describes only the
  original single-receipt study (plus a new "Perspectives" section) — it MUST be updated to the
  scaled study (SROIE, 361 receipts, results, files list incl. sroie.py/stats.py/energy.py/
  compare_vllm_guided.py, vLLM/ollama setups) before the first commit. `slides.html` stays untracked.
- **PROMPT VARIANTS on vLLM (free, 361 receipts, no prefix cache, 2026-11-11) — the schema block carries BAML's gain.**
  Same model/decoding; only the prompt differs. A = BAML's exact rendered prompt; B = our wording + BAML's
  commented schema block (`// description` above each field); C = our wording + inline schema, in BAML's
  order (instruction → image → schema). Strict accuracy %: (company / date / address / total; lenient date)
  ours 91/58/64/91 (97) · **A 92/71/87/91 (96)** · **B 93/70/89/92 (97)** · C 90/60/61/91 (97) ·
  BAML on vLLM 92/71/88/91 (96).
  - A reproduces BAML (address 87 vs 88, date 71 vs 71) → BAML's advantage = its rendered prompt, not
    engine nor parser (raw `json.loads` on BAML outputs scored identically on 361/361).
  - B gets the full gain (address 64→89, strict date 58→70) → it is the schema block, not our wording.
  - C gets none (address 61, date 60) → instruction/image order and instruction split do not matter.
  - Lenient date ≈ 97 in every variant: the model reads the date right; the strict gain is only "no time
    appended". Timing/energy comparable (~1.5 s, 117–120 J/receipt, prefix caching off).
  - Not done: McNemar of A/B vs ours (gaps far outside Wilson CIs, expected to hold).
  - Corrected prompt comparison: BAML sends text "Extract the information from this receipt." → image →
    text "Answer in JSON using this schema:\n{ // desc\n field: type, ... }" (image is NOT first).
- **DECISION (user, 2026-11-11): prompt B becomes the common prompt.** Old results are KEPT as
  protocol versions (never overwritten): outlines v1 (FR) / v2 (EN inline schema) / **v3 = prompt B**;
  vLLM guided v1 (inline prompt + legacy free-A/B/C) / **v2 = prompt B**; ollama format v2 = prompt B
  (v1 = 5 smoke records, inline prompt); BAML v1/v2 unchanged (BAML renders its own block ≈ prompt A, so
  it is already on the better format; no rerun needed). Report "prompt format moves address by ~25 pts"
  as its own finding, with the v2-vs-v3 pairs as evidence.
- **RUNNING (2026-11-11):** ollama + native `format`, prompt B → `results_ollama_format_v2.jsonl`
  (`ollama_format_B.log`, ~2.5 h). Then re-run on prompt B (code already switched; one run each):
  vLLM free + guided-xgrammar (`BACKEND=xgrammar FULL=1`, → vllm_guided v2; ~10 min each, server started
  with `--no-enable-prefix-caching` for honest timing) and HF outlines (`FULL=1 uv run compare_outlines.py`
  → outlines v3, 3 models, ~1 h+; Extract stays on its native prompt, excluded).
- Paired tests above come from `compare_arms.py` (re-runnable, reads the saved JSONL files only).
- **PROMPT-B RESULTS (361 receipts, greedy; 2026-11-11; all runs finished 19:08).** Strict accuracy %
  (company / date / address / total; lenient date), cost per receipt. Files: `results_outlines_v3`,
  `results_vllm_guided_v2`, `results_ollama_format_v2`. Logs: `outlines_v3_B.log`, `vllm_guided_B.log`, `ollama_format_B.log`.
  | arm | struct. valid (JSON/schema) | company | date | address | total | date lenient | cost |
  | HF LFM base, free | 361 / 360 | 90 | 69 | 89 | 92 | 97 | 1.5 s, 125 J |
  | HF LFM base + outlines | 361 / 361 | 92 | 69 | 88 | 93 | 97 | 1.7 s, 127 J |
  | vLLM LFM free | 361 / 360 | 91 | 70 | 89 | 92 | 97 | 1.6 s, 124 J |
  | vLLM LFM guided (xgrammar) | 361 / 361 | 93 | 70 | 89 | 92 | 97 | 1.4 s, 102 J |
  | ollama LFM + `format` | 360 / 360 | 73 | 74 | 79 | 89 | 92 | 28.5 s, 893 J |
  | HF Qwen3-VL-2B, free | 0 / 0 (fences) | 82 | 98 | 91 | 94 | 98 | 3.3 s, 268 J |
  | HF Qwen3-VL-2B + outlines | 361 / 361 | 81 | 97 | 88 | 95 | 98 | 3.8 s, 274 J |
  | HF SmolVLM2-2.2B, free | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 2.1 s, 175 J |
  | HF SmolVLM2 + outlines | 359 / 359 | 65 | 61 | 58 | 79 | 92 | 2.2 s, 177 J |
  (guided-xgrammar timing/energy is valid here: prefix caching was off. BAML on vLLM/ollama rows unchanged, see above.)
  Paired exact McNemar (only-first / only-second right, p; shop clustering makes p optimistic):
  - **Enforcement effect (same engine):** vLLM guided vs free: company 6/0 p=0.03, all else ≤1 discordant
    (date 0/0). HF outlines vs free: company 8/0 p=0.008, total 5/0 p=0.06, date 3/5, address 6/11 (ns).
    Qwen outlines vs free: address 1/12 p=0.003 (outlines HURTS address), rest ns. → masking never raises
    accuracy materially; it only changes structure (Qwen 0→100%, SmolVLM2 0→99% valid) and can cost accuracy.
  - **Engine effect, HF vs vLLM (same prompt):** free 4/1 company, 3/1 date, all ns; guided-vs-outlines all ns
    (date 8/4, address 11/7, total 0/4). → no engine effect between HF and vLLM (confirmed on prompt B).
  - **Ollama vs vLLM guided:** company 4/73 p=2e-17, address 13/50 p=3e-6, lenient date 2/21 p=7e-5, total 9/20
    p=0.06, strict date 38/23 p=0.07 (ollama higher on strict date = less time-stuffing, not better reading).
    vs HF outlines: same direction, total 5/20 p=0.004, date 40/21 p=0.02. → ollama/llama.cpp loses accuracy
    (cause not identified: see side quest).
  - **Prompt effect (inline schema v2 vs prompt B), same model/engine:** HF base address 95/5 p=1e-22, date 38/0
    p=7e-12; HF outlines address 89/4 p=6e-22, date 36/0 p=3e-11; vLLM free address 94/4 p=2e-23, date 42/1
    p=1e-11; company/total ns, lenient date ns. → the schema block format is the single biggest lever found
    (+25 pt address, +11 pt strict date), larger than any engine or enforcement effect; consistent on both engines.
  - **Model finding:** with prompt B, Qwen3-VL-2B reads dates far better than LFM (98 vs 69-70 strict; lenient 98 vs 97)
    but is worse on company (82 vs 90-93); its structural failure is fences only (payload fine), which
    outlines fixes. SmolVLM2 is unusable without masking and mediocre with it (company 65, address 58).
  - Headline still holds: enforcement fixes structure for models that break format and is neutral for those
    that don't; it never lifts accuracy above what the model perceives; prompt format and (for ollama) engine move
    accuracy far more than enforcement does.
- **EXTRACT on prompt B + BAML pairs + clustered analyses (2026-11-12).** Files: `results_outlines_v3` (Extract added),
  `compare_arms_B.txt`, `analysis_B.txt` (from `analyze_results.py`: receipt-level accuracy, shop-clustered bootstrap CIs, error taxonomy).
  - Extract, prompt B: free 24% valid JSON / 12% schema, acc 17/15/16/13 (lenient date 16); + outlines 99% valid, 83/89/79/73 (date lenient 97).
    Extract, native prompt: free 85/80/78/68 (97) = outlines 85/81/78/68 (97) → free == guided once the prompt is right (as the user expected).
    Native outlines vs base+outlines B: company 2/29, address 17/55, total 4/96 (p<1e-5): Extract is worse than the base model
    except strict date (49/6, it does not append the time); lenient date equal.
  - BAML vs prompt-B arms (vLLM): BAML vLLM vs vLLM free/guided: all ns (p≥0.15); vs HF outlines: total 2/11 p=0.02, rest ns
    → BAML (SAP) ≈ free ≈ guided on the same engine. BAML ollama vs BAML vLLM: company 4/49, address 9/46, total 9/40, lenient date 2/40
    (all p<1e-5): the ollama accuracy loss shows up under BAML too (engine, not enforcement). BAML ollama vs ollama+format:
    BAML better on company (58/36 p=0.03), `format` better on total (9/34) and lenient date (7/30) (p<2e-4) → on ollama, masking
    helps total/date, hurts company.
  - Shop-clustered bootstrap (153 shops, 361 receipts): conclusions survive. Significant: vLLM guided−free company +1.8 [0.3,3.6];
    HF outlines−free company +2.4 [0.8,4.4]; Qwen outlines−free address −3.2 [−6.7,−0.4]; ollama−vLLM guided company −19.8 [−27,−13],
    address −9.1 [−17,−2], lenient date −5.3 [−8,−2.5]. Everything else CI includes 0 (HF vs vLLM free: no engine effect).
  - Receipt-level (all 4 fields right): LFM HF/vLLM ≈ 53% strict date / 73% lenient date (bootstrap CI ±12/±9), ollama+format 42/55,
    BAML ollama 49/61, Qwen 70/71 (Qwen > LFM when strict date counts, ≈ when lenient).
  - Error taxonomy (vLLM free): date fails 103 = 92 right date + extra text, 10 wrong, 1 missing; total fails 26 = 23 wrong value of
    right magnitude (other amount on the receipt), 2 far off, 1 missing; company 30 = 22 different text, 4 truncated, 2 near miss, 2 contains+extra;
    address 37 = 17 contains GT + extra, 11 near miss, 7 truncated, 2 different. ollama+format company fails 91 = 61 "contains GT + extra text".
  - README rewritten to the scaled study (2026-11-12). Next: resolution/perception study.
- **SIDE QUEST (future, not for the talk): why is ollama/llama.cpp less accurate than vLLM for the same
  model?** Observed (prompt B, 361 receipts): ollama+`format` company 73 / address 79 / lenient date 92 vs
  vLLM guided 93 / 89 / 97, and BAML on ollama is also lower, so it is neither the prompt nor the masking.
  Hypothesis (UNVERIFIED, raw outputs not yet inspected): llama.cpp's image preprocessing (resize/tiling,
  max image tokens, mmproj/vision-tower handling) differs from the HF/vLLM processor. To investigate:
  compare image-token counts per receipt (ollama `prompt_eval_count` vs vLLM `prompt_tokens`, both saved),
  read the failing outputs, and test pre-resizing the image to a fixed size before sending to both engines.
- **NEXT after that:** d1 date-judge; sampling-variance probe (30 images × 10 runs); side quests
  (codecarbon, ecologits, agent).

### Statistics — how to read the results (Wilson interval & McNemar test; `stats.py`)
Unit of analysis = the receipt. Every accuracy is a count k out of n receipts.

**Wilson 95% interval — "how far could the true accuracy be from what we measured?"**
We only tested n receipts, not all receipts in the world; the interval is the range of
true accuracies compatible with the k/n we saw. Margin at 50% by sample size:
±20 pts @ n=20, ±13 @ 50, ±10 @ 100, **±5 @ 361 (our test split)**, ±3 @ 1000 — it
shrinks with √n, so a 3-point gap between two models on 361 receipts is inside the noise.
Why Wilson and not the textbook `p ± 1.96·√(p(1−p)/n)`: the textbook formula collapses at
the extremes (20/20 → "100% ± 0, certainly perfect"; 0/5 → "0% ± 0"), Wilson stays
honest: 20/20 → 84–100%, 5/5 → 57–100%, 0/5 → 0–43%. Printed as `19/20 = 95% [76-99]`.
- **What it is NOT (common misreading): it does not replace repeated runs.** It measures
  uncertainty from *which receipts we sampled* ("would the score hold on other
  receipts?"), not randomness *of the model* ("would another run give the same output?").
  Two different questions: (1) other receipts → Wilson on the 361; (2) other runs → the
  sampling-variance probe (30 receipts × 10 runs). Under **greedy decoding (our main
  protocol) run-to-run variance is ~0** — verified, regenerated records were identical —
  so repeated runs would add nothing; repetition only matters with sampling (temp > 0)
  or non-deterministic engines. Wilson's real benefit: an uncertainty estimate from one
  pass, without re-running.
- Caveats: assumes independent, representative receipts — SROIE has clusters (many
  receipts from the same shops: Gardenia, Unihakka, MR D.I.Y.…), so intervals are
  slightly optimistic; GT label noise bounds the attainable accuracy. Read as approximate.

**McNemar exact test — "do two methods really differ, on the same receipts?"**
Both methods (or both engines) are scored on the same receipts, so compare *per receipt*
and look only at the **disagreements**: b = receipts A got right and B wrong, c = the
reverse; ties are ignored. If b and c split evenly → no real difference; lopsided
(10 vs 0) → real. "Exact" = the p-value is the exact coin-flip probability of a split at
least this lopsided if each disagreement were 50/50 (`2·Σ_{i≤min(b,c)} C(b+c,i)/2^(b+c)`,
capped at 1), valid even with few disagreements (no normal approximation). p < 0.05 =
unlikely to be chance. Receipts with missing GT are skipped. Example (smoke, n=5):
SmolVLM2 address with outlines lost 3 receipts to 0 → p = 0.25, not enough evidence yet.
Use it for: outlines vs baseline (per model, per column) and ollama vs vLLM (engine
effect). **Do not** compare paired methods by eyeballing overlapping Wilson intervals —
McNemar is sharper because it uses the pairing.
- Multiple comparisons: 4 models × 5 columns × several contrasts = many tests, a few
  p < 0.05 will happen by chance. Report counts (b, c) and effect sizes, don't over-read a
  single p-value; apply Holm if we claim any individual one.

### What each saved result record contains
One JSON per line in `results_outlines_v1.jsonl` / `results_baml_v1.jsonl` (written the
moment it exists; rerun resumes; torn last line ignored).
| field | arm | meaning |
|---|---|---|
| `raw` | both | what the model actually emitted. Outlines arm: baseline = free text (may carry ```json fences), outlines = the constrained JSON. BAML: the text **before** SAP coerced it |
| `pred` | both | the 4 field values parsed from `raw` (read out of fences when needed) |
| `ok` | both | per column (company, date, address, total, date_lenient): True / False / None (= GT missing, excluded) |
| `gt` | both | the SROIE ground truth for that receipt (file is self-contained) |
| `model`, `key`, `time` | both | model (BAML: includes the engine), SROIE receipt id, wall seconds for the generation |
| `energy_j` | both | whole-GPU energy in joules during the generation (NVML counter) |
| `input_tokens`, `output_tokens` | both | prompt+image length and generated length (see caveat) |
| `error` | both | only if the generation crashed (e.g. CUDA OOM); scored as all-wrong, run continues |
| `method` | outlines | `baseline` (prompt only) or `outlines` (constrained) |
| `valid_json`, `valid_schema`, `json_error`, `schema_error` | outlines | structure axis, strict on the raw text |
| `baml_ok`, `valid_schema` | BAML | object produced / passes the Pydantic schema |
Caveats: BAML tokens come from the server's usage report (exact); in the outlines arm
`input_tokens` is computed once per receipt from the chat template (shared by both
methods) and `output_tokens` is the re-tokenized `raw` without EOS (close approximation).
Not saved: the prompt text (it is in the code, versioned by `TAG`), package versions
(`uv.lock`), CPU/RAM energy, d1 judge verdicts (later).

### Ecological side quest — what we measure and what we can derive (not necessarily in the talk)
- **Measured per generation:** `energy_j` from the GPU's own cumulative energy counter
  (NVML, `energy.py`, package `nvidia-ml-py`): exact, no sampling, for the HF models and
  for the ollama/vLLM servers (the client waits while the GPU works). Hardware: NVIDIA
  RTX 4070 Laptop, 8 GiB; idle ≈ 14–15 J per second (~15 W); vLLM smoke ≈ 75 J per
  receipt (≈ 70–90 W while generating). Plus tokens in/out and wall time.
- **Limits to state if shown:** GPU only (no CPU, RAM, PSU, cooling, no training/embodied
  carbon); whole-GPU, so keep the machine otherwise idle (display included); it includes
  the GPU's idle floor (subtract ~15 W × time for a "net" figure — not done); a
  laptop GPU is power-limited, so absolute joules don't transfer to a datacenter GPU.
- **Derivable later (no re-run):** Wh = J / 3600; gCO₂ = kWh × a grid carbon-intensity
  factor (a choice to make and cite); J per correct field; J per 1k output tokens;
  engine comparison (ollama's long prefill costs joules, not just seconds).
- **API comparison:** `ecologits` estimates the impact of *API* models from model + token
  counts → our input/output token counts are exactly what it needs, for a "small local vs
  big closed API" slide. `codecarbon` (RAPL, whole-run CPU+GPU) could cross-check one run;
  not done.
