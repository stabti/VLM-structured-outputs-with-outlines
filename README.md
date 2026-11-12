# Enforcing structured outputs from small VLMs

Receipt → JSON extraction with small open-weights vision-language models
([LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B), its `-Extract` variant,
Qwen3-VL-2B-Instruct, SmolVLM2-2.2B-Instruct), on the 361 receipts of the SROIE test split.
The question: **what does structured-output enforcement actually buy you?**

## Headline findings

1. **Enforcement decides what happens to a misread value; it never makes the value right.**
   Masking the decoder to the schema takes structural validity to ~100% for models that break
   format (Qwen3-VL: 0 → 100%, SmolVLM2: 0 → 99%), changes nothing for a model that already emits
   clean JSON (LFM base, free vs guided: identical on date and address), and can even cost a
   little accuracy (Qwen address −3 pts, CI excludes 0).
2. **The prompt matters more than the enforcement.** Showing the schema as a commented block
   (one `// description` line per field) instead of an inline `{"field": type (...)}` string
   raised address accuracy from 64% to 89% and strict date from 58% to 70%, on both HF and vLLM
   (paired p ≈ 1e-22). BAML's "advantage" over a hand-written prompt was entirely this block.
3. **The inference engine matters, in one case.** HF transformers and vLLM give the same
   accuracy for the same prompt (no significant difference); ollama (llama.cpp, BF16 GGUF) is
   18× slower, uses ~7× the energy and loses ~20 points on company and ~10 on address.
   The cause is not identified (see future work).
4. **Perception is the bottleneck.** The date is read correctly (lenient match) 97% of the
   time; most strict date failures are "right date plus appended time". The remaining errors
   are misreads that no enforcement fixes.

## Two families of enforcement

- **Family A, enforce in the decoder (token masking).** The engine masks logits so only
  schema-valid tokens can be sampled. Hard guarantee. Here: HF + [`outlines`](https://github.com/dottxt-ai/outlines),
  vLLM structured outputs (xgrammar / outlines backends), ollama's native `format` (llama.cpp grammar).
- **Family B, enforce at the client (parse / coerce).** The model generates freely and a client
  library coerces the output afterward. No hard guarantee. Here: [BAML](https://github.com/BoundaryML/baml)
  (Schema-Aligned Parsing), served by ollama or vLLM.

## Protocol

- **Data:** `jsdnrs/ICDAR2019-SROIE`, test split, 361 receipts; fields company, date, address, total.
- **One greedy generation per image**; the image is the statistical unit. Same Pydantic schema,
  same ground truth (SROIE `entities`), same scorer (`sroie.py`) in every arm.
- **Two axes, never collapsed:** *structural validity* (strict `json.loads` + schema, on the raw
  text) and *field accuracy* (each value vs ground truth: dates normalised to ISO, totals as
  floats, company/address Levenshtein similarity ≥ 0.9). Date is reported strict (field must
  hold only the date) and lenient (the right date appears anywhere in the string).
- **Statistics (`stats.py`, `analyze_results.py`):** Wilson 95% intervals (finite-sample
  uncertainty, not run-to-run variance), exact paired McNemar, and shop-clustered bootstrap
  (receipts of one shop are not independent; 153 shops).
- **Cost:** wall time per receipt, GPU energy per receipt from NVML (`energy.py`, whole GPU),
  input/output tokens. RTX 4070 Laptop, 8 GiB.

## Results (prompt B, 361 receipts, greedy)

Prompt B = instruction + schema shown as a commented block. Accuracy in %.

| Arm | valid JSON / schema | company | date strict | address | total | date lenient | per receipt |
|---|---|---|---|---|---|---|---|
| LFM2.5-VL-1.6B, HF, free | 361 / 360 | 90 | 69 | 89 | 92 | 97 | 1.5 s, 125 J |
| LFM2.5-VL-1.6B, HF + outlines | 361 / 361 | 92 | 69 | 88 | 93 | 97 | 1.7 s, 127 J |
| LFM2.5-VL-1.6B, vLLM, free | 361 / 360 | 91 | 70 | 89 | 92 | 97 | 1.6 s, 124 J |
| LFM2.5-VL-1.6B, vLLM guided (xgrammar) | 361 / 361 | 93 | 70 | 89 | 92 | 97 | 1.4 s, 102 J |
| LFM2.5-VL-1.6B, ollama + `format` | 360 / 360 | 73 | 74 | 79 | 89 | 92 | 28.5 s, 893 J |
| LFM2.5-VL-1.6B, BAML on vLLM | n/a | 92 | 71 | 88 | 91 | 96 | 1.5 s, 120 J |
| LFM2.5-VL-1.6B, BAML on ollama | n/a | 80 | 69 | 77 | 82 | 86 | 25.5 s, 958 J |
| Qwen3-VL-2B, HF, free | 0 / 0 (markdown fences) | 82 | 98 | 91 | 94 | 98 | 3.3 s, 268 J |
| Qwen3-VL-2B, HF + outlines | 361 / 361 | 81 | 97 | 88 | 95 | 98 | 3.8 s, 274 J |
| SmolVLM2-2.2B, HF, free | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 2.1 s, 175 J |
| SmolVLM2-2.2B, HF + outlines | 359 / 359 | 65 | 61 | 58 | 79 | 92 | 2.2 s, 177 J |
| LFM2.5-VL-1.6B-Extract, HF, free, prompt B | 85 / 42 | 17 | 15 | 16 | 13 | 16 | 2.3 s, 188 J |
| LFM2.5-VL-1.6B-Extract, HF + outlines, prompt B | 359 / 359 | 83 | 89 | 79 | 73 | 97 | 1.5 s, 119 J |
| LFM2.5-VL-1.6B-Extract, HF, free, **native prompt** | 360 / 360 | 85 | 80 | 78 | 68 | 97 | n/a |
| LFM2.5-VL-1.6B-Extract, HF + outlines, **native prompt** | 361 / 361 | 85 | 81 | 78 | 68 | 97 | n/a |

Notes: BAML renders its own commented schema block (equivalent to prompt A below), so it is
already on the better format. Qwen's "0 valid" is only the markdown fences; the payload is fine
(accuracy is read from the fenced JSON). vLLM timings are with prefix caching off. Receipt-level
accuracy (all four fields right at once) is in `analysis_B.txt`: ~53% strict / ~73% lenient date
for LFM on HF/vLLM, 42% / 55% on ollama, ~70% for Qwen.

**The `-Extract` model** is the clearest case of "enforcement hides a prompt mismatch". On our
prompt B it is mostly unparseable free (24% valid JSON) and masking "rescues" it to 83/89/79/73;
on its own native prompt (YAML field list in the system turn, image-only user turn) free and
outlines are the same (85/80/78/68 vs 85/81/78/68). Even then it is below the base model on
company, address and total (e.g. total 68 vs 93), better only on the strict date (it rarely appends
the time); lenient date is the same (97). The native-prompt run used the earlier English
protocol (outlines v1 file); the numbers are not affected by the later prompt changes.

### Prompt variants (vLLM, free, 361 receipts)

| Prompt | company | date strict | address | total |
|---|---|---|---|---|
| inline schema (our first prompt) | 91 | 58 | 64 | 91 |
| **A** BAML's exact rendered prompt | 92 | 71 | 87 | 91 |
| **B** our wording + BAML's commented schema block | 93 | 70 | 89 | 92 |
| **C** our wording + inline schema, in BAML's order | 90 | 60 | 61 | 91 |

A reproduces BAML; B gets the full gain; C gets none, so it is the schema block, not the
wording or the image/instruction order.

### What the errors look like (`analysis_B.txt`)

- **date:** ~90% of strict failures on LFM are the right date with the time appended.
- **total:** nearly all failures are a wrong value of the right magnitude (another amount of the
  receipt: subtotal, cash, change), not an unparseable one.
- **address / company:** mostly the right text plus extra lines, or a truncation; company misses
  on ollama are mostly "contains the right name plus extra text".

## Reproduce

```bash
uv sync                                   # exact pins from uv.lock (Python 3.13)
FULL=1 uv run compare_outlines.py         # HF arms: baseline vs outlines, one model at a time
uv run baml-cli generate                  # after editing baml_src/
ENGINE=ollama FULL=1 uv run compare_baml.py   # BAML on ollama (BF16 GGUF)
```

vLLM lives in a separate env (`~/envs/vllm`, not in this project's pins):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 ~/envs/vllm/bin/vllm serve LiquidAI/LFM2.5-VL-1.6B \
  --max-model-len 8192 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.6 \
  --no-enable-prefix-caching --structured-outputs-config.backend xgrammar --port 8000 &
ENGINE=vllm FULL=1 uv run compare_baml.py            # BAML on vLLM
BACKEND=xgrammar FULL=1 uv run compare_vllm_guided.py  # free + guided
ENGINE=ollama FULL=1 uv run compare_vllm_guided.py   # ollama native `format` (vLLM server stopped)
uv run compare_arms.py                    # paired McNemar tests between saved arms
uv run analyze_results.py                 # receipt-level accuracy, clustered CIs, error taxonomy
```

Without `FULL=1` the scripts run a 5-receipt smoke test. Every generation is appended to
`results_<arm>_<TAG>.jsonl` and reruns resume; the tag changes when the prompt or settings change,
so protocols never mix (outlines: v1 French prompt, v2 inline English prompt, v3 prompt B;
vLLM guided / ollama format: v1 inline prompt, v2 prompt B). Old results are kept.
`OBJECTIVE.md` is the running lab notebook (decisions, every result, caveats).

## Files

- `compare_outlines.py`: Family A on HF (baseline vs outlines; 4 models, loaded one at a time).
- `compare_vllm_guided.py`: vLLM free vs guided, the prompt variants, and the ollama `format` arm.
- `compare_baml.py`, `baml_src/`: Family B (BAML), ollama or vLLM; `baml_client/` is generated.
- `sroie.py` (loader, normalisers, scorer), `stats.py` (Wilson, McNemar), `energy.py` (NVML).
- `compare_arms.py`, `analyze_results.py`: analyses on the saved results, no model calls.
- `smoke_d1.py`: smoke test of Liquid's d1-2B decision model (judge experiments are future work).
- `results_*.jsonl`: every raw generation, parsed prediction, score, time, energy, tokens.
- `main.py`, `ticket_caisse.jpeg`: the original single-receipt demo.

## Limitations

- One greedy generation per image: the CIs are finite-sample, not run-to-run variance.
- Shop clustering makes McNemar p-values optimistic; the clustered bootstrap in
  `analysis_B.txt` is the check.
- SROIE is public since 2019 and models may have seen it; ground truth has label noise.
- Accuracy depends on the prompt, and the prompt was not tuned per model (Qwen and SmolVLM2
  are very likely under-prompted). The `-Extract` model needs its native prompt (a YAML field
  list in the system turn), see `OBJECTIVE.md`.
- Early arms used a French prompt; the English reruns replace them in the tables above.

## Future work

- Why llama.cpp loses accuracy on the same model (image preprocessing? token counts).
- Resolution/zoom study: test directly that perception is the bottleneck.
- Error detection: logprob confidence baseline vs Liquid's d1-2B as a yes/no judge.
- Contamination check on fresh receipts, sampling variance at temperature > 0, quantization and
  concurrency, ecology cross-check (codecarbon, ecologits), a larger reference model, an
  on-device agent demo.

## The original single-receipt experiment

The study started as a single French receipt (Intermarché, n=20, temperature 0.4, strict
date/time regexes); its code and numbers are in git history. What it showed, and what the
scaled study then confirmed:

- `outlines` makes schema validity 20/20 by construction but only masks errors: a misread time
  became a plausible-looking fake date as the constraints tightened.
- BAML never failed to produce an object, but left malformed values visible.
- A dedicated `time` field captured the time yet did not stop the model conflating it with the date.
- "Good at JSON" (the Extract model) meant low variance, not accuracy.
