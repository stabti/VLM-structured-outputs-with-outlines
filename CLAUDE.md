# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A study, not a product. It compares ways to **enforce structured JSON output from a
small open-weights VLM** (LFM2.5-VL-1.6B) on one OCR task: extracting typed fields
from a receipt image (`ticket_caisse.jpeg`). The headline finding — kept front of
mind when editing — is that enforcement decides *what happens to a misread value*;
it never makes the value right. `OBJECTIVE.md` and `README.md` hold the full writeup
and results; `slides.html` is the talk (intentionally gitignored/not pushed —
`git status` shows it untracked, leave it that way).

## Environment & commands

uv-managed venv, Python 3.13. `uv sync` installs the exact pins from `uv.lock`
(replaces manual pip). Model weights download from the HF Hub on first run — not
vendored.

```bash
uv sync
uv run main.py            # single end-to-end generation (smoke test)
uv run compare_baml.py    # Family B (needs ollama running, see below)
uv run sroie.py           # SROIE loader/normalizer self-checks (+ GT-only smoke print)
uv run stats.py           # Wilson / McNemar self-checks
uv run analyze_results.py  # receipt-level acc, shop-clustered CIs, error taxonomy
uv run compare_arms.py     # paired McNemar tests between saved result arms (no model calls)
```

`uv run energy.py` checks the GPU energy counter (`energy.py`, NVML via `nvidia-ml-py`;
every result record carries `energy_j`, `input_tokens`, `output_tokens` for the ecology
side quest — keep them when editing the loops).

`compare_*.py` run a 5-receipt smoke test by default; `FULL=1 uv run compare_*.py` runs the
full 361-image test split. Every result is appended to `results_<arm>_<TAG>.jsonl` as it is
produced and a rerun **resumes** (skips finished work) — delete the file or bump `TAG`
when the prompt/schema/generation settings change, so protocols never mix.

There is **no test suite, linter, or build step** — this is experiment code, not a
package. The two `compare_*.py` files are the experiments.

## How the experiments are meant to be run

`compare_outlines.py` and `compare_baml.py` are **cell-scripts** (`# %%` markers),
designed for the VS Code Interactive Window on the `.venv` kernel — run cell by
cell so the model loads once and stays in GPU memory across cells. Running them as
plain scripts works too but reloads the model each time. When editing them, respect
the cell boundaries.

`compare_outlines.py` loads its four HF models **one at a time** (an 8 GiB GPU can't
hold both) and frees the GPU between them (`del`, `gc.collect()`,
`torch.cuda.empty_cache()`). Preserve that pattern.

## The two enforcement families (the core mental model)

The whole study is organized around *where* schema enforcement happens. Both arms
extract into the **same Pydantic `Receipt` schema** (SROIE keys, loose on value
format) and score against the **same SROIE ground truth** via `sroie.field_ok` — that
fairness is the point.

- **Family A — mask in the decoder** (`compare_outlines.py`): HF `transformers` +
  `outlines`. The engine masks logits so only schema-valid tokens sample →
  structure guaranteed by construction. Compares baseline (prompt-only) vs.
  outlines-constrained, across base `LFM2.5-VL-1.6B`, its `-Extract` variant,
  `Qwen3-VL-2B-Instruct` and `SmolVLM2-2.2B-Instruct`. Accuracy is scored with the
  payload read out of fenced replies (`lenient_json`); `valid_json`/`valid_schema` stay
  strict on the raw text.
- **Family B — parse at the client** (`compare_baml.py` + `baml_src/`): BAML
  Schema-Aligned Parsing. The model generates freely; BAML coerces to the schema
  afterward → object almost always produced, but malformed values stay visible.

Scoring in both arms separates two axes deliberately: **structural validity**
(parses + passes strict schema) vs. **field accuracy** (each value matches ground
truth). Keep these separate — collapsing them defeats the experiment.

## BAML arm specifics

BAML talks to LFM2.5-VL served by **ollama** over its OpenAI-compatible HTTP
endpoint — no torch/transformers, no `ollama` Python package needed for this arm.

```bash
ollama pull hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16   # BF16 = no quantization loss
uv run baml-cli generate                              # regenerate baml_client/ after editing baml_src/
uv run compare_baml.py                                # ENGINE=ollama (default)
# Same BAML code on vLLM (~12x faster; separate env ~/envs/vllm, NOT in this project's pins):
VLLM_USE_FLASHINFER_SAMPLER=0 ~/envs/vllm/bin/vllm serve LiquidAI/LFM2.5-VL-1.6B \
  --max-model-len 8192 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.6 \
  --port 8000 > vllm.log 2>&1 &   # 4096 required: default 2048 encoder cache rejects big receipts
ENGINE=vllm uv run compare_baml.py
# vLLM guided decoding arm (same server; add --structured-outputs-config.backend xgrammar|outlines):
BACKEND=xgrammar uv run compare_vllm_guided.py   # FULL=1 for all 361; restart server with `outlines`, BACKEND=outlines for the 2nd run
```
# Same script against ollama (native /api/chat `format` = Pydantic JSON schema, llama.cpp grammar; guided only, ~2.5 h):
ENGINE=ollama uv run compare_vllm_guided.py   # results_ollama_format_v2.jsonl (prompt B); vLLM server must be stopped
Never `pkill -f` and start the server in the same Bash call (the pattern matches the call's own
command line and kills the shell): stop in one call, start in the next.
The vLLM server holds ~4.7 GB of the 8 GiB GPU: stop it before running any HF/outlines
cell. The FlashInfer sampler must stay off (no `nvcc` on this machine).

- Schema and client config live in `baml_src/`. **`baml_client/` is generated and
  gitignored** — never hand-edit it; edit `baml_src/*.baml` then re-run
  `baml-cli generate` (ships inside `baml-py`, no separate install).
- The active client is `CustomOllama` (points at `localhost:11434`, temp 0 = greedy to
  mirror the outlines run). The many other clients in `baml_src/clients.baml`
  (GPT-5, Claude, etc.) are BAML init boilerplate — unused by this study.
- `baml_src/receipt.baml`'s `Receipt` class intentionally **mirrors** the Pydantic
  `Receipt` in `compare_outlines.py`. If you change one schema, change both, or the
  comparison stops being fair.

## Consistency invariants (things that must stay in sync)

When touching the experiments, these are coupled by design:
- The schema — SROIE ground-truth keys `company: str, date: str, address: str,
  total: float` — plus the English field hints in the prompt / BAML `@description` (prompt B: commented schema block, see compare_outlines.py) is
  duplicated across `compare_outlines.py`, `compare_baml.py`, and
  `baml_src/receipt.baml`. It is deliberately loose on value format (no date regex):
  normalization lives in the scorer. `total` is a number on purpose — a `str` total made
  outlines mask the -Extract model into junk (typing artifact).
- Scoring is always `sroie.field_ok` (normalized date/total, Levenshtein-fuzzy
  company/address, `None` = GT missing → drop from denominator); ground truth is the
  SROIE `entities` row, never a per-file constant. CIs/paired tests via `stats.py`.
- Testbed: `SPLIT="test"` (361 receipts), `N_IMAGES`, **greedy, one generation per
  image** (the image is the statistical unit), identical in both `compare_*.py`
  (BAML client `temperature 0`). Change them everywhere or not at all.
- Never materialize the dataset: `iter_samples` decodes one receipt at a time (a list
  of all decoded rows OOM-killed the machine); use `select_columns` for GT-only work.
- The original single-ticket experiment (French Intermarché receipt, n=20, temp 0.4,
  strict date/time regexes) lives in git history and the README numbers.

## Notes

- `utils.py` (`draw_bboxes`) is gitignored adjacent tooling, not part of the study.
- Prompts are in **English** since 2026-11-11 (French was v1; kept in the old results files). Console
  output of the scripts is still French.
