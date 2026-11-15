# Floppy LLM — Sub-1MB GPT

A proof-of-concept ~360k-parameter language model trained on 3,500 Wikipedia
history facts, quantized to int8, served by a single-file C99 host. The complete
runnable package (host binary + weights) fits in under 1,000,000 bytes.

See `PLAN-2026-11-15-floppy-llm.md` for the full implementation plan.

## Layout

- `floppy_llm/` — Python package (tokenizer, fact filter, model, quantization)
- `scripts/` — pipeline entry points
- `host/` — C99 inference host
- `tests/` — pytest suite (unit, integration, golden cross-implementation)

## Quick start

(Steps will be documented as the pipeline lands.)
