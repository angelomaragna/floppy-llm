# Floppy LLM

> A ~359k-parameter LLM trained on 3,500 Wikipedia history facts, runnable from a package under 1MB — a floppy disk.

## Description

Floppy LLM is a proof of concept that a tiny GPT can be trained from scratch
on public data and shipped as a self-contained artifact that fits in under
1,000,000 bytes: a single-file C99 inference binary (with the BPE-512
vocabulary embedded) plus int8-quantized weights, with no Python or other
dependencies needed to query the model.


## Observations

The model learned each fact as one fixed string-continuation, not as knowledge;
trained on single-view declarative facts, it only responds to exact corpus
prefixes. Paraphrased or differently-structured queries find no route, so it
hallucinates — underperforming simple search.

Total package: **410,921 bytes (41% of the 1MB budget)**.

## Features

- 358,960-parameter decoder-only GPT (2 layers, d_model 112, 4 heads, GELU
  FFN, RoPE, RMSNorm, tied embeddings)
- BPE-512 SentencePiece tokenizer reproduced exactly in C (pair-merge
  algorithm, verified token-exact against SentencePiece)
- int8 symmetric per-tensor absmax weight quantization in a custom flat
  binary format (FLPY v1)
- Interactive REPL and one-shot generation with temperature sampling
- Full rebuild pipeline from raw Wikipedia stream to shipped binary
- Golden cross-implementation test: the C host must produce argmax streams
  identical to an independent NumPy reference over the same weights

## Requirements

- macOS or Linux with a C99 compiler (`cc`) and `make`
- Python 3.9+ with a virtual environment for the training pipeline only
  (inference needs no Python)
- For training on GPU: Apple Silicon with Metal (MPS); falls back to CPU

## Installation

```bash
git clone <this-repo> && cd ai-lite
python3 -m venv .venv
.venv/bin/pip install torch datasets sentencepiece pytest
```

## Usage

Build and chat (two files, zero dependencies at runtime):

```bash
make -C host
./host/floppy model/model.bin
> Abraham Lincoln
 was an American lawyer, politician, and statesman who served as the 16th
 president of the United States from 1861 until his assassination in 1865.
```

One-shot completion and sampling:

```bash
./host/floppy model/model.bin -p "Abraham Lincoln"
./host/floppy model/model.bin -p "She debuted in February 2003" -t 0.8
```

Rebuild the model from scratch (fetch → tokenize → train → quantize → embed):

```bash
.venv/bin/python scripts/fetch_facts.py         # 3,500 facts from Wikipedia
.venv/bin/python scripts/train_tokenizer.py      # BPE-512 tokenizer
.venv/bin/python scripts/train.py --steps 5000   # ~70s on Apple Silicon
.venv/bin/python scripts/export_quant.py         # int8 FLPY v1 binary
.venv/bin/python scripts/export_vocab_c.py       # embed vocab into C
make -C host
```

## Tests

```bash
.venv/bin/pytest tests/ -q   # 25 tests, ~5s
```

Unit tests for the tokenizer, fact filter, exact parameter budget, and
quantization round-trip; a cross-implementation golden test (C host vs NumPy
reference); and the 1MB budget acceptance test.

## Project structure

```
floppy_llm/    Python package: tokenizer, fact filter, model, quant, binfmt
scripts/       Pipeline entry points (fetch, train, export, budget check)
host/          C99 inference host (floppy.c + generated vocab.c)
tests/         pytest suite incl. golden cross-implementation test
data/          Generated: facts.txt + attribution metadata (not committed)
model/         Generated: tokenizer, checkpoints, model.bin (not committed)
docs/          Working notes (PROJECT_NOTES.md)
```

## Known limitations

- Recall is prefix-dependent: distinctive prompts (60+ characters) recall
  facts verbatim; ambiguous short prompts can drift to a different fact
- The corpus is "history" by broad markers, so it includes biographical
  born/died/award facts
- Exotic Unicode characters (e.g. `²`) are not NFKC-normalized in the C
  tokenizer and map to `<unk>`

## Contributing

Small, self-contained changes with tests preferred; run the full suite before
committing. Detailed execution history, decisions and deviations are recorded
in `PLAN-2026-11-15-floppy-llm.md` (task-by-task) and `docs/PROJECT_NOTES.md`.

## License

Code is MIT; the fact corpus and textual outputs derived from Wikipedia are
CC BY-SA 4.0. See `LICENSE.txt` for the full terms and attribution
requirements (`data/facts.meta.json` records contributing articles).

## Credits

- Dataset: English Wikipedia via `wikimedia/wikipedia` (Hugging Face)
- Architectural references: TinyStories (Microsoft Research),
  nanoGPT and llama2.c (Andrej Karpathy)
