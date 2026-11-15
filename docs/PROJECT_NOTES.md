# Floppy LLM — Sub-1MB GPT

A proof-of-concept ~359k-parameter language model trained on 3,500 Wikipedia
history facts, quantized to int8, served by a single-file C99 host. The
complete runnable package — host binary (with embedded BPE-512 vocab) plus
quantized weights — is **410,921 bytes: 41% of a 1MB floppy**.

```
host/floppy (stripped, vocab embedded)    51,536 bytes
model/model.bin (int8 weights + scales)  359,385 bytes
TOTAL                                    410,921 bytes   (budget 1,000,000)
```

## Run it

```bash
make -C host
./host/floppy model/model.bin
> Abraham Lincoln
 was an American lawyer, politician, and statesman who served as the 16th
 president of the United States from 1861 until his assassination in 1865.
```

One-shot: `./host/floppy model/model.bin -p "Abraham Lincoln"`
Sampling: `-t 0.8`. No Python, no dependencies at query time — two files.

## Pipeline (rebuild from scratch)

```bash
.venv/bin/python scripts/fetch_facts.py        # stream en.wikipedia -> data/facts.txt (3,500 facts)
.venv/bin/python scripts/train_tokenizer.py    # BPE-512 -> model/tokenizer.model
.venv/bin/python scripts/train.py --steps 5000 # TinyGPT -> model/model.pt (loss ~0.24)
.venv/bin/python scripts/export_quant.py       # int8 export -> model/model.bin
.venv/bin/python scripts/export_vocab_c.py && make -C host  # rebuild host with embedded vocab
.venv/bin/python scripts/check_budget.py       # size budget check
```

## Tests

```bash
.venv/bin/pytest tests/ -q
```

Coverage: tokenizer round-trips, fact-filter heuristics, exact 358,960-param
budget, int8 export round-trip, a golden cross-implementation test (C host
must produce argmax streams identical to an independent NumPy reference over
the same quantized weights), and the 1MB budget acceptance test.

## Model

2-layer decoder-only GPT: d_model=112, 4 heads, GELU FFN (448), RoPE,
RMSNorm, tied embeddings, BPE-512 sentencepiece tokenizer, context 128.
358,960 parameters, exactly — asserted by `tests/test_model.py`.
Weights are stored as int8 with one float32 absmax scale per tensor.

## Deviations and notes

- System Python is 3.9, so `requires-python` is `>=3.9` (plan said 3.10).
- Corpus quality: the history markers are deliberately broad, so the 3,500
  facts include biographical "born/died/award" facts, not only political
  history. Filtering can be tightened in `floppy_llm/fact_filter.py`.
- The C host tokenizer reproduces sentencepiece's BPE pair-merge algorithm
  (verified token-exact on 3,494/3,500 facts) but skips NFKC unicode
  normalization, so exotic characters (e.g. `²`) may tokenize differently
  than training. `byte_fallback` is not enabled; unknown chars map to
  `<unk>`.
- Recall is prefix-dependent: distinctive prompts (60+ chars) recall facts
  verbatim; ambiguous short prefixes can drift to a different fact.
- Training used 5,000 steps (plan's escalation path) to reach loss 0.245.

## Licensing

See `LICENSE.txt`. Code: MIT. The fact corpus derives from Wikipedia
(CC BY-SA 4.0) — attribution metadata in `data/facts.meta.json`; share-alike
applies to the corpus and textual outputs that reproduce it.
