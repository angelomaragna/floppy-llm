# Floppy LLM

> A ~359k-parameter LLM trained on 3,500 Wikipedia history facts, runnable from a package under 1MB — a floppy disk.

## Description

Floppy LLM is a proof of concept that a tiny GPT can be trained from scratch
on public data and shipped as a self-contained artifact that fits in under
1,000,000 bytes: a single-file C99 inference binary (with the BPE-512
vocabulary embedded) plus int8-quantized weights, with no Python or other
dependencies needed to query the model.


## Observations

In the first run the model obtained (modelmodel.bin) learned each fact as
one fixed string-continuation, not as knowledge; trained on single-view
declarative facts, it only responds to exact corpus prefixes. Paraphrased or
differently-structured queries find no route, so it hallucinates —
underperforming simple search.

The issue is clear, **topics are too sparse**. An option is to increase the
dataset with synthetic data. Splitting balances per-topic capacity however
doesn't change how facts are learned. Wrapper augmentation prepends phrases
("Historically,") to whole sentences, so the fact still starts with
"The Battle of Trafalgar" — creating zero prompts starting "Trafalgar".
Bare subject prompts remain unseen.

Improvement comes from **synthetic entries** where the subject appears alone as
prompt, so the model learns stub→fact associations instead of only sentence
continuations. Emitting these mechanically from our existing subject
extraction creates the missing retrieval routes for bare prompts. Examples:

```
 1. Trafalgar. The Battle of Trafalgar was a sea battle fought on 21 October 1805...
 2. Battle of Trafalgar: a sea battle fought on 21 October 1805 between the navies...
 3. trafalgar. The Battle of Trafalgar was a sea battle fought on 21 October 1805...
```

Prompt casing has also been improved at inference level: the host scores first-letter
uppercase/lowercase variants of the input and generates from whichever the model
is most confident about.

Total package: **410,921 bytes (41% of the 1MB budget)**.

## Results

The split + prompt-side augmentation operation improved stub-prompt fact
recall on held-out split-2 probes from **27% (generic model) to 59%
(split-2 expert) — a gain of 32 percentage points (2.2x)**, measured over
100 reproducible subject-stub prompts. Method, confusion table, and
per-prompt results: [`docs/EVAL_SPLIT2.md`](docs/EVAL_SPLIT2.md).

## Features

- 358,960-parameter decoder-only GPT (2 layers, d_model 112, 4 heads, GELU
  FFN, RoPE, RMSNorm, tied embeddings)
- BPE-512 SentencePiece tokenizer reproduced exactly in C (pair-merge
  algorithm, verified token-exact against SentencePiece)
- int8 symmetric per-tensor absmax weight quantization in a custom flat
  binary format (FLPY v1)
- Interactive REPL and one-shot generation with temperature sampling
- Prompt-side data augmentation: subject-stub lines (`Trafalgar. <fact>`)
  that create retrieval routes for short prompts
- Case-insensitive prompting: the host scores first-letter casing variants
  of the input by model confidence and generates from the best
  (`trafalgar` routes to the trained `Trafalgar` stub)
- Topic-pure 3-way corpus split and per-topic "expert" models
  (model1of3/2of3/3of3), each trained on its own split
- Debugging: `--logits-debug` per-token probability trace with an
  ANCHORED/DRIFTING verdict, plus `scripts/debug_prompt.py` for
  tokenization, top-k distributions, and corpus proximity
- Full rebuild pipeline from raw Wikipedia stream to shipped binary

## Requirements

- macOS or Linux with a C99 compiler (`cc`) and `make`
- Python 3.9+ with a virtual environment for the training pipeline only
  (inference needs no Python)
- For training on GPU: Apple Silicon with Metal (MPS); falls back to CPU

## Installation

```bash
git clone https://github.com/angelomaragna/floppy-llm.git && cd floppy-llm
python3 -m venv .venv
.venv/bin/pip install torch datasets sentencepiece pytest
```

## Usage

Query the default single model (two files, zero dependencies at runtime):

```bash
make -C host
./host/floppy model/model.bin
> Napoleon Bonaparte
 was a French politician and army leader who ruled France from 1799 to 1814
 and for a short period in 1815.
```

One-shot and sampling: `-p "prompt"`, `-t 0.8`.
Debug a prompt (token-level confidence):

```bash
./host/floppy model/model.bin --logits-debug
.venv/bin/python scripts/debug_prompt.py "Trafalgar" --topk 6
```

### Three-expert pipeline

```bash
# 1. fetch corpus (simple-English Wikipedia; --config 20231101.en for full)
.venv/bin/python scripts/fetch_facts.py --config 20231101.simple

# 2. split into 3 topic-pure chunks, then augment each with prompt-side stubs
.venv/bin/python scripts/split_facts.py
for i in 1 2 3; do .venv/bin/python scripts/augment_facts.py data/facts_split$i.txt; done

# 3. train expert N (repeat for 1, 2, 3)
.venv/bin/python scripts/train_tokenizer.py --data data/facts_split1_augmented.txt --out-dir model/model1of3
.venv/bin/python scripts/train.py           --data data/facts_split1_augmented.txt --out-dir model/model1of3
.venv/bin/python scripts/export_quant.py    --ckpt model/model1of3/model.pt --out model/model1of3/model.bin
.venv/bin/python scripts/export_vocab_c.py --tok model/model1of3/tokenizer.model --out host/vocab_1of3.c
cc -Os -std=c99 -o host/floppy_1of3 host/floppy.c host/vocab_1of3.c -lm

# 4. query the expert that owns the topic
./host/floppy_2of3 model/model2of3/model.bin -p "Trafalgar"
```

## Tests

```bash
.venv/bin/pytest tests/ -q   # 25 tests, ~5s
```

Unit tests for the tokenizer, fact filter, exact parameter budget, and
quantization round-trip; a golden cross-implementation test (C host vs
NumPy reference over the same quantized weights); and the 1MB budget
acceptance test.

## Project structure

```
floppy_llm/    Python package: tokenizer, fact filter, model, quant, binfmt
scripts/       Pipeline: fetch, split, augment, train, export, debug, budget
host/          C99 inference host (floppy.c + generated vocab.c per model)
tests/         pytest suite incl. golden cross-implementation test
docs/          Working notes (PROJECT_NOTES.md)
data/          Generated corpora and splits (not committed)
model/         Generated tokenizers, checkpoints, int8 binaries (not committed)
```

## Known limitations

- The model completes corpus-like prompts; prompts that are trained
  subject stubs (`Trafalgar`, `Napoleon Bonaparte`) recall their facts
  verbatim, other phrasings drift
- Multi-word names are not stubs (`Admiral Nelson` is not a stub;
  `Admiral` is), so two-word prompts do not route; single-word stubs work
  in either casing thanks to the host's casing-variant selection
- Recall is per-expert: each 1of3/2of3/3of3 model answers only its own
  third of the corpus; there is no router between them
- The C tokenizer skips NFKC normalization, so exotic characters
  (`²`, accented letters) may tokenize differently than training
- Year tokens fragment (`194` + `5`), costing generation steps

## Contributing

Small, self-contained changes with tests preferred; run the full suite
before committing. Detailed execution history and decisions are recorded
in `PLAN-2026-11-15-floppy-llm.md` (task-by-task) and
`docs/PROJECT_NOTES.md`.

## License

Code is MIT; the fact corpus and textual outputs derived from Wikipedia
are CC BY-SA 4.0. See `LICENSE.txt` for the full terms and attribution
requirements (`data/facts.meta.json` records contributing articles).

## Credits

- Dataset: Simple English Wikipedia via `wikimedia/wikipedia` (Hugging Face)
- Architectural references: TinyStories (Microsoft Research),
  nanoGPT and llama2.c (Andrej Karpathy)
