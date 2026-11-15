# Architecture Plan — Sub-1MB Floppy LLM PoC

Companion to `TECHNOLOGY_RESEARCH.md`. This is the architecture sketch for the review step; nothing here is built yet.

## 1. Hard budget

| Item | Bytes |
|---|---|
| Total budget (decision pending) | 1,000,000 (or 1,443,584) |
| int8 weights @ 360k params | 360,000 |
| Runtime binary (C, `-Os`) | ~40,000–90,000 |
| Vocab/merges file (byte-level: none) | 0 |
| Per-tensor float32 scales + header | ~few hundred |
| Headroom / compression experiments | large margin |

The budget is comfortable: even at the strict 1MB goal, weights + binary use under half the space.

## 2. Model architecture (decoder-only GPT, weight-tied, byte-level)

Fixed decisions proposed:
- **Tokenization**: raw UTF-8 bytes, vocab = 256. No vocab file. BOS/EOS as two extra ids (vocab 258) or reserve byte 0 as BOS.
- **Embedding ↔ output head weight tying** — mandatory at this size.
- **Rotary embeddings (RoPE)** — zero stored position table.
- **RMSNorm**, SwiGLU or plain GELU FFN (SwiGLU packs more quality per parameter but ~3 matrices; plain GELU-FFN keeps arithmetic simple for a C host — review decision).
- **int8 symmetric per-tensor absmax quantization** (weights only; activations stay float32 at inference; scales stored as float32, one per tensor).

Candidate parameter splits for ~360k total (excluding embeddings unless noted):

| Config | d_model | Layers | Heads | FFN mult | Body params | Notes |
|---|---|---|---|---|---|---|
| A (fits exactly) | 120 | 2 | 4 | 4×d | ~346k + 30k tied emb ≈ **376k total** | Closest to 360k; deep enough for "X happened in year Y" patterns |
| B (deeper, narrower) | 96 | 3 | 3 | 4×d | ~330k body | 3 layers of context, tighter d |
| C (wider, single) | 256 | 1 | 4 | 4×d | ~330k (12·d² at one layer incl. FFN=4d: attn 4d² + ffn 8d²) | TinyStories showed 1 layer can work; weakest for facts |

Recommendation: **Config A**.

Context length: 256 bytes per sequence (a fact is 10–60 tokens; 256 keeps memory tiny and is enough to condition on several facts).

## 3. Data pipeline

1. **Download**: stream `wikimedia/wikipedia` (config `simple` preferred for licensing cleanliness + simpler prose; fallback `en` with filters) via `datasets` streaming mode — never materialize the full dump on disk.
2. **Filter to history**: deterministic heuristics — article belongs to history categories / infobox types (monarch, battle, treaty, empire, year/timeline articles), or lead sentence contains a year pattern. No ML filtering.
3. **Fact extraction**: from each selected article, take the lead paragraph and split into single sentences; keep sentences containing a 4-digit year or a named entity; cap at **3,500 facts**, deduplicate, normalize whitespace, drop facts longer than ~200 bytes.
4. **Output**: one `facts.txt` (one fact per line, newline-delimited training doc) + `facts.meta.json` (source attribution: list of contributing article titles/URLs — needed for CC BY-SA compliance if we ever distribute the corpus or trained outputs).

Expected corpus size: 3,500 facts × ~60 bytes ≈ 210KB of text — small; the model can genuinely memorize it, which is what a memorization-flavored PoC wants.

## 4. Training pipeline (Python, PyTorch, single script)

- `train.py`: nanoGPT-style: batch random 256-byte windows from `facts.txt`, AdamW, cosine LR, train to convergence (loss target < ~0.5–0.8 given the tiny vocab and repetitive corpus; memorization should drive it very low).
- Checkpoint in fp32; `export.py`: quantize per-tensor to int8, write a flat `.bin`: header (magic, version, dims, scales) + raw int8 tensors — same layout the C host reads.
- Estimated training time: minutes on CPU/MPS.

## 5. Inference host (single C file)

- `floppy.c` (~500 lines): reads header + scales, dequantizes tensors on load (360KB int8 → float in memory ~1.4MB RAM, fine), implements embedding lookup, RoPE, attention, FFN, RMSNorm, greedy/temperature sampling, prints completions given a prompt.
- No third-party libs; build with `cc -Os -static`. 
- `floppy_pack.sh`: compiles, concatenates/checks total size of binary + weights against the budget, prints the byte count — the PoC's acceptance metric.

## 6. Repo layout (for the next planning step)

```
ai-lite/
├── data/            # facts.txt (generated, not committed)
├── scripts/         # fetch_facts.py, train.py, export.py
├── host/           # floppy.c, Makefile
├── model/           # exported model.bin (generated)
├── TECHNOLOGY_RESEARCH.md
└── ARCHITECTURE.md
```

## 7. Success criteria (proposed)

1. Binary + weights total < 1MB (or chosen budget).
2. Given a fact prefix (e.g. "The Battle of Hastings took place in"), model completes with the memorized fact.
3. Training runs end-to-end in one command sequence on this machine.

## 8. Risks

- 360k params at byte level may need long training to drive loss down; mitigations: more epochs (corpus is tiny), wider vocab BPE-512 fallback.
- Over-memorization is a feature here, not overfitting — evaluation criteria must reflect "recall facts", not generalization.
- License: keep the attribution metadata even for a PoC.

---

## Outcome (post-execution addendum, 2026-11-15)

The design above was executed with these resolved decisions and deviations:

- **Tokenizer**: byte-level was replaced by **BPE-512 SentencePiece** (review
  decision d) — better convergence per batch; vocab embedded in the C host.
- **Model**: Config "A" was resized to fit the tied embedding inside the
  360k total: **d_model=112** (not 120), 2 layers, 4 heads, FFN 448 —
  exactly **358,960 parameters**, enforced by a test.
- **Dataset**: execution started on `wikimedia/wikipedia` `20231101.en`,
  later switched to **`20231101.simple`** (Simple English Wikipedia) for
  cleaner sentences and better famous-history coverage.
- **Corpus lifecycle**: the 3,500-fact corpus was later split into three
  topic-pure chunks (`scripts/split_facts.py`), each expanded with
  prompt-side subject-stub augmentation (`scripts/augment_facts.py`),
  producing three expert models (`model/model1of3..3of3`) instead of one.
- **Size**: final package is 410,921 bytes — 41.1% of the 1MB budget; the
  degradation step (d=104) was never needed.
- Training required two correctness fixes along the way (RoPE even/odd
  pairing; small-std init for the tied embedding), documented in the PLAN.
- **Query behavior**: the C host performs case-insensitive prompting — it
  scores first-letter casing variants of the input by the model's own
  next-token confidence and generates from the best-scoring variant,
  so `trafalgar` and `Trafalgar` share one retrieval route.

See `README.md` for current usage and `PLAN-2026-11-15-floppy-llm.md` for
the task-by-task execution record.
