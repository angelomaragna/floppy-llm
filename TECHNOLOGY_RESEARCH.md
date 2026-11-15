# Technology Research — Sub-1MB Floppy LLM PoC

Goal: train a ~360k-parameter LLM on ~3,500 Wikipedia historical facts, export weights as int8, and ship a program + weights that fit inside 1MB (1,000,000 bytes; decide whether the stretch goal is the true 1,443,584-byte floppy capacity).

## 1. What the research shows is achievable

- **TinyStories (Microsoft, arXiv:2305.07759)** demonstrates that models with **under 10M parameters — even a single transformer block — can produce coherent English** when trained on simple, narrow-domain text. This is the key precedent: a 360k model will not be "smart", but on a narrow corpus (historical facts in simple sentences) it can memorize/reproduce factual patterns. TinyStories models at 1M/3M/9M params are on Hugging Face as reference points.
- **nanoGPT (Karpathy)** shows a GPT-2-style model can be a single ~300-line PyTorch file, trained on a ~1MB text corpus in tens of minutes on a laptop. Good skeleton for the trainer, but we must shrink config far below its defaults (his "babystyle" config already targets ~120k–10M params).
- **llama2.c (Karpathy)** is the strongest reference for the runtime side: **a single-file pure-C transformer inference engine** with no dependencies, compiling to a tiny binary, loading raw `.bin` weight files. He also trained tokenizer-free / small-vocab (vocab 512–4096 sentencepiece) models from scratch (his "baby llama" line, ASCII-only even). Our host program will be modeled on this approach but written from scratch and int8-native.

## 2. Technology choices to review

### Training side (Python)
| Option | Pros | Cons | Verdict |
|---|---|---|---|
| PyTorch (vanilla) | Full control, known-good, tiny model trains on CPU/MPS in minutes | Heavy install | **Recommended** — write a nanoGPT-style trainer ourselves, single file |
| tinygrad | Light, cute | Smaller ecosystem, portability quirks | Fallback |
| PyTorch + HuggingFace `transformers Trainer` | Batteries included | Massive overkill, config overhead | Rejected |

### Dataset (Hugging Face)
| Option | Notes |
|---|---|
| `wikimedia/wikipedia` (20231101 config) | Official dumps; only the `simple` English config is licensed CC BY-SA **and** GFDL — the plain `en` config is CC BY-SA only. Large download but we stream/filter. |
| `wikimedia/structured-wikipedia` | Per-article structured fields; convenient for extracting short "fact-like" lead sentences. Same licensing situation (dual-licensed where applicable). |
| `olm/wikipedia` | Preprocessed Wikipedia snapshots for LM training (2023/2024 releases). |

Plan: stream one of these, filter/filter to **history-related articles**, then compress each article's lead paragraph into one-line facts, keeping exactly ~3,500. **Note: rather than ML-based filtering, prefer deterministic heuristics** (category/infobox markers containing "history", years present, etc.) so the pipeline is reproducible. Any distributed corpus must carry CC BY-SA attribution.

### Tokenizer — the critical size constraint
| Option | Vocab size | Stored cost | Verdict |
|---|---|---|---|
| Byte-level (chars) | 256 | Zero extra storage (runtime is ASCII/UTF-8 aware) | **Recommended starting point** — model reads raw bytes, no vocab file at all |
| Tiny BPE (sentencepiece) | 512–2048 | KBs of vocab data + training complexity | Stretch option if byte-level underfits |
| GPT-2/LLaMA tokenizer | 50k+ | Way too big | Rejected |

At 360k params, embedding table dominates the budget. With byte-level vocab (256), the embedding + head can share weights (weight tying), leaving most parameters for the transformer body.

### Weight storage
- int8 per-weight symmetric quantization: store `round(w / scale)` with one float32 scale **per tensor** (absmax). 360k params = **360,000 bytes** raw; minus the tied embedding (if the 360k is total including embeddings, the math must be done in the architecture doc).
- No compression on weights initially; zlib/lzma can be evaluated later if we need extra room within the size budget.

### Inference host language
| Option | Binary size | Verdict |
|---|---|---|
| C (single .c file, no libc beyond stdio/math) | A llama2.c-style engine compiles to roughly tens of KB; with `-Os` and static linking it can be ~50–100KB | **Recommended** |
| Rust | Similar size possible but heavier toolchain story for "fits on a floppy" romanticism | Fallback |
| Assembly | Maximum cred, slower to write | Rejected for PoC |

## 3. Open decisions for the review step
1. **Size budget**: literal 1,000,000 bytes vs. true floppy capacity 1,443,584 bytes (1.44MB).
2. **Whether "360k parameters" includes the embedding table** or only the transformer body (drastically changes architecture numbers).
3. Dataset config choice (`wikimedia/wikipedia` `simple` vs `en` filtered; `olm/wikipedia`) and the fact-extraction heuristic.
4. Tokenizer: byte-level vs tiny BPE (512/1024).
5. Architecture family: decoder-only GPT (recommended) vs other minimal designs.
6. Evaluation criteria for "proof of concept" success (reproduces memorized facts? sensible completions? loss threshold?).
