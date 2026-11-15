# Floppy LLM — Sub-1MB Local LLM Proof of Concept

**Goal:** Train a ~360k-parameter GPT on 3,500 Wikipedia history facts and ship a runnable package (C host binary + int8 weights + BPE-512 vocab) that fits in under 1,000,000 bytes, queryable interactively from the command line.

**Architecture:** Python/PyTorch pipeline streams English Wikipedia from Hugging Face, extracts history facts with deterministic heuristics, trains a 2-layer decoder-only GPT (d_model=112, 4 heads, GELU FFN, RoPE, RMSNorm, tied embeddings, BPE-512 sentencepiece tokenizer), quantizes weights to per-tensor absmax int8, and exports a flat `.bin` read by a single-file pure-C inference host. The C host dequantizes on load, runs inference in float32, and prompts the user interactively.

**Components Involved:** data pipeline (`scripts/fetch_facts.py`), tokenizer (`scripts/train_tokenizer.py`), model + trainer (`scripts/train.py`), quantization/export (`scripts/export_quant.py`), host (`host/floppy.c`), budget check (`scripts/check_budget.sh`), tests (`tests/`).

**Base Project Folder:** `/Users/developer/business/ai-lite` — git is initialized here and commits happen after every task.

**Locked decisions:** budget 1,000,000 bytes; 360k params is the TOTAL model including embedding table; dataset `wikimedia/wikipedia` config `20231101.en`; BPE-512 tokenizer; GELU FFN; weights int8 per-tensor absmax.

**Model spec (fixed, verified param math):**
- vocab 512, d_model=112, 2 layers, 4 heads (head_dim 28), FFN dim 448 (4×d, GELU), RoPE (no param), RMSNorm (no bias), tied embedding↔head.
- Params: tied emb 512×112 = 57,344; per layer: attention 4×112² = 50,176, FFN 2×112×448 = 100,352, 2 RMSNorms = 224 → 150,752; 2 layers = 301,504; final RMSNorm = 112; **total = 358,960** (< 360k). Quantized size ≈ 358,960 bytes + (<2KB) vocab + scales/header.

**Dependency manifest:** none exists yet — Task 1 creates `pyproject.toml`; deps: `torch`, `datasets`, `sentencepiece`, `pytest` (installed in a project-local venv, activated via `.venv/bin/`).

## Task Structure

### Task 1: Project scaffold and local venv

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `tests/__init__.py` (empty)

`.pyproject.toml` contains project name `floppy-llm`, Python >=3.10, and dependencies `torch`, `datasets`, `sentencepiece`, `pytest`. `.gitignore` ignores `.venv/`, `data/`, `model/`, `host/floppy`, `__pycache__/`, `.pytest_cache/`.

- [x] **Step 1:** Run `git init` (completed in 3 seconds) in `/Users/developer/business/ai-lite` (folder already has the two MD docs; commit them in this task as the initial commit).
- [x] **Step 2:** Create `.venv` (completed in ~200 seconds incl. dependency install) with `python3 -m venv .venv` and install deps: `.venv/bin/pip install torch datasets sentencepiece pytest`. State-change step — run alone, not bundled with file writing.
- [x] **Step 3:** Write the four scaffold files listed above. (completed in 30 seconds; deviation: requires-python lowered to >=3.9 because system Python is 3.9)
- [x] **Step 4:** Verify: `.venv/bin/python -c "import torch, datasets, sentencepiece; print('ok')"` → prints `ok` (verified;\n  venv install completed in ~3 minutes). If it fails: read the pip error, check whether `torch` wheel for this macOS/arch failed, fix the install command, rerun.
- [x] **Step 5:** Commit all scaffold files plus existing MD docs, message `Project scaffold for floppy LLM PoC`. (completed in 5 seconds)

### Task 2: Sentence tokenizer module + tests

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `floppy_llm/tokenizer.py`, `tests/test_tokenizer.py`

`floppy_llm/tokenizer.py` (new package dir, add `floppy_llm/__init__.py`): wraps sentencepiece. Two functions: `train_tokenizer(corpus_path, model_out, vocab_size=512)` (trains BPE, `model_type=bpe`, `byte_fallback` not needed at 512; `unk_id=0`, `bos_id=1`, `eos_id=2`, `pad_id=-1`) and `class Tok` with `encode(str) -> list[int]`, `decode(list[int]) -> str`, and `vocab_size` property. All special-id constants live as module-level names.

- [x] **Step 1:** Write `floppy_llm/tokenizer.py` with the exact API above. (completed in 60 seconds)
- [x] **Step 2:** Write `tests/test_tokenizer.py`: unit tests that (a) train on a small inline corpus (~200 words repeated), (b) assert `encode`/`decode` round-trips 5 sample strings including years ("The battle took place in 1415.") byte-exactly, (c) assert `vocab_size == 512`, (d) assert `encode` of empty string returns `[]`.
- [x] **Step 3:** Run `.venv/bin/pytest tests/test_tokenizer.py -q` → all pass. (completed in ~10 minutes incl. debugging: sentencepiece silently drops lines >4192 chars and needs enough unique merge material for vocab 512 — test corpus now builds 2000 varied sentences; the real facts corpus is already sentence-per-line so unaffected) If it fails: pytest output names the failing assertion; fix code (not the test) unless the test contradicts this spec.
- [x] **Step 4:** Commit, message `Add BPE-512 sentencepiece tokenizer wrapper`. (completed in 5 seconds)

### Task 3: Wikipedia fact fetcher + tests

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `scripts/fetch_facts.py`, `floppy_llm/fact_filter.py`, `tests/test_fact_filter.py`, `tests/fixtures/sample_article.json`

`floppy_llm/fact_filter.py` contains the pure logic (no network): `is_history_article(title, text) -> bool` (true when title matches year patterns `\b(1|2)\d{3}\b` OR text head contains markers like "was a battle", "was king of", "was a treaty", "dynasty", "empire", "was born", "died", "reign" — case-insensitive, exact list stored as a module constant `HISTORY_MARKERS`), and `extract_facts(text, max_facts_per_article=5) -> list[str]` (split lead section into sentences; keep sentences containing a `\b(1|2)\d{3}\b` year; strip refs/brackets; drop >200-byte sentences).

`scripts/fetch_facts.py`: streams `wikimedia/wikipedia` config `20231101.en` via `datasets.load_dataset(..., streaming=True, split="train")`; applies fact_filter; dedupes by normalized sentence; stops at 3,500 facts; writes `data/facts.txt` (one fact per line) and `data/facts.meta.json` (contributing article titles + Wikipedia URL attribution for CC BY-SA).

- [ ] **Step 1:** Write `tests/fixtures/sample_article.json` — a realistic hand-written fixture: one `{"title": "1415", "text": "Events of 1415 ... Battle of Agincourt ... Henry V ..."}` style article of ~6 short paragraphs covering one history and one non-history case.
- [ ] **Step 2:** Write `floppy_llm/fact_filter.py` per spec above.
- [ ] **Step 3:** Write `tests/test_fact_filter.py`: unit tests against the fixture — `is_history_article` returns True/False for the two cases; `extract_facts` returns only year-bearing, ≤200-byte, bracket-stripped sentences; a sentence with no year is excluded.
- [ ] **Step 4:** Run `.venv/bin/pytest tests/test_fact_filter.py -q` → all pass. If it fails: read the failing assertion, adjust filter constants, rerun.
- [ ] **Step 5:** Write `scripts/fetch_facts.py` per spec.
- [ ] **Step 6:** Run `.venv/bin/python scripts/fetch_facts.py` → `data/facts.txt` has EXACTLY 3,500 lines, each non-empty, and `data/facts.meta.json` parses with a non-empty titles list. This downloads streams a lot — expect minutes. If fewer than 3,500 facts are found after the full stream: loosen `HISTORY_MARKERS` (more markers) before reducing the cap, and record in README.
- [ ] **Step 7:** Commit, message `Add Wikipedia history-fact extraction pipeline (3500 facts)` — commit only code and fixture, `data/` is gitignored.

### Task 4: Tokenizer training on the fact corpus

**Executor:** main session — **Mode:** spec

**Files:**
- Create: nothing new — uses `floppy_llm/tokenizer.py` from Task 2
- Create: `scripts/train_tokenizer.py` (thin CLI: trains on `data/facts.txt`, writes `model/tokenizer.model`, prints vocab size)

- [ ] **Step 1:** Write `scripts/train_tokenizer.py`.
- [ ] **Step 2:** Run `.venv/bin/python scripts/train_tokenizer.py` → `model/tokenizer.model` exists; round-trip sanity: encode+decode of line 1 of `data/facts.txt` is byte-identical. If domain vocab overflows 512 (sentencepiece `Vocabulary size too small` error): raise vocab to 1024, note in README, and re-check the total budget in Task 8 (weights are unaffected; vocab file ~doubles to still-negligible KB).
- [ ] **Step 4:** Commit, message `Train BPE-512 tokenizer on fact corpus`.

### Task 5: Model definition and trainer + tests

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `floppy_llm/model.py`, `scripts/train.py`, `tests/test_model.py`

`floppy_llm/model.py`: `class TinyGPT(nn.Module)` with the fixed spec (vocab 512, d=112, 2 layers, 4 heads, FFN 448, GELU, RoPE, RMSNorm, tied `lm_head.weight = tok_emb.weight`). `count_params(model) -> int` helper counts ALL parameters. `scripts/train.py`: loads `data/facts.txt`, tokenizes with `model/tokenizer.model`, one fact = one document ending in EOS; random-batch training loop, AdamW (lr 3e-3), cosine schedule, context window 128 tokens, batch 64, device = `mps` if available else `cpu`, saves checkpoint to `model/model.pt`; prints loss every 200 steps; default epochs chosen so total steps ≈ 2,000 (tune only if loss plateaus above ~1.0).

- [ ] **Step 1:** Write `floppy_llm/model.py`.
- [ ] **Step 2:** Write `tests/test_model.py`: (a) `count_params(TinyGPT()) == 358,960` — the hard budget test, asserting exact int count; (b) a forward pass on batch shape (2, 16) returns logits shape (2, 16, 512); (c) `model.lm_head.weight is model.tok_emb.weight` (tying actually shared, not just equal-valued).
- [ ] **Step 3:** Run `.venv/bin/pytest tests/test_model.py -q` → all pass. If the param count mismatches: recount per the param math (fix code, never the test).
- [ ] **Step 4:** Write `scripts/train.py`.
- [ ] **Step 5:** Run training to completion: `.venv/bin/python scripts/train.py`. Expected: final loss < 1.0 on this repetitive 3,500-fact corpus (memorization regime), minutes on MPS/CPU. If loss plateaus > 1.0 after 2,000 steps: extend to 5,000 steps, note in README.
- [ ] **Step 6:** Commit, message `TinyGPT model and training script (358,960 params)` — `model/model.pt` is gitignored (large fp32 artefact).

### Task 6: int8 quantization and flat binary export + tests

**Executor:** main session — **Mode:** spec (wire format is risk-routed: the export header writer and the C reader share one spec; a mismatch is the expensive failure, so BOTH sides are validated by the Task 6 round-trip test and the Task 8 golden test)

**Files:**
- Create: `floppy_llm/quant.py`, `scripts/export_quant.py`, `tests/test_quant.py`

**Binary format (single source of truth — `host/floppy.c` must implement exactly this):**
```
Header (fixed, little-endian):
  bytes 0-3   magic u32 = 0x464C5059 ("FLPY")
  bytes 4-7   version u32 = 1
  bytes 8-11  vocab_size u32 (=512)
  bytes 12-15  dims packed: d_model u16=112, n_layer u16=2  -> bytes 12-13 d_model, 14-15 n_layer
  bytes 16-19  n_heads u32 = 4
  bytes 20-23  ffn_dim u32 = 448
  bytes 24-27  context_len u32 = 128
  bytes 28-31  n_tensors u32
Then per tensor: name_len u8, name bytes (ASCII), shape_len u8, shape u32s, scale f32, then raw int8 data.
```
Tensor names (wire format fixed): `tok_emb`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `norm1`, `ffn_fc`, `ffn_proj`, `norm2` (with `_L0`/`_L1` suffix per layer, e.g. `attn_q_L0`), `final_norm`.

`floppy_llm/quant.py`: `quantize_tensor(t: torch.Tensor) -> (int8 ndarray, float32 scale)` using symmetric per-tensor absmax (`scale = max(abs(w))/127`, `q = round(w/scale)`); `dequantize(q, scale) -> ndarray`. `scripts/export_quant.py`: loads `model/model.pt`, quantizes every tensor, writes `model/model.bin` in the format above; prints total bytes of weights and header.

- [ ] **Step 1:** Write `floppy_llm/quant.py`.
- [ ] **Step 2:** Write `tests/test_quant.py`: (a) round-trip: max abs error of `dequantize(quantize(t))` vs `t` ≤ scale (per-tensor bound); (b) quantize of all-zeros tensor yields scale 0 and does not divide-by-zero (handle by setting scale to 1.0 and data to zeros); (c) export → re-parse `model.bin` in Python: magic/version/tensor names/count match this spec, and per-tensor int8 length equals product(shape).
- [ ] **Step 3:** Run `.venv/bin/pytest tests/test_quant.py -q` → all pass. If parsing fails: verify every field offset against the spec table above with a manual hexdump of the first 64 bytes (`xxd model/model.bin | head -4`).
- [ ] **Step 4:** Run `.venv/bin/python scripts/export_quant.py` → prints weights size ≈ 358,960 bytes (<360,000). If over: a tensor is unquantized or double-counted — find which by comparing the printed per-tensor sizes to the param math.
- [ ] **Step 5:** Commit, message `int8 per-tensor quantization and FLPY v1 binary export`.

### Task 7: Single-file C inference host

**Executor:** main session — **Mode:** spec (full C implementation written inline in the main session: wire-format parsing + attention arithmetic is exactly the risk class the skill routes away from delegated execution)

**Files:**
- Create: `host/floppy.c`, `host/Makefile`, `host/vocab.c` (generated: embeds the BPE-512 piece table as C arrays produced by a small Python emitter in `scripts/export_vocab_c.py`)

`host/floppy.c` (pure C99, only stdio/stdlib/string/math): reads `model.bin` path + `vocab.c`-embedded pieces; implements: header/tensor parser exactly per Task 6 format, dequantize-on-load into float32, embedding lookup, RoPE, 4-head causal attention, GELU FFN, RMSNorm, greedy sampling with temperature (arg parse: `-t <temp>`), and an interactive REPL: prints `> ` prompt, reads a line, generates until EOS/`\n`/max 128 new tokens, prints completion. Also supports one-shot: `./floppy model.bin -p "The Battle of Hastings"`. Memory: KV cache for one sequence ≤ 128×112 floats per layer ≈ 115KB total — all static or malloc'd, no fixed huge arrays.

`host/Makefile`: `floppy: floppy.c vocab.c` → `cc -Os -std=c99 -o floppy floppy.c vocab.c -lm`. Embedded vocab instead of a vocab file saves parser code AND bytes.

- [ ] **Step 1:** Write `scripts/export_vocab_c.py` (emits `host/vocab.c`: two arrays `vocab_spieces[]` (string literals) + `vocab_score[]`, exact same ordering sentencepiece uses so ids match) and run it.
- [ ] **Step 2:** Write `host/floppy.c` in one go, implementing the format above.
- [ ] **Step 3:** Write `host/Makefile`.
- [ ] **Step 4:** State-change build step alone: `make -C host` → produces `host/floppy` with no compiler errors/warnings (`-Wall`). If warnings: fix them; if link errors: missing `-lm`.
- [ ] **Step 5:** Commit, message `Single-file C99 inference host with embedded BPE vocab`.

### Task 8: Cross-implementation golden test

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `tests/test_golden.py`, `tests/golden.py` (helper comparing C and Python outputs)

The purpose: the C host and the Python model must produce IDENTICAL logits for the same prompt (this catches every wire-format, RoPE, and quantization mismatch in one assertion — the strongest regression guard for future modifications).

- [ ] **Step 1:** Write `tests/golden.py`: `reference_logits(prompt) -> list[float]` — loads `model/model.bin`, dequantizes, runs the SAME forward pass in pure Python/NumPy (no torch), returns argmax token for the prompt; `c_host_output(prompt) -> list[int]` — runs `host/floppy model/model.bin -p "<prompt>" --logits-debug` mode (Task 7 must expose a hidden flag that prints per-position argmax ids to stderr).
- [ ] **Step 2:** Write `tests/test_golden.py`: integration test — for 3 prompts from distinct facts in `data/facts.txt`, assert the C host's greedy completion CONTAINS the full original fact (memorization check) and the argmax stream matches `reference_logits` exactly (subject to documenting whether ties are possible — if any position differs, dequantized-forward mismatch must be diagnosed tensor by tensor, not relaxed).
- [ ] **Step 3:** Run `.venv/bin/pytest tests/test_golden.py -q` → pass. If it fails: add `FLOPPY_DEBUG` env dumping per-tensor checksums to compare Python vs C sums (sum of floats per tensor) — the first checksum divergence names the broken tensor.
- [ ] **Step 4:** Commit, message `Cross-implementation golden test: C host vs Python reference`.

### Task 9: Size-budget check, LICENSE.txt, full suite

**Executor:** main session — **Mode:** spec

**Files:**
- Create: `scripts/check_budget.py`, `tests/test_budget.py`, `LICENSE.txt`

- [ ] **Step 1:** Write `scripts/check_budget.py`: computes the floppified package size = size(`host/floppy` stripped) + size(`model/model.bin`) (vocab is embedded in the binary, so no separate file) and prints a per-item table + total; exit code 0 if total < 1,000,000, else 1.
- [ ] **Step 2:** Write `tests/test_budget.py`: unit/integration test invoking `scripts/check_budget.py`'s `main()` and asserting `total < 1_000_000` — THE acceptance test of the whole PoC. If it fails: the binary bloat run `strip host/floppy` and reduce `-Os` flags first, weights second (d_model 112→104 → ~312k params, re-run Tasks 5–8; the plan allows this one degradation step, note it in README).
- [ ] **Step 3:** Run full suite: `.venv/bin/pytest tests/ -q` → ALL tests pass in one run.
- [ ] **Step 4:** Run `.venv/bin/python scripts/check_budget.py` → prints table with total <1,000,000 and "FITS ON FLOPPY (1MB)".
- [ ] **Step 5:** Write `LICENSE.txt`: (a) project code (scripts, host, tests) under MIT; (b) a clearly separated "Data and Model Outputs" section stating that `data/facts.txt`, `data/facts.meta.json`, and any generated model outputs derived from Wikipedia text are licensed **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/), with attribution instructions pointing to `facts.meta.json` and the note that share-alike applies to derivative distributions of the corpus or its textual outputs; (c) a line stating that model weights trained on CC BY-SA text are provided as-is and redistributors of the corpus/outputs must comply with CC BY-SA.
- [ ] **Step 6:** Update `README.md` with: run-anywhere instructions (build host, run REPL), the degradation log (any deviations from this plan), and a licensing section mirroring `LICENSE.txt` and pointing to `data/facts.meta.json`.
- [ ] **Step 7:** Commit, message `Budget check: package fits under 1MB, suite green; LICENSE.txt (MIT code, CC BY-SA data/outputs)`.

## Verification summary (all must hold at project end)

1. `.venv/bin/pytest tests/ -q` — entire suite green (tokenizer, fact filter, model, quant, golden, budget).
2. `host/floppy model/model.bin` — interactive REPL; a fact prefix completes with the memorized fact.
3. `scripts/check_budget.py` — binary + weights < 1,000,000 bytes total.
4. Every artefact needed to query the LLM travels as: `host/floppy` (with embedded vocab) + `model/model.bin` — two files, no Python needed at query time.
5. `LICENSE.txt` exists distinguishing MIT (code) from CC BY-SA 4.0 (data/outputs) with attribution instructions referencing `data/facts.meta.json`.

## Risks and their plan-level counters

- **Streaming `en` Wikipedia is a multi-GB stream** — fact filter must be CPU-cheap; stop-early at 3,500 facts avoids the full stream in most runs.
- **512 vocab may overflow on domain text** — Task 4 escalation rule (1024) with budget re-check.
- **Loss may not drop below 1.0** — Task 5 step-lengthen rule.
- **C/Python numeric drift** — golden test with checksum debugging protocol names the divergent tensor.
- **en-config Wikipedia licensing is CC BY-SA-only** — attribution metadata kept in `data/facts.meta.json` and README; distribution guidance documented.
