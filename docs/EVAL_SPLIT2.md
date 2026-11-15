# Split-2 Evaluation: generic model vs split-2 expert

**Headline: the split + synthetic-stub-augmented expert scores 59.0% on
split-2 stub prompts versus 27.0% for the original generic model — an
improvement of +32.0 percentage points (+119% relative),
36 prompt(s) flipped to correct and 4 to incorrect.**

## Method

- Take the 100 sampled split-2 facts (seeded, reproducible) whose
  subjects are single or few words; prompt each model with the subject stub.
- Run via the C hosts (`--logits-debug`-identical `-p` mode, greedy decoding).
- A completion is CORRECT when it reproduces at least the first 70% of the
  original fact sentence verbatim.
- Generic: `host/floppy` + `model/model.bin` (full 3,500-fact training, no
  stub augmentation). Expert: `host/floppy_2of3` + `model/model2of3/model.bin`
  (1,167 split-2 facts + subject-stub synthetic lines).

## Confusion

| | correct | incorrect |
|---|---|---|
| generic | 27 | 73 |
| expert | 59 | 41 |

## Per-prompt results

| # | prompt | generic | expert |
|---|---|---|---|
| 1 | 1481 | ok | ok |
| 2 | Qing Dynasty | x | x |
| 3 | 2002 | ok | ok |
| 4 | Finding Dory | x | ok |
| 5 | Detroit USA | x | ok |
| 6 | Chrysler | x | ok |
| 7 | 1921 | x | x |
| 8 | Local Government Act | x | ok |
| 9 | 1824 | ok | ok |
| 10 | Tony | x | ok |
| 11 | Chessmaster | x | x |
| 12 | Yankees | x | ok |
| 13 | My Opposition the Diaries of | x | ok |
| 14 | Kingdom of Saxony | x | ok |
| 15 | Kingdom of Serbs Croats Slov | x | ok |
| 16 | G | x | x |
| 17 | Chernobyl | x | ok |
| 18 | Crash | x | x |
| 19 | Shortly | ok | x |
| 20 | 1130 | x | x |
| 21 | For | x | x |
| 22 | Varg | x | ok |
| 23 | Bond Girl | x | ok |
| 24 | 1596 | ok | ok |
| 25 | 1909 | x | x |
| 26 | Mr | x | ok |
| 27 | Children | x | x |
| 28 | Pearson | x | ok |
| 29 | Yale | x | ok |
| 30 | Football | x | ok |
| 31 | President | x | x |
| 32 | Amersfoort | x | x |
| 33 | 1884 | ok | ok |
| 34 | Italian Plague | x | x |
| 35 | 1781 | x | x |
| 36 | Albert | x | ok |
| 37 | Chairman | x | ok |
| 38 | Internet Archive | x | x |
| 39 | 1932 | x | x |
| 40 | FV | x | x |
| 41 | Liberal | x | ok |
| 42 | Incredibles | x | x |
| 43 | Kentikian | ok | ok |
| 44 | Jane | x | ok |
| 45 | 1156 | ok | ok |
| 46 | Enabling Act | x | ok |
| 47 | 1973 | x | x |
| 48 | Pope Paul III | ok | ok |
| 49 | Tale | x | ok |
| 50 | US | x | x |
| 51 | Les | x | ok |
| 52 | 1989 | ok | ok |
| 53 | Tsarist | x | ok |
| 54 | Gere | ok | ok |
| 55 | 1367 | ok | ok |
| 56 | 1284 | ok | ok |
| 57 | King Kong | ok | ok |
| 58 | Who | x | x |
| 59 | Bionic | x | x |
| 60 | Tour de France | x | x |
| 61 | BBC | x | x |
| 62 | Colin Powell | x | ok |
| 63 | General Moses | x | x |
| 64 | 2023 | x | x |
| 65 | 1860 | x | x |
| 66 | Lebanon | x | x |
| 67 | 1676 | ok | ok |
| 68 | 1744 | ok | ok |
| 69 | Philadelphia | ok | ok |
| 70 | Korczak Ziolkowski | ok | x |
| 71 | Anne of Cleves | ok | x |
| 72 | Lady Jane Grey | x | x |
| 73 | K | x | x |
| 74 | Calafate | x | ok |
| 75 | Eschelbronn | ok | ok |
| 76 | War | x | x |
| 77 | 1196 | ok | ok |
| 78 | Auschwitz I | x | ok |
| 79 | Syrian | x | ok |
| 80 | 1953 | ok | ok |
| 81 | Annita van Iersel | ok | x |
| 82 | President Barack Obama | ok | ok |
| 83 | Levon Ter Petrosyan | x | x |
| 84 | Ever | x | x |
| 85 | USSR | x | ok |
| 86 | Chiang Kai | ok | ok |
| 87 | Honorary Award | x | ok |
| 88 | 2020 | x | x |
| 89 | Coming | x | x |
| 90 | 1877 | ok | ok |
| 91 | Heath | ok | ok |
| 92 | Margrethe N | x | x |
| 93 | Europe | x | ok |
| 94 | Philip II of France | x | ok |
| 95 | Cardiff | x | ok |
| 96 | Virtues of Harmony | x | x |
| 97 | Haui | x | ok |
| 98 | Carmack | x | x |
| 99 | Searchers | x | ok |
| 100 | Magdalen | x | ok |

## Interpretation

Success measures *stub-prompt retrieval* — the specific capability the
split + prompt-side augmentation operation was designed to add. It does not
measure general question answering or paraphrase robustness (see README
known limitations).
