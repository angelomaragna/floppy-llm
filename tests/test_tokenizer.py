import os
import tempfile

import pytest

from floppy_llm.tokenizer import Tok, VOCAB_SIZE, train_tokenizer

CORPUS = (
    "The battle took place in 1415. Henry V won the battle of Agincourt. "
    "The treaty was signed in 1215. King John sealed the Magna Carta. "
    "The empire fell in 1453. Constantinople was captured by the Ottomans. "
) * 40

# Diverse filler vocabulary so the BPE has enough merge material to reach 512.
_Filler_WORDS = (
    "harbor veteran lantern meadow orchard pilgrim garrison cavalry "
    "senate dynasty fortress province legion scholar cathedral monastery "
    "plague famine rebellious expedition navigator merchant artisan "
    "chancellor regent sultan emperor pharaoh consul aristocrat bishop "
    "aqueduct catapult citadel township hamlet peninsula archipelago "
    "millsmith blacksmith cooper cobbler weaver dyer mason carpenter "
    "velvet amber granite marble timber iron copper silver bronze "
    "wheat barley rye oats honey spice silk linen wool cotton indigo "
    "justice treachery allegiance heraldry banner sigil crest crown "
    "truce siege raid conquest pilgrimage drought harvest winter summer "
    "chronicle parchment quill ledger decree edict statute doctrine "
).split()

# sentencepiece drops lines longer than 4192 chars and needs enough distinct
# merge material for vocab 512: build varied 4-6 word sentences from the list.
import random as _random  # deterministic seed, no test-order dependence
_rng = _random.Random(1234)
_lines = []
for i in range(2000):
    n = 4 + (i % 3)
    words = [_rng.choice(_Filler_WORDS) for _ in range(n)]
    _lines.append(f"The {' '.join(words)} was recorded in {1000 + (i * 7) % 900}.")
CORPUS += "\n".join(_lines)


@pytest.fixture(scope="module")
def tok():
    with tempfile.TemporaryDirectory() as td:
        corpus = os.path.join(td, "corpus.txt")
        with open(corpus, "w") as f:
            # sentencepiece drops lines longer than 4192 chars, so split
            # the corpus into sentence-per-line form, like facts.txt
            lines = CORPUS.replace(". ", ".\n").split("\n")
            f.write("\n".join(l.strip() for l in lines if l.strip()))
        model_path = os.path.join(td, "tok.model")
        train_tokenizer(corpus, model_path)
        yield Tok(model_path)


SAMPLES = [
    "The battle took place in 1415.",
    "King John sealed the Magna Carta.",
    "The empire fell in 1453.",
    "Constantinople was captured by the Ottomans.",
    "treaty signed 1215",
]


def test_roundtrip(tok):
    for s in SAMPLES:
        ids = tok.encode(s)
        assert tok.decode(ids) == s


def test_vocab_size(tok):
    assert tok.vocab_size == VOCAB_SIZE


def test_empty_string(tok):
    assert tok.encode("") == []


def test_special_ids_exist(tok):
    pieces = [tok.id_to_piece(i) for i in range(3)]
    assert pieces[0] == "<unk>"
    assert pieces[1] == "<s>"
    assert pieces[2] == "</s>"
