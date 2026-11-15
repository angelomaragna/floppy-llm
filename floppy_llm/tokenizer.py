"""BPE-512 sentencepiece tokenizer wrapper for the floppy LLM."""

import sentencepiece as _spm

UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 512


def train_tokenizer(corpus_path: str, model_out: str, vocab_size: int = VOCAB_SIZE) -> None:
    """Train a BPE tokenizer on the corpus and save it to model_out."""
    _spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_out[:-6] if model_out.endswith(".model") else model_out,
        model_type="bpe",
        vocab_size=vocab_size,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=-1,
        input_sentence_size=100000,
        shuffle_input_sentence=True,
        character_coverage=0.9995,
    )


class Tok:
    """Encode/decode wrapper around a trained sentencepiece model."""

    def __init__(self, model_path: str):
        self._sp = _spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    def encode(self, text: str) -> list:
        return self._sp.encode(text)

    def decode(self, ids: list) -> str:
        return self._sp.decode(ids)

    def id_to_piece(self, id_: int) -> str:
        return self._sp.id_to_piece(id_)
