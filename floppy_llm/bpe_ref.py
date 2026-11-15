"""Pure-Python reimplementation of SentencePiece BPE encoding.

Model: each vocabulary piece has a rank (lower = merged earlier in training).
Encoding a word starts from characters and repeatedly merges the adjacent
pair whose merged piece exists in the vocab with the lowest rank.

This mirrors the algorithm the C host implements; tests verify it matches
sentencepiece's own encoder exactly on the fact corpus.
"""

PIECE_SPACE = "\u2581"  # ▁


class BPEDecoder:
    """Stand-alone BPE encoder identical to the C host's implementation."""

    def __init__(self, pieces):
        # pieces: list of piece strings, index == id, order == merge rank
        self.pieces = pieces
        self.ranks = {}
        for rank, piece in enumerate(pieces):
            if "=" in piece or piece.startswith("<"):
                continue  # control/unk pieces are never merge results
            if len(piece) < 2:
                continue
            self.ranks[piece] = rank
        self.id_of = {p: i for i, p in enumerate(pieces)}

    def encode_word(self, word: str) -> list:
        symbols = list(word)
        while len(symbols) > 1:
            best = None
            best_rank = None
            for i in range(len(symbols) - 1):
                merged = symbols[i] + symbols[i + 1]
                r = self.ranks.get(merged)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = i
            if best is None:
                break
            symbols[best:best + 2] = [symbols[best] + symbols[best + 1]]
        ids = []
        for s in symbols:
            tid = self.id_of.get(s)
            if tid is None:
                # char fallback, then <unk> — matches sentencepiece behaviour
                for ch in s:
                    cid = self.id_of.get(ch)
                    ids.append(cid if cid is not None else self.id_of["<unk>"])
            else:
                ids.append(tid)
        return ids

    def encode(self, text: str) -> list:
        text = " ".join(text.split())  # remove extra whitespace
        if not text:
            return []
        words = text.split(" ")
        out = []
        for w in words:
            piece = PIECE_SPACE + w
            out.extend(self.encode_word(piece))
        return out

    def decode(self, ids: list) -> str:
        text = "".join(self.pieces[i] for i in ids if 0 <= i < len(self.pieces))
        return text.replace(PIECE_SPACE, " ").strip()
