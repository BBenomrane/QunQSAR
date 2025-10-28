from __future__ import annotations

from typing import Iterable, List, Sequence


class ByteTokenizer:
    """A minimal byte-level tokenizer.

    - Vocabulary is the 256 possible byte values [0, 255].
    - Encoding converts UTF-8 text to its raw bytes (one token per byte).
    - Decoding converts tokens back to bytes and decodes as UTF-8.

    This tokenizer requires no training and is robust for character-level
    (byte-level) language modeling on arbitrary text corpora.
    """

    def __init__(self) -> None:
        self._vocab_size: int = 256

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of byte IDs.

        The text is encoded in UTF-8 and each byte becomes one token ID in [0, 255].
        """
        byte_values = text.encode("utf-8", errors="strict")
        return list(byte_values)

    def decode(self, ids: Sequence[int] | Iterable[int]) -> str:
        """Decode a sequence of byte IDs back into a string.

        Invalid UTF-8 byte sequences are replaced to avoid decode errors.
        """
        if not isinstance(ids, (list, tuple)):
            ids = list(ids)
        return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")
