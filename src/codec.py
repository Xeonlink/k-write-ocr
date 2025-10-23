from typing import Protocol

import numpy as np
from jamo import h2j, j2h, j2hcj

from constants import 가능성길이, 공백문자, 한글문자


class Codec(Protocol):
    def encode(self, label: str) -> np.ndarray:
        raise NotImplementedError

    def decode(self, encoded: np.ndarray) -> str:
        raise NotImplementedError


class KoreanCodec(Codec):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def encode(self, label: str) -> np.ndarray:
        arr = np.zeros((self.max_length, len(한글문자), 가능성길이), dtype=np.uint8)

        label = (label + 공백문자 * self.max_length)[: self.max_length]

        for i, letter in enumerate(label):
            jamo = j2hcj(h2j(letter))
            for j in range(len(한글문자)):
                idx = jamo[j] if len(jamo) > j else 공백문자
                arr[i, j, 한글문자[j].index(idx)] = 1

        return arr

    def decode(self, encoded: np.ndarray) -> str:
        encoded = encoded.reshape((self.max_length, len(한글문자), 가능성길이))
        letters: list[str] = []
        for i in range(self.max_length):
            chars = [한글문자[j][np.argmax(encoded[i, j])] for j in range(len(한글문자))]
            chars = [c for c in chars if c != 공백문자]
            if len(chars) > 1:
                letters.append(j2h(*chars))
            elif len(chars) == 1:
                letters.append(chars[0])
            else:
                letters.append("")
        return "".join(letters)


if __name__ == "__main__":
    print(j2h("ㄱ"))
