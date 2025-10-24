from typing import Protocol

import numpy as np
from jamo import h2j, j2h, j2hcj

from constants import 가능성길이, 공백문자, 한글문자


class Codec(Protocol):
    def encode(self, label: str) -> np.ndarray:
        raise NotImplementedError

    def decode2단어(self, encoded: np.ndarray) -> str:
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

    def decode2단어(self, encoded: np.ndarray) -> str:
        초성중성종성 = len(한글문자)
        encoded = encoded.reshape((self.max_length, 초성중성종성, 가능성길이))
        철자들: list[str] = []
        for i in range(self.max_length):
            자소들: list[str] = []

            for j in range(초성중성종성):
                자소위치 = np.argmax(encoded[i, j])
                if len(한글문자[j]) < 자소위치:
                    자소들.append(공백문자)
                else:
                    자소들.append(한글문자[j][자소위치])

            if 자소들[0] != 공백문자 and 자소들[1] != 공백문자 and 자소들[2] != 공백문자:
                철자들.append(j2h(*자소들))
            elif 자소들[0] != 공백문자 and 자소들[1] != 공백문자 and 자소들[2] == 공백문자:
                철자들.append(j2h(자소들[0], 자소들[1]))
            else:
                철자들.append(공백문자)

        return "".join(철자들)

    def decode2자소나열(self, encoded: np.ndarray) -> str:
        초성중성종성 = len(한글문자)
        encoded = encoded.reshape((self.max_length, 초성중성종성, 가능성길이))
        자소들: list[str] = []
        for i in range(self.max_length):

            for j in range(초성중성종성):
                자소위치 = np.argmax(encoded[i, j])
                if len(한글문자[j]) <= 자소위치:
                    자소들.append(공백문자)
                else:
                    자소들.append(한글문자[j][자소위치])

        return "".join(자소들)


if __name__ == "__main__":
    print(j2h("ㄱ"))
