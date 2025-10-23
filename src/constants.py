from pathlib import Path

import numpy as np
from rich.console import Console

APP_NAME = "K-Write-OCR"
DATA_DIR = Path("data")
UINT8_MAX = np.iinfo(np.uint8).max
MAX_LABEL_LENGTH = 5
PRECISION = np.float32
TRAIN_DATA_LIMIT = -1  # 음수일 경우 모든 데이터 사용
TEST_DATA_LIMIT = 1  # 음수일 경우 모든 데이터 사용
TEST_PER_ITER = 10

console = Console()

공백문자 = "-"
한글문자 = [
    "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ" + 공백문자,  # 초성 19자 + 공백 = 20자
    "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ" + 공백문자,  # 중성 21자 + 공백 = 22자
    "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ" + 공백문자,  # 종성 27자 + 공백 = 28자
]
가능성길이 = max(len(chars) for chars in 한글문자)


if __name__ == "__main__":
    from rich.table import Table, box

    # 상수 목록 출력
    table = Table(header_style="green", box=box.ROUNDED)
    table.add_column("이름")
    table.add_column("값")

    table.add_row("APP_NAME", APP_NAME)
    table.add_row("DATA_DIR", str(DATA_DIR))
    table.add_row("공백문자", 공백문자)
    table.add_row("한글 초성", 한글문자[0])
    table.add_row("한글 중성", 한글문자[1])
    table.add_row("한글 종성", 한글문자[2])
    table.add_row("가능성길이", str(가능성길이))
    table.add_row("UINT8_MAX", str(UINT8_MAX))

    console.print(table)
