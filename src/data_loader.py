import csv
from collections import abc
from pathlib import Path
from typing import Generator, Literal

import cv2
import numpy as np
from jamo import h2j, j2h, j2hcj

from codec import Codec
from constants import DATA_DIR, MAX_LABEL_LENGTH, 가능성길이, 공백문자, 한글문자


class DataLoader(abc.Sequence):

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class LanguageDataLoader(DataLoader):
    def __init__(self, data_root_path: Path, codec: Codec, train_ratio: float = 0.9, batch_size: int = 100):
        self.data_root_path = data_root_path
        self.batch_size = batch_size
        self.codec = codec

        with open(data_root_path / "labels.csv", "r") as f:
            data_list = list(csv.reader(f))

        train_data_count = int(len(data_list) * train_ratio)
        self.data = {
            "train": data_list[:train_data_count],
            "test": data_list[train_data_count:],
        }

    def __load_image(self, image_path: Path) -> np.ndarray:
        img: np.ndarray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        img = img[np.newaxis, :, :]
        return img

    def loader(self, type: Literal["train", "test"]) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        주어진 type("train" 또는 "test")에 따라 이미지와 라벨을 numpy 배열로 묶어서 batch 단위로 반환하는 generator입니다.
        """
        data = self.data[type]

        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]

            # image ndarray 생성
            batch_images = [self.__load_image(self.data_root_path / path) for path, _ in batch]
            # label ndarray 생성
            batch_labels = [self.codec.encode(label) for _, label in batch]

            yield np.stack(batch_images), np.stack(batch_labels)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table, box

    from codec import KoreanCodec

    console = Console()
    codec = KoreanCodec(max_length=MAX_LABEL_LENGTH)
    data_loader = LanguageDataLoader(DATA_DIR, codec=codec)

    table = Table(header_style="green", box=box.ROUNDED)
    table.add_column("Description")
    table.add_column("Shape")
    table.add_column("Dtype")

    ndarray = data_loader.codec.encode("안녕하세요")
    table.add_row("codec.encode('안녕하세요')", str(ndarray.shape), str(ndarray.dtype))

    ndarray = data_loader.__load_image(DATA_DIR / "images" / "000" / "001.png")
    table.add_row("load_image('images/000/001.png')", str(ndarray.shape), str(ndarray.dtype))

    train_loader = data_loader.loader("train")
    x, t = next(train_loader)
    table.add_row("train_loader x", str(x.shape), str(x.dtype))
    table.add_row("train_loader t", str(t.shape), str(t.dtype))

    test_loader = data_loader.loader("test")
    x, t = next(test_loader)
    table.add_row("test_loader x", str(x.shape), str(x.dtype))
    table.add_row("test_loader t", str(t.shape), str(t.dtype))

    console.print(table)
