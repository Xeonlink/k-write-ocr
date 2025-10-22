import csv
import math
from collections import abc
from pathlib import Path

import cv2
import numpy as np

from codec import Codec
from constants import DATA_DIR, MAX_LABEL_LENGTH


class DataLoader[T](abc.Sequence[T]):

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> T:
        raise NotImplementedError


class LanguageDataLoader(DataLoader[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, metafile_path: Path, codec: Codec, batch_size: int = 100):
        self.batch_size = batch_size
        self.codec = codec

        with open(metafile_path, "r") as f:
            data_list = list(csv.reader(f))

        self.data_list = data_list

    def load_image(self, image_path: Path) -> np.ndarray:
        """주의: 이미지가 gray scale인 경우에만 작동합니다."""
        img: np.ndarray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        img = img[np.newaxis, :, :]
        return img

    def __len__(self) -> int:
        return math.ceil(len(self.data_list) / self.batch_size)

    def __getitem__(self, index: int):
        batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]

        # image ndarray 생성
        batch_images = np.stack([self.load_image(DATA_DIR / path) for path, _ in batch])
        # label ndarray 생성
        batch_labels = np.stack([self.codec.encode(label) for _, label in batch])

        return batch_images, batch_labels


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table, box

    from codec import KoreanCodec

    # 데이터 로드
    codec = KoreanCodec(max_length=MAX_LABEL_LENGTH)
    train_data_loader = LanguageDataLoader(DATA_DIR / "train_labels.csv", codec=codec)

    IMAGE_PATH = DATA_DIR / "images" / "000" / "001.png"
    TARGETE_LABEL = "안녕하세요"
    label_ndarray = train_data_loader.codec.encode(TARGETE_LABEL)
    image_ndarray = train_data_loader.load_image(IMAGE_PATH)
    x, t = train_data_loader[0]

    # 데이터 출력
    console = Console()

    table = Table(header_style="green", box=box.ROUNDED)
    table.add_column("Description")
    table.add_column("Shape")
    table.add_column("Dtype")

    table.add_row(f"codec.encode('{TARGETE_LABEL}')", str(label_ndarray.shape), str(label_ndarray.dtype))
    table.add_row(f"load_image('{IMAGE_PATH}')", str(image_ndarray.shape), str(image_ndarray.dtype))
    table.add_row("x", str(x.shape), str(x.dtype))
    table.add_row("t", str(t.shape), str(t.dtype))

    console.print(table)
