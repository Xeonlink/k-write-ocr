from typing import Annotated

import numpy as np
import typer
from rich.panel import Panel
from rich.progress import track

from codec import KoreanCodec
from common import nn
from common.optimizer import Adam
from constants import DATA_DIR, MAX_LABEL_LENGTH, UINT8_MAX, console
from data_loader import LanguageDataLoader
from net import KOCRNet

app = typer.Typer(
    # name="train",
    # help="학습 과정을 관리 \n(학습, 검증, 테스트 등)",
    rich_markup_mode="rich",
)


@app.command("train", help="학습을 시작합니다.")
def train(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
    max_epoch: Annotated[int, typer.Option(help="학습 에폭 수")] = 10,
    batch_size: Annotated[int, typer.Option(help="배치 크기")] = 32,
    max_iter: Annotated[int, typer.Option(help="최대 반복 횟수(디버깅용)")] = None,
    # verbose: Annotated[bool, typer.Option(help="상세 로깅 여부")] = False,
) -> None:
    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 학습 과정 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"- 아래의 파라미터로 학습을 시작합니다.",
                f"- max_epoch: [yellow]{max_epoch}[/]",
                f"- batch_size: [yellow]{batch_size}[/]",
                f"- max_iter: [yellow]{'미지정' if max_iter is None else max_iter}[/]",
            ]
        )
        panel = Panel(
            panel_content,
            title="Train",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 학습 시작
    codec = KoreanCodec(max_length=MAX_LABEL_LENGTH)
    train_loader = LanguageDataLoader(DATA_DIR / "train_labels.csv", codec, batch_size)
    test_loader = LanguageDataLoader(DATA_DIR / "test_labels.csv", codec, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    model = KOCRNet(
        input_shape=(batch_size, 1, 260, 660),
        output_shape=(batch_size, 10, 3, 28),
    )

    iter_counter = 0
    for epoch in range(max_epoch):

        train_losses = []
        for i, (x, t) in enumerate(train_loader):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32)

            # 모델 학습
            B, L, M, S = t.shape  # B: batch size, L: max label length, M: letter member, S: character set size
            pred = model.forward(x)  # foward 값 계산 (gradient 계산 때 사용)
            pred = pred.reshape(B * L * M, S)
            t = t.reshape(B * L * M, S)
            loss = criterion.forward(pred, t)  # 손실 계산
            dout = criterion.backward()  # gradient 계산
            model.backward(dout)  # gradient 계산
            grads = model.gradient()  # gradient 값 추출
            optimizer.update(model.params, grads)  # 파라미터 update
            train_losses.append(loss)
            iter_counter += 1
            console.print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}")

            # 최대 반복 횟수 도달 시 종료 (Debug용)
            if max_iter is not None and iter_counter >= max_iter:
                console.print(f"[red]Max iteration reached[/]")
                return

        total_count = len(test_loader)
        correct_count = 0
        for x, t in track(test_loader, description="Testing..."):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32)

            # 모델 평가
            pred = model.forward(x)
            decoded_pred = [codec.decode(pred[i]) for i in pred]
            decoded_t = [codec.decode(t[i]) for i in t]
            correct_count += sum(1 for pred, true in zip(decoded_pred, decoded_t) if pred == true)

        console.print(f"Accuracy: {correct_count / total_count * 100:03f}")
