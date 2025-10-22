from typing import Annotated

import numpy as np
import typer
from rich.panel import Panel
from rich.progress import track
from rich.table import Table, box

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
    verbose: Annotated[bool, typer.Option(help="상세 로깅 여부")] = False,
) -> None:
    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 학습 과정 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
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

    loss_function = nn.MSELoss()
    optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
    model = KOCRNet()

    for epoch in range(max_epoch):

        train_losses = []
        iter = 0
        for iter, (x, t) in enumerate(train_loader):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32) / UINT8_MAX

            # 모델 학습
            pred = model.forward(x)  # foward 값 계산 (gradient 계산 때 사용)
            loss = loss_function.forward(pred, t)  # 손실 계산
            dout = loss_function.backward()  # 손실 함수 미분
            model.backward(dout)  # gradient 계산
            grads = model.gradient()  # gradient 값 추출
            optimizer.update(model.params, grads)  # 파라미터 update
            train_losses.append(loss)
            console.print(f"Epoch: {epoch+1}, Iter: {iter+1}, Loss: {loss}")

        total_count = len(test_loader)
        correct_count = 0
        for x, t in track(test_loader, description="Testing..."):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32) / UINT8_MAX

            # 모델 평가
            pred = model.forward(x)
            decoded_pred = [codec.decode(pred[i]) for i in pred]
            decoded_t = [codec.decode(t[i]) for i in t]
            correct_count += sum(1 for pred, true in zip(decoded_pred, decoded_t) if pred == true)

        console.print(f"Accuracy: {correct_count / total_count * 100:03f}")
