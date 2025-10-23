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
    max_epoch: Annotated[int, typer.Option(help="학습 에폭 수")] = 30,
    batch_size: Annotated[int, typer.Option(help="배치 크기")] = 32,
    patience: Annotated[int, typer.Option(help="Early stopping 인내 횟수")] = 5,
    debug: Annotated[bool, typer.Option(help="디버깅 모드")] = False,
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
                f"- debug: [yellow]{debug}[/]",
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
    train_loader = LanguageDataLoader(
        DATA_DIR / "train_labels.csv",
        codec,
        batch_size,
        max_data_count=batch_size * 3 if debug else None,
    )
    test_loader = LanguageDataLoader(
        DATA_DIR / "test_labels.csv",
        codec,
        batch_size,
        max_data_count=batch_size if debug else None,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    model = KOCRNet(
        input_shape=(1, 260, 660),
        output_shape=(10, 3, 28),
    )

    # Early stopping 변수
    best_accuracy = -1
    patience_counter = 0

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
            console.print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}")

        total_count = len(test_loader)
        correct_count = 0
        for x, t in track(test_loader, description="Testing..."):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32)

            # 모델 평가
            pred = model.forward(x, is_train=False)
            decoded_pred = [codec.decode(pred_) for pred_ in pred]
            decoded_t = [codec.decode(t_) for t_ in t]
            correct_count += sum(1 for pred, true in zip(decoded_pred, decoded_t) if pred == true)

            if debug:
                table = Table(header_style="green", box=box.ROUNDED)
                table.add_column("예측 -> 정답")
                for pred, true in zip(decoded_pred, decoded_t):
                    table.add_row(f"{pred} -> {true}")

                console.print(f"")
                console.print(table)

        accuracy = correct_count / total_count * 100
        console.print(f"Accuracy: {accuracy:03f}")

        # Early Stopping 확인
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            console.print(f"[yellow]patience: {patience_counter} / {patience}[/]")
            if patience_counter >= patience:
                console.print(f"[bold red]Eearly Stopped: max accuracy {best_accuracy:.3f}%[/]")
                break
