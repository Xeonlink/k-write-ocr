from typing import Annotated

import numpy as np
import typer
from rich.panel import Panel
from rich.progress import track
from rich.table import Table, box

from codec import KoreanCodec
from common import nn
from common.optimizer import Adam
from constants import (
    DATA_DIR,
    MAX_LABEL_LENGTH,
    PRECISION,
    TEST_DATA_LIMIT,
    TEST_PER_ITER,
    TRAIN_DATA_LIMIT,
    UINT8_MAX,
    console,
)
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
    patience: Annotated[int, typer.Option(help="Early stopping 인내 횟수")] = None,
    debug: Annotated[bool, typer.Option(help="디버깅 모드")] = False,
    # verbose: Annotated[bool, typer.Option(help="상세 로깅 여부")] = False,
) -> None:
    # 파라미터 검증
    if patience is not None and patience <= 0:
        console.print(f"[red]--patience 는 0보다 큰 값이어야 합니다.[/]")
        return

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
        max_data_count=batch_size * TRAIN_DATA_LIMIT if debug and TRAIN_DATA_LIMIT >= 0 else None,
    )
    test_loader = LanguageDataLoader(
        DATA_DIR / "test_labels.csv",
        codec,
        batch_size,
        max_data_count=batch_size * TEST_DATA_LIMIT if debug and TEST_DATA_LIMIT >= 0 else None,
    )

    criterion = nn.SoftmaxWithLoss()
    optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
    model = KOCRNet(
        input_shape=(1, 260, 660),
        output_shape=(MAX_LABEL_LENGTH, 3, 28),
    )

    # Early stopping 변수
    best_accuracy = -1
    patience_counter = 0

    for epoch in range(max_epoch):

        train_losses = []
        for i, (x, t) in enumerate(train_loader):
            # 0~1 범위로 정규화
            x = np.clip(x.astype(PRECISION) / UINT8_MAX, 0, 1)
            t = np.clip(t.astype(PRECISION), 0, 1)

            # 모델 학습
            B, L, M, S = t.shape
            pred = model.forward(x)  # foward 값 계산 (gradient 계산 때 사용)
            pred = pred.reshape(B * L * M, S)
            t = t.reshape(B * L * M, S)
            loss = criterion.forward(pred, t)  # 손실 계산
            dout = criterion.backward()  # gradient 계산
            model.backward(dout)  # gradient 계산
            grads = model.gradient()  # gradient 추출
            optimizer.update(model.params, grads)  # 파라미터 update
            train_losses.append(loss)
            console.print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}")

            if debug and (epoch + (i + 1)) % TEST_PER_ITER == 0:
                total_count = len(test_loader)
                correct_count = 0
                preds = []
                ts = []
                for x, t in track(test_loader, description="Testing..."):
                    # 0~1 범위로 정규화
                    x = np.clip(x.astype(PRECISION) / UINT8_MAX, 0, 1)
                    t = np.clip(t.astype(PRECISION), 0, 1)

                    # 모델 평가
                    B, L, M, S = t.shape
                    pred = model.forward(x, is_train=False)
                    decoded_pred = [codec.decode2자소나열(pred_) for pred_ in pred]
                    decoded_t = [codec.decode2자소나열(t_) for t_ in t]
                    correct_count += sum(1 for pred, true in zip(decoded_pred, decoded_t) if pred == true)
                    preds.append(pred)
                    ts.append(t)

                if debug:
                    table = Table(header_style="green", box=box.ROUNDED)
                    table.add_column("예측 자소나열")
                    table.add_column("예측 단어")
                    table.add_column("정답 단어")
                    for pred, t in zip(preds, ts):
                        for pred_, t_ in zip(pred, t):
                            table.add_row(codec.decode2자소나열(pred_), codec.decode2단어(pred_), codec.decode2단어(t_))

                    console.print(table)

                accuracy = correct_count / total_count * 100
                console.print(f"Accuracy: {accuracy:03f}")

        total_count = len(test_loader)
        correct_count = 0
        preds = []
        ts = []
        for x, t in track(test_loader, description="Testing..."):
            # 0~1 범위로 정규화
            x = np.clip(x.astype(PRECISION) / UINT8_MAX, 0, 1)
            t = np.clip(t.astype(PRECISION), 0, 1)

            # 모델 평가
            B, L, M, S = t.shape
            pred = model.forward(x, is_train=False)
            decoded_pred = [codec.decode2자소나열(pred_) for pred_ in pred]
            decoded_t = [codec.decode2자소나열(t_) for t_ in t]
            correct_count += sum(1 for pred, true in zip(decoded_pred, decoded_t) if pred == true)
            preds.append(pred)
            ts.append(t)

        if debug:
            table = Table(header_style="green", box=box.ROUNDED)
            table.add_column("예측 자소나열")
            table.add_column("예측 단어")
            table.add_column("정답 단어")
            for pred, t in zip(preds, ts):
                for pred_, t_ in zip(pred, t):
                    table.add_row(codec.decode2자소나열(pred_), codec.decode2단어(pred_), codec.decode2단어(t_))

            console.print(table)

        accuracy = correct_count / total_count * 100
        console.print(f"Accuracy: {accuracy:03f}")

        # Early Stopping 확인
        if patience is not None:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                console.print(f"[yellow]patience: {patience_counter} / {patience}[/]")
                if patience_counter >= patience:
                    console.print(f"[bold red]Eearly Stopped: max accuracy {best_accuracy:.3f}%[/]")
                    break
