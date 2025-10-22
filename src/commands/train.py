from typing import Annotated

import numpy as np
import typer
from rich.panel import Panel

from common.optimizer import Adam
from constants import BACKWARD_INIT_VALUE, DATA_DIR, UINT8_MAX, console
from data_loader import LanguageDataLoader
from net import KOCRNet

app = typer.Typer(
    # name="train",
    # help="학습 과정을 관리 \n(학습, 검증, 테스트 등)",
    rich_markup_mode="rich",
)


# TODO: 학습 과정 추가
@app.command("train", help="학습을 시작합니다.")
def train(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
    max_epoch: Annotated[int, typer.Option(help="학습 에폭 수")] = 10,
    batch_size: Annotated[int, typer.Option(help="배치 크기")] = 32,
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
    data_loader = LanguageDataLoader(DATA_DIR, batch_size=batch_size)
    optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
    model = KOCRNet()

    for epoch in range(max_epoch):
        for x, t in data_loader.loader("train"):
            # 0~1 범위로 정규화
            x = x.astype(np.float32) / UINT8_MAX
            t = t.astype(np.float32) / UINT8_MAX

            # 모델 학습
            model.forward(x)  # foward 값 계산 (gradient 계산 때 사용)
            model.backward(BACKWARD_INIT_VALUE)  # gradient 계산
            grads = model.gradient()  # gradient 값 추출
            optimizer.update(model.params, grads)  # 파라미터 update

            print(x.shape, t.shape)
            break
        break
