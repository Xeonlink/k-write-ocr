import csv
import json
import re
import shutil
from typing import Annotated, Literal

import cv2
import numpy as np
import typer
from PIL import Image
from rich.panel import Panel
from rich.progress import track

from constants import DATA_DIR
from model.label_file import LabelFile
from shared.console import console

app = typer.Typer(
    name="preprocess",
    help="데이터 전처리 과정을 관리 \n(소스이미지 분리, 데이터 양 조절, 불필요한 데이터 제거 등)",
    rich_markup_mode="rich",
)


@app.command("crop", help="소스이미지에서 학습에 사용할 이미지를 자릅니다.")
def crop_source_inplace(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
) -> None:
    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"1. {DATA_DIR}/labels 폴더에 있는 모든 json 파일을 읽어서, bbox 정보를 추출합니다.",
                f"2. bbox 정보를 추출한 후, {DATA_DIR}/images 폴더에 있는 이미지 파일을 자릅니다.",
                f"3. 자른 이미지 파일을 {DATA_DIR}/images 폴더에 원래이름_index.png로 이름을 변경하여 저장합니다.",
                f"4. {DATA_DIR}/labels.csv 파일에 자른 이미지의 경로, 라벨 정보(이미지에 적힌 글자) row를 추가합니다.\n"
                f"5. 원본 json 파일을 제거합니다.",
                f"6. 원본 이미지 파일을 제거합니다.",
                f"7. {DATA_DIR}/labels 폴더를 제거합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Crop Source Image",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    labels_dir = DATA_DIR / "labels"
    images_dir = DATA_DIR / "images"
    output_csv = DATA_DIR / "labels.csv"

    # Check necessary folders
    if not labels_dir.is_dir():
        console.print(f"[red]labels/ 폴더가 존재하지 않습니다.[/]")
        return
    if not images_dir.is_dir():
        console.print(f"[red]images/ 폴더가 존재하지 않습니다.[/]")
        return

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        console.print(f"[red]labels/ 폴더에 처리할 json파일이 없습니다.[/]")
        return

    output_rows = []

    for label_path in track(label_files, description="소스 이미지 자르는 중"):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            label = LabelFile(**label_data)
        except Exception as e:
            console.print(f"[red][실패] {label_path}: {e}[/]")
            continue

        image_path = images_dir / (label_path.stem + ".png")
        pil_img = Image.open(image_path)

        for idx, bbox in enumerate(label.bbox, 1):
            x_min, x_max = min(bbox.x), max(bbox.x)
            y_min, y_max = min(bbox.y), max(bbox.y)
            cropped = pil_img.crop((x_min, y_min, x_max, y_max))

            cropped_image_path = images_dir / (label_path.stem + f"_{idx}.png")
            cropped.save(cropped_image_path)

            output_rows.append({"image_path": str(cropped_image_path.relative_to(DATA_DIR)), "label": bbox.data})
            console.print(f"[Crop] {cropped_image_path}")

        label_path.unlink()
        console.print(f"[Remove] {label_path}")

        image_path.unlink()
        console.print(f"[Remove] {image_path}")

    # Write CSV
    if output_rows:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
            writer.writerows(output_rows)
        console.print(f"[Write] {output_csv}")

    shutil.rmtree(labels_dir)
    console.print(f"[Remove] {labels_dir}")

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("gray", help="이미지를 흑백으로 바꿉니다.")
def image_to_gray_scale_inplace(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
    threshold: Annotated[
        int | None,
        typer.Option(
            help="임계값 1-254, 이 값보다 작으면 0, 크면 255로 색 변환, 지정하지 않으면 기본적인 gray scale 변환만 수행"
        ),
    ] = None,
) -> None:
    # 파라미터 검증
    if threshold is not None and (threshold < 1 or threshold > 254):
        console.print(f"[red]--threshold 는 1보다 크고 254보다 작은 값이어야 합니다. [default: None][/]")
        return

    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"- {DATA_DIR}/**/*.png 파일을 흑백으로 바꿉니다.",
                f"- 픽셀의 값이 임계값보다 작으면 0, 크면 255로 픽셀의 색을 바꿉니다",
                f"- 임계값을 지정하지 않으면 기본적인 gray scale 변환만 수행합니다.",
                f"- 임계값: [yellow]{threshold}[/]",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Image to Gray Scale",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    image_paths = list(DATA_DIR.rglob("*.png"))
    for image_path in track(image_paths, description="흑백 변환 중"):
        with Image.open(image_path) as img:
            gray_img = img.convert("L")
            if threshold is not None:
                gray_img = np.where(np.array(gray_img) < threshold, 0, 255).astype(np.uint8)
                gray_img = Image.fromarray(gray_img, mode="L")
            gray_img.save(image_path)

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("resize", help="모든 이미지의 크기를 변경합니다.")
def image_resize_inplace(
    width: Annotated[int, typer.Argument(help="변경할 가로 크기")],
    height: Annotated[int, typer.Argument(help="변경할 세로 크기")],
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
    algo: Annotated[Literal["nearest", "bilinear", "lanczos"], typer.Option(help="Resize 알고리즘")] = "nearest",
) -> None:
    # 파생된 파라미터
    resample_algo = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "lanczos": Image.LANCZOS,
    }[algo]

    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"- {DATA_DIR}/**/*.png 파일을 크기를 [yellow]{width}x{height}[/]로 변경합니다.",
                f"- [yellow]{algo}[/] 알고리즘을 사용합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Image Resize",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    image_paths = list(DATA_DIR.rglob("*.png"))
    if not image_paths:
        console.print(f"[red]{DATA_DIR}/**/*.png 파일을 찾을 수 없습니다.[/]")
        return

    for image_path in track(image_paths, description="이미지 크기 변경 중"):
        image = Image.open(image_path)
        image = image.resize((width, height), resample=resample_algo)
        image.save(image_path)

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("filter:korean", help="한글이 포함되지 않은 데이터를 삭제합니다.")
def image_filter_korean_inplace(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
) -> None:
    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"1. {DATA_DIR}/labels.csv 에서 label이 한글이 아닌 행을 삭제합니다.",
                f"2. 삭제된 행과 연결된 이미지 파일도 삭제합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Image Filter Korean",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    labels_csv_path = DATA_DIR / "labels.csv"
    if not labels_csv_path.exists():
        console.print(f"[red]{labels_csv_path} 파일을 찾을 수 없습니다.[/]")
        return

    rows: list[list[str]] = []
    filtered_rows: list[list[str]] = []

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    korean_pattern = re.compile(r"^[ㄱ-ㅣ가-힣]+$")
    filtered_rows = [row for row in rows if korean_pattern.match(row[1])]
    deleted_rows = [row for row in rows if row not in filtered_rows]

    for row in track(deleted_rows, description="한글 필터링 중"):
        img_path = DATA_DIR / row[0]
        img_path.unlink(missing_ok=True)  # 파일이 없어도 에러 발생하지 않음

    with open(labels_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("drop", help="전체 데이터 파일의 수를 줄입니다. (학습이 오래걸리는 경우)")
def image_drop_inplace(
    count: Annotated[int, typer.Argument(help="삭제할 데이터 수 \n(0: 필터에 걸리는 모든 파일 삭제)")] = 100,
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
) -> None:
    # 파라미터 검증
    if count <= 0:
        console.print(f"[red]--count 는 0보다 커야 합니다.[/]")
        return

    # 작업 설명 출력하기
    if not yes:
        panel_content = "".join(
            [
                f"📂 데이터셋 폴더의 경로\n",
                f"[blue]./{DATA_DIR}[/]\n",
                f"\n",
                f"[green]Jobs[/]\n",
                f"- {DATA_DIR}/labels.csv 에서 마지막 [yellow]{count}[/]개의 행을 삭제합니다.\n" if count > 0 else "",
                f"- 이름순으로 정렬하여 마지막 [yellow]{count}[/]개의 파일을 삭제합니다.\n" if count > 0 else "",
                f"- 삭제된 행과 연결된 이미지 파일도 삭제합니다.",
                f"\n",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Image Drop",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    labels_csv_path = DATA_DIR / "labels.csv"
    if not labels_csv_path.exists():
        console.print(f"[red]{labels_csv_path} 파일을 찾을 수 없습니다.[/]")
        return

    rows: list[list[str]] = []
    with open(labels_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < count:
        console.print(f"[red]filter된 행의 수({len(rows)}) < {count}[/]")
        return

    for row in track(rows[-count:], description="데이터 삭제 중"):
        image_path = (DATA_DIR / row[0]).resolve()
        image_path.unlink(missing_ok=True)

    rows_to_keep = rows[:-count]
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_keep)

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")
