import glob
import math
import shutil
import zipfile
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.panel import Panel
from rich.progress import track
from rich.table import Table, box
from yaspin import yaspin

from constants import DATA_DIR
from shared.console import console

app = typer.Typer(
    name="data",
    help="데이터셋의 폴더형태를 정의 및 관리 \n(압축해제, 폴더정리, 불필요한 파일제거 등)",
    rich_markup_mode="rich",
)


@app.command("unzip", help="데이터셋의 압축을 해제하고, zip파일을 삭제합니다.")
def unzip_raw(
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
) -> None:
    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 원본 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"1. 원본 데이터셋 폴더내의 zip파일을 압축해제 합니다.",
                f"2. 압축 해제가 완료된 zip파일을 [red]삭제[/] 합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title=f"Unzip Raw Data",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    zip_files = glob.iglob(str(DATA_DIR / "**" / "*.zip"), recursive=True)
    exist_zip_files_flag = False

    for zip_path in zip_files:
        exist_zip_files_flag = True
        zip_file = Path(zip_path)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(zip_file.parent)
        console.print(f"[Unzip] {zip_file}")
        zip_file.unlink()
        console.print(f"[Remove] {zip_file}")

    if not exist_zip_files_flag:
        console.print(f"[red]ZIP 파일이 존재하지 않습니다.[/]")
        return

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("compact", help="데이터셋의 폴더 구조를 정리합니다.")
def compact_data(
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
                f"1. {DATA_DIR}에서 모든 *.json 파일을 찾습니다.",
                f"2. 찾은 *.json 파일을 '{DATA_DIR}/labels' 폴더로 이동시킵니다.",
                f"3. {DATA_DIR}에서 모든 .png 파일을 찾습니다.",
                f"4. 찾은 .png 파일을 '{DATA_DIR}/images' 폴더로 이동시킵니다.",
                f"5. 나머지 폴더를 삭제합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Compact Data Folder",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    json_files = glob.iglob(str(DATA_DIR / "**" / "*.json"), recursive=True)
    png_files = glob.iglob(str(DATA_DIR / "**" / "*.png"), recursive=True)
    exist_json_files_flag = False
    exist_png_files_flag = False

    for src_path in json_files:
        exist_json_files_flag = True
        src_file = Path(src_path)
        dest_file = DATA_DIR / "labels" / src_file.name
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dest_file)
        console.print(f"[Move] {src_file}")

    if not exist_json_files_flag:
        console.print(f"[red]JSON 파일이 존재하지 않습니다.[/]")
        return

    for src_path in png_files:
        exist_png_files_flag = True
        src_file = Path(src_path)
        dest_file = DATA_DIR / "images" / src_file.name
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dest_file)
        console.print(f"[Move] {src_file}")

    if not exist_png_files_flag:
        console.print(f"[red]PNG 파일이 존재하지 않습니다.[/]")
        return

    # 나머지 폴더 삭제하기
    for dir_path in DATA_DIR.iterdir():
        if dir_path.is_dir() and dir_path.name not in ["labels", "images"]:
            shutil.rmtree(dir_path)
            console.print(f"[Remove] {dir_path}")

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("sync", help="데이터와 라벨의 수를 동일하게 맞춥니다.")
def sync_data(
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
                f"1. 라벨과 짝을 이루는(파일이 이름이 같은) 이미지 파일을 찾습니다.",
                f"2. 짝을 이루지 못한 라벨 파일을 삭제합니다.",
                f"3. 짝을 이루지 못한 이미지 파일을 삭제합니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Sync Data",
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

    if not labels_dir.is_dir():
        console.print(f"[red]labels/ 폴더가 존재하지 않습니다.[/]")
        return
    if not images_dir.is_dir():
        console.print(f"[red]images/ 폴더가 존재하지 않습니다.[/]")
        return

    label_stems = set(f.stem for f in labels_dir.glob("*.json"))
    image_stems = set(f.stem for f in images_dir.glob("*.png"))
    unmatched_stems = label_stems ^ image_stems  # 대칭차집합

    for stem in track(unmatched_stems, description="데이터 동기화 중"):
        label_file = labels_dir / f"{stem}.json"
        image_file = images_dir / f"{stem}.png"
        if label_file.exists():
            label_file.unlink()
        if image_file.exists():
            image_file.unlink()

    # 작업 결과 출력하기
    console.print("")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("drop", help="전체 데이터 파일의 수를 줄입니다. (학습이 오래걸리는 경우)")
def drop_data(
    rate: Annotated[float, typer.Argument(help="줄일 비율")],
    yes: Annotated[bool, typer.Option(help="확인없이 진행")] = False,
) -> None:
    # 파라미터 검증
    if rate <= 0 or rate >= 1:
        console.print(f"[red]--rate 는 0보다 크고 1 보다 작은 값이어야 합니다. [default: 0.9][/]")
        return

    # 작업 설명 출력하기
    if not yes:
        panel_content = "\n".join(
            [
                f"📂 데이터셋 폴더의 경로",
                f"[blue]./{DATA_DIR}[/]",
                f"",
                f"[green]Jobs[/]",
                f"1. 데이터 파일 수를 [yellow]{rate * 100}%[/] 줄입니다.",
                f"",
                f"이 작업은 되돌릴 수 없습니다.",
            ]
        )
        panel = Panel(
            panel_content,
            title="Drop Data",
            title_align="left",
            border_style="green bold",
            padding=(1, 2),
        )
        console.print(panel)

        if not typer.confirm(f"진행하시겠습니까?"):
            console.print("Operation cancelled.", style="red")
            return

    # 작업하기
    paths = list((DATA_DIR / "labels").glob("*.json"))
    paths.sort(reverse=True)
    remove_count = int(len(paths) * rate)
    paths_to_remove = paths[:remove_count]
    for path in track(paths_to_remove, description="데이터 삭제 중"):
        file = Path(path)
        file.unlink()
        image_file = DATA_DIR / "images" / (file.stem + ".png")
        image_file.unlink()

    # 작업 결과 출력하기
    console.print("")
    console.print(f"원본 데이터 수: {len(paths)}")
    console.print(f"줄인 데이터 수: {remove_count}")
    console.print(f"남은 데이터 수: {len(paths) - remove_count}")
    console.print(f"[green]🎉 모든 작업이 완료되었습니다.[/]")


@app.command("info", help=f"{DATA_DIR}/**/*.png 들의 통계정보를 출력합니다.")
def image_info() -> None:
    # 데이터 불러오기
    image_files = list(DATA_DIR.rglob("*.png"))
    if not image_files:
        console.print(f"[red]{DATA_DIR}/**/*.png 파일을 찾을 수 없습니다.[/]")
        return

    # 기초값 계산하기
    # [이미지 파일 경로, 가로길이, 세로길이, 파일크기(KB)] 형태의 리스트 생성
    type ImageInfo = tuple[Path, int | None, int | None, int]
    stats: list[ImageInfo] = []
    for img_path in track(image_files, description="이미지 정보 추출 중..."):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None  # 이미지 열기 실패한 경우
        file_size_bytes = img_path.stat().st_size
        stats.append((img_path.name, width, height, int(file_size_bytes / 1024)))

    # 통계 계산하기
    num_files = len(stats)
    stats_with_dims = [s for s in stats if s[1] is not None and s[2] is not None]
    min_width_img = min(stats_with_dims, key=lambda s: s[1] or math.inf)
    max_width_img = max(stats_with_dims, key=lambda s: s[1] or -math.inf)
    min_height_img = min(stats_with_dims, key=lambda s: s[2] or math.inf)
    max_height_img = max(stats_with_dims, key=lambda s: s[2] or -math.inf)
    min_size_img = min(stats, key=lambda s: s[3] or math.inf)
    max_size_img = max(stats, key=lambda s: s[3] or -math.inf)

    # rich table 생성
    table = Table(header_style="green", box=box.ROUNDED)
    table.add_column("항목")
    table.add_column("파일명")
    table.add_column("가로")
    table.add_column("세로")
    table.add_column("파일크기")

    table.add_row(
        "가로길이 최대",
        f"{max_width_img[0]}",
        f"[blue bold]{max_width_img[1]:,}[/]",
        f"{max_width_img[2]:,}",
        f"{max_width_img[3]:,} KB",
    )
    table.add_row(
        "가로길이 최소",
        f"{min_width_img[0]}",
        f"[yellow bold]{min_width_img[1]:,}[/]",
        f"{min_width_img[2]:,}",
        f"{min_width_img[3]:,} KB",
    )
    table.add_row(
        "세로길이 최대",
        f"{max_height_img[0]}",
        f"{max_height_img[1]:,}",
        f"[blue bold]{max_height_img[2]:,}[/]",
        f"{max_height_img[3]:,} KB",
    )
    table.add_row(
        "세로길이 최소",
        f"{min_height_img[0]}",
        f"{min_height_img[1]:,}",
        f"[yellow bold]{min_height_img[2]:,}[/]",
        f"{min_height_img[3]:,} KB",
    )
    table.add_row(
        "용량 최대",
        f"{max_size_img[0]}",
        f"{max_size_img[1]:,}",
        f"{max_size_img[2]:,}",
        f"[blue bold]{max_size_img[3]:,} KB[/]",
    )
    table.add_row(
        "용량 최소",
        f"{min_size_img[0]}",
        f"{min_size_img[1]:,}",
        f"{min_size_img[2]:,}",
        f"[yellow bold]{min_size_img[3]:,} KB[/]",
    )

    console.print("[bold green]📊 이미지 통계[/]")
    console.print(f"총 파일 수: [yellow]{num_files}[/]\n")
    console.print(table)
