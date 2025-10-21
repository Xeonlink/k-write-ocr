from yaspin import yaspin

with yaspin(text="Module Loading..."):
    import os
    import sys

    sys.path.append("src")

    from pathlib import Path

    import pyfiglet
    import typer

    from commands.data import app as data_app
    from commands.preprocess import app as preprocess_app
    from constants import APP_NAME


def main():
    pyfiglet.print_figlet(APP_NAME)

    app = typer.Typer(name=APP_NAME, help="한국어 손글씨 OCR 모델 바닥부터 만들기", rich_markup_mode="rich")
    app.add_typer(data_app)
    app.add_typer(preprocess_app)
    app()


if __name__ == "__main__":
    main()
