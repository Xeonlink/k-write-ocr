from yaspin import yaspin

with yaspin(text="Module Loading..."):
    import sys

    sys.path.append("src")

    import pyfiglet
    import typer

    from commands.data import app as data_app
    from commands.preprocess import app as preprocess_app
    from commands.train import app as train_app
    from constants import APP_NAME, console


def main():
    try:
        pyfiglet.print_figlet(APP_NAME)

        app = typer.Typer(name=APP_NAME, help="한국어 손글씨 OCR 모델 바닥부터 만들기", rich_markup_mode="rich")
        app.add_typer(data_app)
        app.add_typer(preprocess_app)
        app.add_typer(train_app)
        app()
    except Exception as e:
        console.print_exception(show_locals=False)


if __name__ == "__main__":
    main()
