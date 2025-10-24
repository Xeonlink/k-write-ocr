# 📖 개발 이야기

### 🖥️ 왜 UI요소를 프로젝트에 추가했는가?

처음에는 UI 요소를 프로젝트에 추가하려고 생각하지 않았습니다. 기능을 파일별로 구분하고 uv run {파일명}.py를 사용해서 실행하는 간단한 구성으로 하려고 했습니다.

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--decimal", ...)
parser.add_argument("--fast", ...)
args = parser.parse_args()

print(args.decimal)
print(args.fast)
```

하지만 데이터를 다운받고 폴더형태를 변형하는 동안, 여러 파일들이 생각했던 것만큼 독립적으로 작동하지 않고, 서로가 서로에게 의존하게되는 구조가 형성되었습니다. 또한 파일이 많아지니 파일의 이름을 까먹거나, 각 명령 파일에서 상요할 수 있는 argument나 option들을 잊어버리는 문제가 있었습니다.

```python
app = typer.Typer(name=APP_NAME, ...)
app.add_typer(data_app)
app.add_typer(preprocess_app)
app.add_typer(train_app)
app()
```

이 문제를 해결하기 위해서 typer를 도입했습니다. typer는 구조화된 command 구조를 통해서 비슷한 command끼리 묶어서 관리할 수 있었고, option이나 arg에 대한 설명을 적어두면 --help option을 통해서 언제든지 어떤 기능을 하는 옵션이었는지 확인할 수 있었습니다.

```bash
un run main.py data unzip --help # arg 및 옵션 보기
```
