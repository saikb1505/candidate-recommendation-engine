from pathlib import Path
import yaml

_path = Path(__file__).parent / "prompts.yaml"

with _path.open() as _f:
    prompts: dict = yaml.safe_load(_f)
