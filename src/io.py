import json
from typing import Any


def read_json(path) -> dict[str, Any]:
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file
