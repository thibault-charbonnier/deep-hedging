import json


def json_to_dict(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)
