import json
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None


def json_to_dict(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv_row(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if p.exists():
        existing_cols = list(pd.read_csv(p, nrows=0).columns)
        if existing_cols:
            for col in existing_cols:
                if col not in row_df.columns:
                    row_df[col] = ""
            row_df = row_df[existing_cols]
    row_df.to_csv(p, mode="a", header=not p.exists(), index=False)


def build_run_id(tag: str = "main") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    return f"{ts}_{safe_tag}_{short}"


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


