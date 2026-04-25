import json
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


def json_to_dict(json_file: str) -> dict:
    """Load a JSON file into a Python dict."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create the directory at ``path`` (with parents) and return it as a ``Path``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_json(path: str | Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` as pretty-printed JSON (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv_row(path: str | Path, row: dict) -> None:
    """Append ``row`` to the CSV at ``path``, aligning columns with the existing header if any."""
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
    """Build a filesystem-safe run id of the form ``<timestamp>_<tag>_<uuid6>``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    return f"{ts}_{safe_tag}_{short}"


def cvar(values, alpha: float = 0.95) -> float:
    """Empirical CVaR at level ``alpha`` for a cost distribution.

    Returns the mean of the observations greater than or equal to the
    ``alpha``-quantile. For a cost series (higher = worse) this is the
    average of the worst ``1 - alpha`` fraction of outcomes. Returns
    NaN when no finite value is available.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    threshold = float(np.quantile(finite, alpha))
    tail = finite[finite >= threshold]
    if tail.size == 0:
        return float("nan")
    return float(tail.mean())


def nanskewness(values) -> float:
    """Population skewness (3rd standardised moment) ignoring NaNs.

    Returns NaN if there are fewer than 3 finite values or if the
    standard deviation is zero. Uses ``ddof=0``.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    if std == 0:
        return float("nan")
    return float(np.mean(((finite - mean) / std) ** 3))


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python's ``random``, NumPy and (if available) PyTorch for reproducible runs."""
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


