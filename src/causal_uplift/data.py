"""Loader and checks for the Criteo Uplift benchmark (a randomised trial with no true effect)."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Criteo Uplift Prediction dataset v2.1, 13,979,592 rows, ~297 MiB gzipped CSV.
# Dataset page: https://huggingface.co/datasets/criteo/criteo-uplift
# Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike; cite Diemert et al., AdKDD 2018).
CRITEO_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
CRITEO_URL = f"https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/{CRITEO_FILENAME}"
CRITEO_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
CRITEO_BYTES = 311_422_618

# Column order and dtypes. Features are float32 and the binary columns int8 to keep the full
# 14M-row file near 0.7 GB in memory.
CRITEO_COLUMNS: dict[str, str] = {
    **{f"f{i}": "float32" for i in range(12)},
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}
BINARY_COLUMNS = ("treatment", "conversion", "visit", "exposure")

DEFAULT_SEED = 20260830
_CHUNK = 1 << 20
_PROGRESS_EVERY = 50 * _CHUNK


def validate_criteo(df: pd.DataFrame) -> None:
    """Raise `ValueError` if `df` is not a Criteo uplift frame (columns and binary values)."""
    missing = [c for c in CRITEO_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CRITEO_COLUMNS]
    if missing or extra:
        raise ValueError(f"Criteo columns do not match: missing {missing}, unexpected {extra}")
    for col in BINARY_COLUMNS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            values = sorted(df.loc[bad, col].unique().tolist(), key=str)
            raise ValueError(f"column {col!r} must be 0 or 1, found {values[:5]}")


def load_criteo(path: str | Path, n: int = 1_000_000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Read the file at `path` (.csv, .csv.gz or .parquet), validate it, and subsample.

    A frame longer than `n` is reduced to exactly `n` rows drawn without replacement with
    `default_rng(seed)`, keeping the original row order; a shorter one is returned whole.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif name.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path, dtype=CRITEO_COLUMNS)
    else:
        raise ValueError(f"unsupported file type {path.name!r}; use .csv, .csv.gz or .parquet")
    validate_criteo(df)
    df = df[list(CRITEO_COLUMNS)]
    if len(df) > n:
        keep = np.sort(np.random.default_rng(seed).choice(len(df), size=n, replace=False))
        df = df.iloc[keep]
    return df.reset_index(drop=True)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download_criteo(dest: Path) -> Path:
    """Download the dataset into directory `dest` and write `<file>.sha256` beside it.

    The streamed hash is checked against the published one; a mismatch deletes the partial
    file and raises. An intact existing file is not downloaded again. Needs network access and
    ~300 MB of disk; the data is licensed CC BY-NC-SA 4.0 and must stay out of version control.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / CRITEO_FILENAME
    if target.exists() and _sha256_of(target) == CRITEO_SHA256:
        digest = CRITEO_SHA256
    else:
        part = target.with_name(target.name + ".part")
        hasher = hashlib.sha256()
        done = 0
        with urllib.request.urlopen(CRITEO_URL, timeout=60) as response, part.open("wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                hasher.update(chunk)
                if done // _PROGRESS_EVERY != (done + len(chunk)) // _PROGRESS_EVERY:
                    print(f"  {done / 1e6:,.0f} / {CRITEO_BYTES / 1e6:,.0f} MB", file=sys.stderr)
                done += len(chunk)
        digest = hasher.hexdigest()
        if digest != CRITEO_SHA256:
            part.unlink()
            raise ValueError(f"checksum mismatch: got {digest}, expected {CRITEO_SHA256}")
        part.replace(target)
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n")
    print(f"{digest}  {target}")
    return target
