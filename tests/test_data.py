from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_uplift.data import CRITEO_COLUMNS, load_criteo, validate_criteo


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A synthetic frame with the real Criteo column names."""
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n) for i in range(12)}
    for col in ("treatment", "conversion", "visit", "exposure"):
        data[col] = rng.integers(0, 2, size=n)
    data["f0"] = np.arange(n, dtype=float)  # a unique row marker
    return pd.DataFrame(data)


def test_validate_accepts_a_well_formed_frame() -> None:
    validate_criteo(_frame(50))


def test_validate_names_missing_and_extra_columns() -> None:
    df = _frame(50).drop(columns=["f3"]).assign(surprise=1)
    with pytest.raises(ValueError, match=r"missing \['f3'\].*unexpected \['surprise'\]"):
        validate_criteo(df)


def test_validate_rejects_non_binary_treatment() -> None:
    df = _frame(50)
    df.loc[7, "treatment"] = 2
    with pytest.raises(ValueError, match=r"'treatment' must be 0 or 1, found \[2\]"):
        validate_criteo(df)


@pytest.mark.parametrize("name", ["criteo.csv", "criteo.csv.gz"])
def test_load_subsamples_deterministically(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    _frame(300).to_csv(path, index=False)
    a = load_criteo(path, n=100, seed=5)
    b = load_criteo(path, n=100, seed=5)
    c = load_criteo(path, n=100, seed=6)
    assert list(a.columns) == list(CRITEO_COLUMNS)
    assert len(a) == 100
    assert a["f0"].is_unique
    assert a["f0"].is_monotonic_increasing  # original row order is kept
    pd.testing.assert_frame_equal(a, b)
    assert not a["f0"].equals(c["f0"])
    assert a["treatment"].dtype == np.int8


def test_load_returns_every_row_when_fewer_than_n(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _frame(40).to_csv(path, index=False)
    df = load_criteo(path, n=1_000)
    assert len(df) == 40
    assert df["f0"].tolist() == list(range(40))


def test_load_rejects_bad_schema_and_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    _frame(10).drop(columns=["visit"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_criteo(bad)
    with pytest.raises(ValueError, match="unsupported file type"):
        load_criteo(tmp_path / "data.xlsx")


def test_load_reads_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "criteo.parquet"
    _frame(60).to_parquet(path)
    assert len(load_criteo(path, n=25, seed=1)) == 25
