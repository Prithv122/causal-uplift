"""Outcome-only sanity check on the Criteo Uplift data (randomised; no true effect is known).

Because treatment was assigned at random, ``qini_observed`` (Radcliffe's Qini area from outcomes
only) is an unbiased ranking score on a held-out half, so each estimator can be compared with
random targeting without knowing any individual's true effect. There is no true tau on real data,
so there is no PEHE, Spearman or policy value against truth here.

What is reported, kept deliberately simple: for each outcome and estimator, the mean and the
standard deviation of ``qini_observed`` across the seeded train/test splits, and the share of
splits in which it beats a random-score baseline scored on the same test half. No normalisation
by a ceiling is attempted, because no perfect-information ceiling exists on real data. The check
has no hypothesis and does not feed H1-H3.

The data are licensed CC BY-NC-SA 4.0 and stay out of the repository: only per-split aggregates
are written, never user rows.
"""

from __future__ import annotations

import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import CRITEO_FILENAME, DEFAULT_SEED, load_criteo
from .estimators import ESTIMATORS
from .estimators.base import BaseKind
from .metrics import qini_observed

FEATURES = tuple(f"f{i}" for i in range(12))
DEFAULT_OUTCOMES = ("visit", "conversion")
DEFAULT_ESTIMATORS = ("response", "t_learner", "x_learner", "dr_learner")
RANDOM_NAME = "random"
LICENCE = "CC BY-NC-SA 4.0"
_EN_DASH = chr(0x2013)
FIXED_LINE = f"No hypothesis reads this check; it does not affect H1{_EN_DASH}H3."

COLUMNS = [
    "split",
    "outcome",
    "estimator",
    "qini_observed",
    "fit_seconds",
    "n_train",
    "n_test",
    "ate_test",
    "ate_se",
]


def _seed31(entropy: list[int]) -> int:
    """A 31-bit seed derived from a SeedSequence entropy list."""
    return int(np.random.SeedSequence(entropy).generate_state(1)[0] & 0x7FFFFFFF)


def _name_key(name: str) -> int:
    """A stable integer for a name, so a row does not depend on which others are run."""
    return zlib.crc32(name.encode())


def _read_sha256(path: Path) -> str | None:
    side = path.with_name(path.name + ".sha256")
    if not side.exists():
        return None
    parts = side.read_text(encoding="utf-8").split()
    return parts[0] if parts else None


def _welch_ate(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Difference in means (treated minus control) and its Welch standard error."""
    y1, y0 = y[t == 1].astype(float), y[t == 0].astype(float)
    se = math.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    return float(y1.mean() - y0.mean()), se


def run_criteo_check(
    path: str | Path,
    n: int = 1_000_000,
    splits: int = 5,
    seed: int = DEFAULT_SEED,
    base: BaseKind = "lgbm",
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES,
    estimators: tuple[str, ...] = DEFAULT_ESTIMATORS,
) -> pd.DataFrame:
    """Score each estimator's ranking by ``qini_observed`` on held-out halves of a subsample.

    ``n`` rows are drawn with ``load_criteo(path, n, seed)``. For each of ``splits`` seeded 50/50
    splits (stratified on treatment), each estimator is fitted on the training half per outcome and
    its score on the test half is evaluated with ``qini_observed``. A random baseline (seeded
    uniform scores, estimator name ``random``) is scored on the same test half, and the test-half
    difference-in-means ATE with its Welch SE is attached to every row of that split and outcome.
    All seeds come from ``SeedSequence``; estimator and outcome seeds depend on names, not order.

    Returns one row per split, outcome and estimator with columns ``split, outcome, estimator,
    qini_observed, fit_seconds, n_train, n_test, ate_test, ate_se``. Run settings are kept in
    ``df.attrs`` for ``write_criteo_report``.
    """
    unknown = [e for e in estimators if e not in ESTIMATORS]
    if unknown:
        raise ValueError(f"unknown estimators: {unknown}")
    if splits < 1:
        raise ValueError("splits must be at least 1")
    path = Path(path)
    data = load_criteo(path, n=n, seed=seed)
    X = data[list(FEATURES)].to_numpy(dtype=float)
    t = data["treatment"].to_numpy(dtype=np.int64)
    rows: list[dict] = []
    for split in range(splits):
        split_seed = _seed31([seed, 1, split])
        idx_train, idx_test = train_test_split(
            np.arange(len(data)), test_size=0.5, stratify=t, random_state=split_seed
        )
        for outcome in outcomes:
            y = data[outcome].to_numpy(dtype=np.int64)
            ate, ate_se = _welch_ate(y[idx_test], t[idx_test])
            common = {
                "split": split,
                "outcome": outcome,
                "n_train": len(idx_train),
                "n_test": len(idx_test),
                "ate_test": ate,
                "ate_se": ate_se,
            }
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, 2, split, _name_key(outcome)])
            )
            baseline = qini_observed(y[idx_test], t[idx_test], rng.random(len(idx_test)))
            rows.append(
                {**common, "estimator": RANDOM_NAME, "qini_observed": baseline, "fit_seconds": 0.0}
            )
            for name in estimators:
                est_seed = _seed31([seed, 3, split, _name_key(outcome), _name_key(name)])
                start = time.perf_counter()
                model = ESTIMATORS[name](est_seed, base).fit(
                    X[idx_train], t[idx_train], y[idx_train]
                )
                seconds = time.perf_counter() - start
                score = model.predict_tau(X[idx_test])
                value = qini_observed(y[idx_test], t[idx_test], score)
                rows.append(
                    {**common, "estimator": name, "qini_observed": value, "fit_seconds": seconds}
                )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.attrs.update(
        {
            "file": path.name,
            "sha256": _read_sha256(path),
            "n_requested": n,
            "n_loaded": len(data),
            "seed": seed,
            "base": base,
            "splits": splits,
        }
    )
    return df


def _fmt(x: float, digits: int = 6) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _summarise(sub: pd.DataFrame) -> list[list[str]]:
    """Per-estimator mean, sd across splits and share of splits beating random."""
    random_q = sub[sub["estimator"] == RANDOM_NAME].set_index("split")["qini_observed"]
    order = [RANDOM_NAME] + [e for e in sub["estimator"].drop_duplicates() if e != RANDOM_NAME]
    body = []
    for name in order:
        q = sub[sub["estimator"] == name].set_index("split")["qini_observed"]
        sd = q.std(ddof=1) if len(q) > 1 else float("nan")
        if name == RANDOM_NAME:
            beat = "-"
        else:
            wins = int((q > random_q.reindex(q.index)).sum())
            beat = f"{wins}/{len(q)}"
        body.append([name, _fmt(float(q.mean())), _fmt(float(sd)), beat])
    return body


def write_criteo_report(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write ``criteo.csv`` (per-split aggregates) and ``CRITEO.md`` into ``out_dir``.

    Returns the path of ``CRITEO.md``. The frame holds no user rows, only one row per split,
    outcome and estimator.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "criteo.csv", index=False)
    meta = df.attrs
    sha = meta.get("sha256") or "not recorded"
    n_loaded = meta.get("n_loaded", int(df["n_train"].iloc[0] + df["n_test"].iloc[0]))
    lines = [
        "# Outcome-only real-data sanity check (Criteo Uplift, randomised)",
        "",
        f"- file: {meta.get('file', CRITEO_FILENAME)}",
        f"- sha256: {sha}",
        f"- licence: {LICENCE}",
        f"- subsample: {n_loaded} rows, seed {meta.get('seed', DEFAULT_SEED)}",
        f"- splits: {df['split'].nunique()} seeded 50/50 splits, stratified on treatment",
        f"- base learner: {meta.get('base', 'unknown')}",
        "- metric: qini_observed on the held-out half; sd is across splits; beats random counts "
        "splits where the estimator's value exceeds the random baseline's on the same split",
        "",
    ]
    for outcome in df["outcome"].drop_duplicates():
        sub = df[df["outcome"] == outcome]
        ate = sub.drop_duplicates("split")
        lines += [
            f"## {outcome}",
            "",
            _table(
                ["estimator", "mean qini_observed", "sd across splits", "splits beating random"],
                _summarise(sub),
            ),
            "",
            "Test-half ATE (difference in means, mean over splits): "
            f"{_fmt(float(ate['ate_test'].mean()))} "
            f"(mean Welch SE {_fmt(float(ate['ate_se'].mean()))})",
            "",
        ]
    lines += [FIXED_LINE, ""]
    target = out / "CRITEO.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
