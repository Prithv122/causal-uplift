"""Turn ``grid.csv`` into ``REPORT.md``: summary tables and the mechanical H1-H3 verdicts.

The decision rules are copied from ``PREREGISTRATION.md`` (including amendment A0) and applied
without discretion. The report states numbers and rule outcomes only; interpretation belongs in
the README.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

Z_RULE = 2.0  # "greater than 2 x MC SE" and "biased: |bias| > 2 x own SE"
COVERAGE_FLOOR = 0.90
POINT_ATE_TOLERANCE = 0.01  # S/T/X-learner ATEs have no SE: one percentage point instead

H2_CELLS = ((0.5, 0.0), (1.0, 0.0), (2.0, 0.0))
H3_CELLS = ((0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0))
H1_LEARNERS = ("x_learner", "dr_learner")

SUPPORTED, NOT_SUPPORTED, NOT_EVALUABLE = "supported", "not supported", "not evaluable"


@dataclass
class Verdict:
    """A hypothesis outcome: status, every condition checked, and what was missing."""

    name: str
    status: str = NOT_EVALUABLE
    rows: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        if self.status == NOT_EVALUABLE:
            return f"not evaluable (missing: {'; '.join(self.missing)})"
        return self.status


# ---------------------------------------------------------------- small numeric helpers


def _mc_se(values: np.ndarray | pd.Series) -> float:
    """Standard deviation across replications divided by sqrt(count); NaN for fewer than two."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")


def _mean(values: np.ndarray | pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.nanmean(v)) if v.size and not np.isnan(v).all() else float("nan")


def _num(x: float, digits: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _pm(mean: float, se: float, digits: int = 4) -> str:
    if math.isnan(mean):
        return "-"
    return f"{mean:+.{digits}f} ± {_num(se, digits)}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _find_cell(df: pd.DataFrame, gamma: float, hidden: float) -> str | None:
    """The cell id for a (gamma, hidden) pair, or None when the results lack it."""
    hit = df[np.isclose(df["gamma"], gamma) & np.isclose(df["hidden"], hidden)]
    return None if hit.empty else str(sorted(hit["cell_id"].unique())[0])


def _rows(df: pd.DataFrame, cell_id: str, estimator: str) -> pd.DataFrame:
    return df[(df["cell_id"] == cell_id) & (df["estimator"] == estimator)]


@dataclass(frozen=True)
class AteStats:
    mean_bias: float
    own_se: float
    coverage: float
    reps: int


def _ate_stats(df: pd.DataFrame, cell_id: str, estimator: str) -> AteStats | None:
    """Mean bias, mean own SE and coverage for one estimator in one cell (None if no ATE)."""
    sub = _rows(df, cell_id, estimator)
    bias = sub["ate_hat"] - sub["true_ate"]
    if sub.empty or bias.isna().all():
        return None
    return AteStats(_mean(bias), _mean(sub["ate_se"]), _mean(sub["covered"]), len(sub))


def _paired_qini_diff(
    df: pd.DataFrame, cell_id: str, estimator: str, reference: str = "response"
) -> tuple[float, float, int] | None:
    """Mean paired difference in qini_true vs ``reference`` (paired by rep), its MC SE, pairs."""
    a = _rows(df, cell_id, estimator).set_index("rep")["qini_true"]
    b = _rows(df, cell_id, reference).set_index("rep")["qini_true"]
    diff = (a - b).dropna()
    if diff.empty:
        return None
    return float(diff.mean()), _mc_se(diff), int(diff.size)


def _cell_label(df: pd.DataFrame, cell_id: str) -> str:
    row = df[df["cell_id"] == cell_id].iloc[0]
    return f"gamma={row['gamma']:g}, hidden={row['hidden']:g}"


# ---------------------------------------------------------------- hypotheses


def h1_verdicts(df: pd.DataFrame) -> dict[str, Verdict]:
    """H1 per learner: paired qini_true gain over the response model > 2 MC SE at gamma=0."""
    cell = _find_cell(df, 0.0, 0.0)
    out = {}
    for learner in H1_LEARNERS:
        verdict = Verdict(f"H1 {learner}")
        out[learner] = verdict
        if cell is None:
            verdict.missing.append("cell gamma=0, hidden=0")
            continue
        for name in (learner, "response"):
            if _rows(df, cell, name).empty:
                verdict.missing.append(f"{name} in {cell}")
        if verdict.missing:
            continue
        paired = _paired_qini_diff(df, cell, learner)
        if paired is None or math.isnan(paired[1]):
            verdict.missing.append("at least 2 paired replications")
            continue
        mean_diff, se, pairs = paired
        met = mean_diff > Z_RULE * se
        verdict.status = SUPPORTED if met else NOT_SUPPORTED
        verdict.rows.append(
            {
                "cell": cell,
                "estimator": learner,
                "condition": "mean(qini_true - response qini_true) > 2 x MC SE",
                "mean_diff": mean_diff,
                "mc_se": se,
                "threshold": Z_RULE * se,
                "pairs": pairs,
                "met": met,
            }
        )
    return out


def _check_cells(
    df: pd.DataFrame,
    verdict: Verdict,
    cells: tuple[tuple[float, float], ...],
    estimators: tuple[str, ...],
) -> dict[tuple[str, str], AteStats]:
    """Collect ATE stats for every (cell, estimator) pair, recording whatever is missing."""
    stats: dict[tuple[str, str], AteStats] = {}
    for gamma, hidden in cells:
        cell = _find_cell(df, gamma, hidden)
        if cell is None:
            verdict.missing.append(f"cell gamma={gamma:g}, hidden={hidden:g}")
            continue
        for name in estimators:
            found = _ate_stats(df, cell, name)
            if found is None or math.isnan(found.own_se):
                verdict.missing.append(f"{name} ATE with SE in {cell}")
            else:
                stats[(cell, name)] = found
    return stats


def _condition(cell: str, name: str, text: str, value: float, threshold: float, met: bool) -> dict:
    return {
        "cell": cell,
        "estimator": name,
        "condition": text,
        "value": value,
        "threshold": threshold,
        "met": met,
    }


def h2_verdict(df: pd.DataFrame) -> Verdict:
    """H2: naive biased, ipw and dr_learner usable, in each of gamma 0.5, 1.0, 2.0 (hidden 0)."""
    verdict = Verdict("H2")
    stats = _check_cells(df, verdict, H2_CELLS, ("naive", "ipw", "dr_learner"))
    if verdict.missing:
        return verdict
    for gamma, hidden in H2_CELLS:
        cell = _find_cell(df, gamma, hidden)
        s = stats[(cell, "naive")]
        verdict.rows.append(
            _condition(
                cell,
                "naive",
                "|mean bias| > 2 x own SE",
                abs(s.mean_bias),
                Z_RULE * s.own_se,
                abs(s.mean_bias) > Z_RULE * s.own_se,
            )
        )
        for name in ("ipw", "dr_learner"):
            s = stats[(cell, name)]
            verdict.rows.append(
                _condition(
                    cell,
                    name,
                    "|mean bias| <= 1 x own SE",
                    abs(s.mean_bias),
                    s.own_se,
                    abs(s.mean_bias) <= s.own_se,
                )
            )
            verdict.rows.append(
                _condition(
                    cell,
                    name,
                    "coverage >= 0.90",
                    s.coverage,
                    COVERAGE_FLOOR,
                    s.coverage >= COVERAGE_FLOOR,
                )
            )
    verdict.status = SUPPORTED if all(r["met"] for r in verdict.rows) else NOT_SUPPORTED
    return verdict


def h3_verdict(df: pd.DataFrame) -> Verdict:
    """H3: ipw and dr_learner both biased (|mean bias| > 2 x own SE) in all four hidden cells."""
    verdict = Verdict("H3")
    stats = _check_cells(df, verdict, H3_CELLS, ("ipw", "dr_learner"))
    if verdict.missing:
        return verdict
    for gamma, hidden in H3_CELLS:
        cell = _find_cell(df, gamma, hidden)
        for name in ("ipw", "dr_learner"):
            s = stats[(cell, name)]
            verdict.rows.append(
                _condition(
                    cell,
                    name,
                    "|mean bias| > 2 x own SE",
                    abs(s.mean_bias),
                    Z_RULE * s.own_se,
                    abs(s.mean_bias) > Z_RULE * s.own_se,
                )
            )
    verdict.status = SUPPORTED if all(r["met"] for r in verdict.rows) else NOT_SUPPORTED
    return verdict


# ---------------------------------------------------------------- breaking points


def _ate_usable(s: AteStats) -> bool:
    if math.isnan(s.own_se):
        return abs(s.mean_bias) <= POINT_ATE_TOLERANCE
    return abs(s.mean_bias) <= s.own_se and s.coverage >= COVERAGE_FLOOR


def breaking_points(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Per estimator, the smallest hidden=0 gamma at which the ATE and the ranking stop working."""
    clean = df[np.isclose(df["hidden"], 0.0)]
    ordered = clean[["cell_id", "gamma"]].drop_duplicates().sort_values(["gamma", "cell_id"])
    cells = list(ordered.itertuples(index=False))
    rows = []
    for name in df["estimator"].drop_duplicates():
        ate_first, ate_seen, rank_first, rank_seen = None, False, None, False
        for cell_id, gamma in cells:
            stats = _ate_stats(clean, cell_id, name)
            if stats is not None:
                ate_seen = True
                if ate_first is None and not _ate_usable(stats):
                    ate_first = gamma
            if name == "response":
                continue
            paired = _paired_qini_diff(clean, cell_id, name)
            if paired is not None and not math.isnan(paired[1]):
                rank_seen = True
                if rank_first is None and not paired[0] > Z_RULE * paired[1]:
                    rank_first = gamma
        rows.append(
            {
                "estimator": name,
                "ate": _break_label(ate_seen, ate_first, "n/a (no ATE)"),
                "ranking": "reference"
                if name == "response"
                else _break_label(rank_seen, rank_first, "n/a (no ranking)"),
            }
        )
    return pd.DataFrame(rows), [g for _, g in cells]


def _break_label(seen: bool, first: float | None, unavailable: str) -> str:
    if not seen:
        return unavailable
    return "never in grid" if first is None else f"gamma = {first:g}"


# ---------------------------------------------------------------- rendering


def _summary_section(df: pd.DataFrame) -> str:
    order = list(df["estimator"].drop_duplicates())
    cells = (
        df[["cell_id", "gamma", "hidden"]].drop_duplicates().sort_values(["hidden", "gamma"])
    ).itertuples(index=False)
    headers = [
        "estimator",
        "reps",
        "ATE bias ± MC SE",
        "own SE",
        "coverage",
        "PEHE",
        "Spearman",
        "Qini ± MC SE",
        "policy top20",
        "oracle top20",
        "random top20",
        "fit s",
    ]
    parts = ["## 1. Summary by cell and estimator", ""]
    for cell_id, gamma, hidden in cells:
        sub = df[df["cell_id"] == cell_id]
        parts += [f"### {cell_id} (gamma={gamma:g}, hidden={hidden:g})", ""]
        body = []
        for name in order:
            rows = sub[sub["estimator"] == name]
            if rows.empty:
                continue
            bias = rows["ate_hat"] - rows["true_ate"]
            body.append(
                [
                    name,
                    str(len(rows)),
                    _pm(_mean(bias), _mc_se(bias)),
                    _num(_mean(rows["ate_se"])),
                    _num(_mean(rows["covered"]), 3),
                    _num(_mean(rows["pehe"])),
                    _num(_mean(rows["spearman"]), 3),
                    _pm(_mean(rows["qini_true"]), _mc_se(rows["qini_true"]), 3),
                    _num(_mean(rows["policy_top20"])),
                    _num(_mean(rows["oracle_top20"])),
                    _num(_mean(rows["random_top20"])),
                    _num(_mean(rows["fit_seconds"]), 2),
                ]
            )
        parts += [_table(headers, body), ""]
    return "\n".join(parts)


def _condition_table(verdict: Verdict) -> str:
    body = [
        [
            r["cell"],
            r["estimator"],
            r["condition"],
            _num(r["value"]),
            _num(r["threshold"]),
            "yes" if r["met"] else "no",
        ]
        for r in verdict.rows
    ]
    return _table(["cell", "estimator", "condition", "value", "threshold", "holds"], body)


def _h1_section(verdicts: dict[str, Verdict]) -> str:
    parts = ["### H1 - uplift models vs the response model at gamma=0, hidden=0", ""]
    body = []
    for learner, v in verdicts.items():
        if v.rows:
            r = v.rows[0]
            body.append(
                [
                    learner,
                    _num(r["mean_diff"], signed=True),
                    _num(r["mc_se"]),
                    _num(r["threshold"]),
                    str(r["pairs"]),
                    v.headline,
                ]
            )
        else:
            body.append([learner, "-", "-", "-", "-", v.headline])
    headers = ["learner", "mean paired diff", "MC SE", "2 x MC SE", "pairs", "verdict"]
    return "\n".join([*parts, _table(headers, body), ""])


def _hypothesis_section(title: str, rule: str, verdict: Verdict) -> str:
    parts = [f"### {title}", "", rule, "", f"Verdict: **{verdict.headline}**", ""]
    if verdict.rows:
        parts += [_condition_table(verdict), ""]
    return "\n".join(parts)


def _verdict_section(df: pd.DataFrame) -> str:
    h1 = h1_verdicts(df)
    return "\n".join(
        [
            "## 2. Pre-registered verdicts (mechanical)",
            "",
            _h1_section(h1),
            _hypothesis_section(
                "H2 - observed confounding, hidden=0, gamma = 0.5, 1.0, 2.0",
                "Every row must hold: naive biased; ipw and dr_learner usable.",
                h2_verdict(df),
            ),
            _hypothesis_section(
                "H3 - hidden confounding, (gamma, hidden) = (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), "
                "(1.0, 1.0)",
                "Every row must hold: ipw and dr_learner biased.",
                h3_verdict(df),
            ),
        ]
    )


def _breaking_section(df: pd.DataFrame) -> str:
    table, gammas = breaking_points(df)
    seen = ", ".join(f"{g:g}" for g in gammas) if gammas else "none"
    body = [[r.estimator, r.ate, r.ranking] for r in table.itertuples()]
    return "\n".join(
        [
            "## 3. Breaking points (descriptive)",
            "",
            "Smallest gamma at which the estimator stops being usable (hidden = 0). ATE rule: "
            "|mean bias| <= own SE and coverage >= 0.90 (S/T/X-learner ATEs, which report no SE: "
            "|mean bias| <= 0.01). Ranking rule: mean qini_true no longer above the response "
            "model's by 2 x MC SE of the paired difference.",
            "",
            f"hidden = 0 gammas in these results: {seen}",
            "",
            _table(["estimator", "ATE", "ranking"], body),
            "",
        ]
    )


def _header(df: pd.DataFrame, meta: dict | None) -> str:
    per_cell = df.groupby("cell_id")["rep"].nunique()
    reps = (
        str(per_cell.iloc[0])
        if per_cell.nunique() == 1
        else f"{per_cell.min()} to {per_cell.max()}"
    )
    lines = [
        "# Results report",
        "",
        f"- rows: {len(df)}",
        f"- cells: {df['cell_id'].nunique()}",
        f"- replications per cell: {reps}",
        f"- base learner: {', '.join(sorted(df['base'].astype(str).unique()))}",
    ]
    if meta:
        commit = meta.get("git_commit") or "unknown"
        changed = meta.get("git_tracked_changes")
        state = "" if changed is None else (" (uncommitted changes)" if changed else " (clean)")
        lines.append(f"- commit at run start: {commit}{state}")
        lines.append(f"- run started: {meta.get('started', 'unknown')}")
    return "\n".join(lines) + "\n"


def build_report(grid_csv_path: str | Path) -> str:
    """The full REPORT.md text for a ``grid.csv`` (reads ``run_meta.json`` beside it if present)."""
    path = Path(grid_csv_path)
    df = pd.read_csv(path)
    meta_path = path.with_name("run_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return "\n".join(
        [_header(df, meta), _summary_section(df), _verdict_section(df), _breaking_section(df)]
    )


def write_report(results_dir: str | Path) -> Path:
    """Write ``REPORT.md`` into ``results_dir`` from its ``grid.csv`` and return the path."""
    results = Path(results_dir)
    grid = results / "grid.csv"
    if not grid.exists():
        raise FileNotFoundError(f"{grid} not found; run the grid first")
    target = results / "REPORT.md"
    target.write_text(build_report(grid), encoding="utf-8")
    return target
