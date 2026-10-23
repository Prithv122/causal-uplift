# Provenance of the full grid

- **Code that ran:** commit `f0bb079` (pre-registration A1 committed), clean working tree, started
  2026-10-23 11:56 UTC, finished 13:30 UTC, 10 workers, Windows 11, Python 3.12.13, numpy 2.5.4,
  pandas 3.1.0, scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1.
- **Known defect in `run_meta.json`:** the runner at `f0bb079` read the git state when the run
  *ended*, so the file names `50db6ba` with "uncommitted changes" (commits made while the grid ran:
  the Criteo check, tests, CI and notes; none touches the simulator, estimators, metrics or
  runner). Fixed in `ebdc0c8`, which reads it before the first replication. `run_meta.json` and
  the `REPORT.md` header are left exactly as the run wrote them.
- **Reproduction check:** a fresh clone of `f0bb079` with `uv sync --frozen` re-ran replications
  0, 57 and 199 of cells `g0.00_h0.00`, `g1.00_h0.00` and `g1.00_h1.00` (9 replications,
  72 rows). All 72 rows match `grid/*.csv` exactly (relative tolerance 1e-9), except
  `fit_seconds`, which is wall-clock time.
