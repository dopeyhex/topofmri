from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ripser import ripser


EXPERIMENT_STATES: dict[str, list[str]] = {
    "LANGUAGE": ["math", "story"],
    "MOTOR": ["l", "r"],
    "SOCIAL": ["mental", "rnd"],
}

STATE_LABELS: dict[str, str] = {
    "math": "Math",
    "story": "Story",
    "l": "Left hand or foot",
    "r": "Right hand or foot",
    "mental": "Mental interaction",
    "rnd": "Random motion",
}

SOURCE_LABELS: dict[str, str] = {
    "low": r"$\mathcal{L}_c$",
    "top": r"$\mathcal{M}_c$",
}

SOURCE_COLORS: dict[str, str] = {
    "low": "#4c78a8",
    "top": "#f58518",
}

PHASE_ENCODINGS: tuple[str, ...] = ("RL", "LR")

SeriesGetter = Callable[[str, str, str, str, int], np.ndarray]


@dataclass(frozen=True)
class TakensConfig:
    dim: int = 3
    delay: int = 2
    stride: int = 1
    maxdim: int = 1


@dataclass(frozen=True)
class SurrogateConfig:
    null_model: str = "shuffle"
    n_surrogates: int = 5
    seed: int = 42


@dataclass(frozen=True)
class RegionSeriesRecord:
    experiment: str
    source: str
    state: str
    subject_id: str
    region_id: int
    values: np.ndarray


def validate_experiments(experiments: Iterable[str]) -> list[str]:
    normalized = [experiment.upper() for experiment in experiments]
    unknown = sorted(set(normalized) - set(EXPERIMENT_STATES))
    if unknown:
        raise ValueError(
            f"Unknown experiments: {unknown}. Expected one of {sorted(EXPERIMENT_STATES)}"
        )
    return normalized


def validate_phase_encoding(phase_encoding: str) -> str:
    phase_encoding = phase_encoding.upper()
    if phase_encoding not in PHASE_ENCODINGS:
        raise ValueError(
            f"Unknown phase encoding: {phase_encoding}. Expected one of {PHASE_ENCODINGS}"
        )
    return phase_encoding


def validate_state(experiment: str, state: str) -> str:
    experiment = validate_experiments([experiment])[0]
    if state not in EXPERIMENT_STATES[experiment]:
        raise ValueError(f"Unknown state {state!r} for experiment {experiment}.")
    return state


def validate_source(source: str) -> str:
    if source not in SOURCE_LABELS:
        raise ValueError(f"Unknown source {source!r}. Expected one of {sorted(SOURCE_LABELS)}")
    return source


def validate_takens_config(config: TakensConfig) -> None:
    if config.dim <= 0:
        raise ValueError("Takens embedding dimension must be positive.")
    if config.delay <= 0:
        raise ValueError("Takens delay must be positive.")
    if config.stride <= 0:
        raise ValueError("Takens stride must be positive.")
    if config.maxdim < 1:
        raise ValueError("maxdim must be at least 1 because the analysis reports H1 summaries.")


def validate_surrogate_config(config: SurrogateConfig) -> None:
    if config.null_model not in {"shuffle", "cyclic_shift"}:
        raise ValueError("null_model must be either 'shuffle' or 'cyclic_shift'.")
    if config.n_surrogates <= 0:
        raise ValueError("n_surrogates must be positive.")


def records_from_frame(
    frame: pd.DataFrame,
    experiment: str,
    source: str,
    state: str,
    subject_id: str,
    region_column: str | None = None,
) -> list[RegionSeriesRecord]:
    """Convert one loaded region-by-time matrix into in-memory series records."""
    experiment = validate_experiments([experiment])[0]
    source = validate_source(source)
    state = validate_state(experiment, state)

    if frame.empty:
        return []

    region_column = region_column or str(frame.columns[0])
    time_columns = [column for column in frame.columns if column != region_column]
    if not time_columns:
        raise ValueError("A region frame must contain at least one time column.")

    records: list[RegionSeriesRecord] = []
    for _, row in frame.iterrows():
        records.append(
            RegionSeriesRecord(
                experiment=experiment,
                source=source,
                state=state,
                subject_id=str(subject_id),
                region_id=int(row[region_column]),
                values=row[time_columns].to_numpy(dtype=float),
            )
        )
    return records


def zscore_series(values: np.ndarray) -> np.ndarray:
    """Standardize one regional time series."""
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def takens_embedding(values: np.ndarray, config: TakensConfig) -> np.ndarray:
    """Construct delay-coordinate vectors from one standardized time series."""
    validate_takens_config(config)
    window = (config.dim - 1) * config.delay + 1
    if len(values) < window:
        raise ValueError(
            f"Time series length {len(values)} is too short for "
            f"dim={config.dim}, delay={config.delay}"
        )
    starts = np.arange(0, len(values) - window + 1, config.stride)
    offsets = np.arange(config.dim) * config.delay
    return np.stack([values[starts + offset] for offset in offsets], axis=1)


def compute_diagrams(
    series: np.ndarray,
    config: TakensConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute Vietoris-Rips persistence diagrams for one Takens point cloud."""
    embedded = takens_embedding(series, config)
    diagrams = ripser(embedded, metric="euclidean", maxdim=config.maxdim)["dgms"]
    return embedded, diagrams


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return np.empty((0, 2))
    return diagram[np.isfinite(diagram[:, 1])]


def lifetimes(diagram: np.ndarray) -> np.ndarray:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return np.array([], dtype=float)
    return np.clip(finite[:, 1] - finite[:, 0], a_min=0.0, a_max=None)


def total_persistence(diagram: np.ndarray) -> float:
    values = lifetimes(diagram)
    return float(values.sum()) if values.size else 0.0


def max_lifetime(diagram: np.ndarray) -> float:
    values = lifetimes(diagram)
    return float(values.max()) if values.size else 0.0


def persistence_entropy(diagram: np.ndarray) -> float:
    values = lifetimes(diagram)
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total == 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def summarize_region(
    record: RegionSeriesRecord,
    series: np.ndarray,
    embedded: np.ndarray,
    diagrams: list[np.ndarray],
) -> dict:
    """Create one output row for one regional time series."""
    row = {
        "experiment": record.experiment,
        "source": record.source,
        "state": record.state,
        "subject_id": record.subject_id,
        "region_id": int(record.region_id),
        "n_timepoints": int(len(series)),
        "n_embedded_points": int(len(embedded)),
    }
    for dim, diagram in enumerate(diagrams):
        row[f"H{dim}_features"] = len(finite_diagram(diagram))
        row[f"H{dim}_total_persistence"] = total_persistence(diagram)
        row[f"H{dim}_max_lifetime"] = max_lifetime(diagram)
        row[f"H{dim}_entropy"] = persistence_entropy(diagram)
    return row


def build_actual_rows(
    records: Iterable[RegionSeriesRecord],
    takens: TakensConfig,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Compute persistence summaries for loaded regional time series."""
    validate_takens_config(takens)
    rows: list[dict] = []
    started = time.time()

    for index, record in enumerate(records, start=1):
        if progress_every and (index == 1 or index % progress_every == 0):
            elapsed = time.time() - started
            print(f"[actual] processed {index} region-series after {elapsed:.1f}s", flush=True)

        series = zscore_series(record.values)
        embedded, diagrams = compute_diagrams(series, takens)
        rows.append(summarize_region(record, series, embedded, diagrams))

    return pd.DataFrame(rows)


def make_surrogate(
    series: np.ndarray,
    config: SurrogateConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one surrogate time series."""
    if config.null_model == "shuffle":
        return rng.permutation(series)
    if config.null_model == "cyclic_shift":
        if len(series) < 2:
            return series.copy()
        shift = int(rng.integers(1, len(series)))
        return np.roll(series, shift)
    raise ValueError(f"Unsupported null model: {config.null_model}")


def build_null_rows(
    actual_df: pd.DataFrame,
    series_getter: SeriesGetter,
    takens: TakensConfig,
    surrogate: SurrogateConfig,
    progress_every: int | None = 500,
) -> pd.DataFrame:
    """Compare original persistence summaries with in-memory or notebook-loaded surrogates."""
    validate_takens_config(takens)
    validate_surrogate_config(surrogate)

    rng = np.random.default_rng(surrogate.seed)
    rows: list[dict] = []
    started = time.time()
    groups = actual_df[
        ["experiment", "source", "state", "subject_id", "region_id", "H1_total_persistence",
         "H1_features", "H1_max_lifetime"]
    ].drop_duplicates()
    total_groups = len(groups)

    for group_index, group in enumerate(groups.itertuples(index=False), start=1):
        if progress_every and (
            group_index == 1 or group_index % progress_every == 0 or group_index == total_groups
        ):
            elapsed = time.time() - started
            print(
                f"[null] processed {group_index}/{total_groups} region-series "
                f"after {elapsed:.1f}s",
                flush=True,
            )

        raw_series = series_getter(
            group.experiment,
            group.source,
            group.state,
            str(group.subject_id),
            int(group.region_id),
        )
        series = zscore_series(raw_series)

        surrogate_tp = []
        surrogate_feat = []
        surrogate_max = []
        for _ in range(surrogate.n_surrogates):
            surrogate_series = make_surrogate(series, surrogate, rng)
            _, diagrams = compute_diagrams(surrogate_series, takens)
            h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
            surrogate_tp.append(total_persistence(h1))
            surrogate_feat.append(len(finite_diagram(h1)))
            surrogate_max.append(max_lifetime(h1))

        surrogate_tp = np.asarray(surrogate_tp, dtype=float)
        surrogate_feat = np.asarray(surrogate_feat, dtype=float)
        surrogate_max = np.asarray(surrogate_max, dtype=float)

        rows.append(
            {
                "experiment": group.experiment,
                "source": group.source,
                "state": group.state,
                "subject_id": group.subject_id,
                "region_id": int(group.region_id),
                "actual_H1_total_persistence": float(group.H1_total_persistence),
                "null_H1_total_persistence_mean": float(surrogate_tp.mean()),
                "null_H1_total_persistence_std": float(surrogate_tp.std(ddof=0)),
                "actual_H1_features": int(group.H1_features),
                "null_H1_features_mean": float(surrogate_feat.mean()),
                "actual_H1_max_lifetime": float(group.H1_max_lifetime),
                "null_H1_max_lifetime_mean": float(surrogate_max.mean()),
                "delta_H1_total_persistence": float(
                    group.H1_total_persistence - surrogate_tp.mean()
                ),
                "delta_H1_features": float(group.H1_features - surrogate_feat.mean()),
                "delta_H1_max_lifetime": float(group.H1_max_lifetime - surrogate_max.mean()),
                "empirical_p_H1_total_persistence": float(
                    (np.sum(surrogate_tp >= group.H1_total_persistence) + 1)
                    / (surrogate.n_surrogates + 1)
                ),
                "empirical_p_H1_max_lifetime": float(
                    (np.sum(surrogate_max >= group.H1_max_lifetime) + 1)
                    / (surrogate.n_surrogates + 1)
                ),
                "null_model": surrogate.null_model,
            }
        )

    return pd.DataFrame(rows)


def aggregate_actual(actual_df: pd.DataFrame) -> pd.DataFrame:
    return (
        actual_df.groupby(["experiment", "state", "source"])
        .agg(
            n_rows=("region_id", "size"),
            n_subjects=("subject_id", "nunique"),
            H1_features_mean=("H1_features", "mean"),
            H1_features_std=("H1_features", "std"),
            H1_total_persistence_mean=("H1_total_persistence", "mean"),
            H1_total_persistence_std=("H1_total_persistence", "std"),
            H1_max_lifetime_mean=("H1_max_lifetime", "mean"),
            H1_max_lifetime_std=("H1_max_lifetime", "std"),
        )
        .reset_index()
    )


def aggregate_null(null_df: pd.DataFrame) -> pd.DataFrame:
    return (
        null_df.groupby(["experiment", "state", "source", "null_model"])
        .agg(
            n_rows=("region_id", "size"),
            n_subjects=("subject_id", "nunique"),
            delta_H1_total_persistence_mean=("delta_H1_total_persistence", "mean"),
            delta_H1_total_persistence_std=("delta_H1_total_persistence", "std"),
            delta_H1_max_lifetime_mean=("delta_H1_max_lifetime", "mean"),
            delta_H1_max_lifetime_std=("delta_H1_max_lifetime", "std"),
            p_lt_005_rate=(
                "empirical_p_H1_total_persistence",
                lambda s: float(np.mean(s < 0.05)),
            ),
            p_lt_010_rate=(
                "empirical_p_H1_total_persistence",
                lambda s: float(np.mean(s < 0.10)),
            ),
        )
        .reset_index()
    )


def build_subjectwise_comparison(actual_df: pd.DataFrame) -> pd.DataFrame:
    """Compare region sets after averaging within each subject."""
    subject_mean = (
        actual_df.groupby(["experiment", "state", "subject_id", "source"])
        .agg(
            H1_features=("H1_features", "mean"),
            H1_total_persistence=("H1_total_persistence", "mean"),
            H1_max_lifetime=("H1_max_lifetime", "mean"),
        )
        .reset_index()
    )
    metrics = ["H1_features", "H1_total_persistence", "H1_max_lifetime"]
    rows = []

    for (experiment, state), group in subject_mean.groupby(["experiment", "state"]):
        wide = group.pivot(index="subject_id", columns="source", values=metrics)
        row = {"experiment": experiment, "state": state, "n_subjects": len(wide)}
        for metric in metrics:
            diff = wide[(metric, "low")] - wide[(metric, "top")]
            row[f"{metric}_mean_low_minus_top"] = float(diff.mean())
            row[f"{metric}_rate_low_gt_top"] = float((diff > 0).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["experiment", "state"])


def _style_boxplot(bp) -> None:
    for patch, color in zip(bp["boxes"], [SOURCE_COLORS["low"], SOURCE_COLORS["top"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)


def make_actual_comparison_figures(actual_df: pd.DataFrame) -> dict[str, object]:
    """Create pooled P(D_1) figures without saving them to disk."""
    import matplotlib.pyplot as plt

    figures: dict[str, object] = {}

    for experiment in sorted(actual_df["experiment"].unique()):
        states = EXPERIMENT_STATES[experiment]
        fig, axes = plt.subplots(1, len(states), figsize=(7 * len(states), 5), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, state in zip(axes, states):
            state_df = actual_df[
                (actual_df["experiment"] == experiment) & (actual_df["state"] == state)
            ]
            low = state_df[state_df["source"] == "low"]["H1_total_persistence"].to_numpy()
            top = state_df[state_df["source"] == "top"]["H1_total_persistence"].to_numpy()

            bp = ax.boxplot(
                [low, top],
                tick_labels=[SOURCE_LABELS["low"], SOURCE_LABELS["top"]],
                patch_artist=True,
            )
            _style_boxplot(bp)
            ax.set_title(f"{experiment}: {STATE_LABELS[state]}")
            ax.set_ylabel(r"$P(D_1)$")
            ax.grid(alpha=0.25)

        fig.tight_layout()
        figures[f"{experiment.lower()}_actual_comparison.png"] = fig

    return figures


def make_null_comparison_figures(null_df: pd.DataFrame) -> dict[str, object]:
    """Create original-minus-surrogate P(D_1) figures without saving them to disk."""
    import matplotlib.pyplot as plt

    figures: dict[str, object] = {}
    null_model = str(null_df["null_model"].iloc[0])

    for experiment in sorted(null_df["experiment"].unique()):
        states = EXPERIMENT_STATES[experiment]
        fig, axes = plt.subplots(1, len(states), figsize=(7 * len(states), 5), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, state in zip(axes, states):
            state_df = null_df[
                (null_df["experiment"] == experiment) & (null_df["state"] == state)
            ]
            low = state_df[state_df["source"] == "low"]["delta_H1_total_persistence"].to_numpy()
            top = state_df[state_df["source"] == "top"]["delta_H1_total_persistence"].to_numpy()

            bp = ax.boxplot(
                [low, top],
                tick_labels=[SOURCE_LABELS["low"], SOURCE_LABELS["top"]],
                patch_artist=True,
            )
            _style_boxplot(bp)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{experiment}: {STATE_LABELS[state]}")
            ax.set_ylabel(r"$P(D_1)$ original - surrogate mean")
            ax.grid(alpha=0.25)

        fig.tight_layout()
        if null_model == "shuffle":
            filename = f"{experiment.lower()}_null_comparison.png"
        else:
            filename = f"{experiment.lower()}_{null_model}_null_comparison.png"
        figures[filename] = fig

    return figures
