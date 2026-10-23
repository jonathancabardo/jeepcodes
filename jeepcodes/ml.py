"""Reproducible synthetic driving-time regression, independent of route search."""

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from .data import Network, ValidationError, _number


PERIODS = (("off_peak", 0.90), ("morning_peak", 1.30),
           ("midday", 1.05), ("evening_peak", 1.40))
DISRUPTIONS = (1.0, 1.5, 2.5)
FEATURES = ("base_minutes", "congestion_multiplier", "disruption_factor",
            *(f"period_{name}" for name, _ in PERIODS))
LIMITATIONS = (
    "All observations are simulated classroom examples, not measured traffic.",
    "Targets are generated from a known recipe using the same inputs available to the model; "
    "low errors mainly show recovery of that recipe, not real-world generalization.",
    "Groups prevent repeated versions of a service segment crossing the split. "
    "Different services can still share physical stops or equivalent numeric inputs.",
    "The same synthetic conditions occur in both partitions; this is not a future-time "
    "or unseen-disruption test. Only one fixed holdout is evaluated, without tuning on it.",
    "Noise is independently simulated; real traffic may have correlated delays, "
    "unknown causes, different distributions, and conditions outside this training range.",
    "The required naive baseline ignores time-of-day and disruption factors. "
    "The known generation recipe could be used directly; ML is an instructional comparison.",
    "Prediction covers driving minutes only. Arrival and transfer costs are excluded. "
    "ML predictions are not connected to routing until Step 5.",
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    replicates: int = 10
    test_fraction: float = 0.2
    noise_fraction: float = 0.05
    forest_trees: int = 200

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValidationError("ML seed must be an integer from 0 to 4294967295.")
        for name, value in (("replicates", self.replicates), ("forest trees", self.forest_trees)):
            if type(value) is not int or value < 1:
                raise ValidationError(f"ML {name} must be a positive integer.")
        if not 0 < _number(self.test_fraction, "ML test fraction") < 1:
            raise ValidationError("ML test fraction must be strictly between 0 and 1.")
        if not 0 <= _number(self.noise_fraction, "ML noise fraction") < 1:
            raise ValidationError("ML noise fraction must be from 0 up to, but excluding, 1.")


@dataclass(frozen=True)
class Observation:
    sample_id: int
    group: str
    source_row: int
    service: str
    endpoint_a: str
    endpoint_b: str
    base_minutes: float
    congestion_multiplier: float
    period: str
    disruption_factor: float
    noise_factor: float
    target_minutes: float

    @property
    def features(self) -> tuple[float, ...]:
        # Never include target, realized noise, source row, code, group, or stop names.
        return (self.base_minutes, self.congestion_multiplier, self.disruption_factor,
                *(float(self.period == name) for name, _ in PERIODS))


@dataclass(frozen=True)
class Metrics:
    mae_minutes: float
    rmse_minutes: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    observations: tuple[Observation, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    predictions: tuple[float, ...]
    naive_predictions: tuple[float, ...]
    model_metrics: Metrics
    naive_metrics: Metrics
    model: RandomForestRegressor
    source_sha256: str | None

    def report(self) -> dict:
        rows = [asdict(row) for row in self.observations]
        digest = sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
        return {
            "synthetic": True, "target": "segment driving minutes; excludes access/transfer delays",
            "source_sha256": self.source_sha256, "dataset_sha256": digest,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
            "config": asdict(self.config), "feature_names": list(FEATURES),
            "recipe": {"formula": "base_minutes * congestion_multiplier * period_factor * disruption_factor * noise_factor",
                       "period_factors": dict(PERIODS), "disruption_factors": list(DISRUPTIONS),
                       "noise": f"independent uniform [{1-self.config.noise_fraction:g}, {1+self.config.noise_fraction:g}]"},
            "sample_count": len(rows),
            "group_count": len({row.group for row in self.observations}),
            "split_strategy": "GroupShuffleSplit: code + unordered source endpoints; all simulated variants together",
            "train_sample_count": len(self.train_indices), "test_sample_count": len(self.test_indices),
            "train_groups": sorted({self.observations[i].group for i in self.train_indices}),
            "test_groups": sorted({self.observations[i].group for i in self.test_indices}),
            "test_sample_ids": [self.observations[i].sample_id for i in self.test_indices],
            "model": {"name": "RandomForestRegressor", "parameters": self.model.get_params()},
            "model_metrics": asdict(self.model_metrics), "naive_metrics": asdict(self.naive_metrics),
            "naive_formula": "base_minutes * congestion_multiplier",
            "limitations": list(LIMITATIONS),
        }


def generate_observations(network: Network, config: ExperimentConfig = ExperimentConfig()) -> tuple[Observation, ...]:
    config.validate()
    if network.scenario_id != "S1":
        raise ValidationError("Generate ML examples from the original S1 source network.")
    rng = np.random.default_rng(config.seed)
    records, groups = [], set()
    edges = sorted((edge for edge in network.segments if edge.status == "OPEN"),
                   key=lambda edge: (edge.code, *sorted((edge.origin, edge.destination))))
    for edge in edges:
        if edge.disruption_multiplier != 1:
            raise ValidationError("ML source segments must not already include a disruption overlay.")
        a, b = sorted((edge.origin, edge.destination))
        group = json.dumps([edge.code, a, b], separators=(",", ":"))
        if group in groups:
            raise ValidationError("Duplicate ML source segment, including reversed endpoints.")
        groups.add(group)
        for period, period_factor in PERIODS:
            for disruption in DISRUPTIONS:
                for _ in range(config.replicates):
                    noise = float(rng.uniform(1-config.noise_fraction, 1+config.noise_fraction))
                    target = edge.travel_minutes * edge.congestion_multiplier * period_factor * disruption * noise
                    if not np.isfinite(target) or target < 0:
                        raise ValidationError("Synthetic target must be finite and nonnegative.")
                    records.append(Observation(len(records), group, edge.source_row, edge.code, a, b,
                                               edge.travel_minutes, edge.congestion_multiplier,
                                               period, disruption, noise, target))
    if len(groups) < 2:
        raise ValidationError("At least two OPEN source segment groups are needed for an ML holdout.")
    return tuple(records)


def calculate_metrics(actual, predicted) -> Metrics:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (actual.ndim != 1 or actual.size == 0 or predicted.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual < 0).any() or (predicted < 0).any()):
        raise ValidationError("Evaluation needs equally sized finite nonnegative target/prediction vectors.")
    return Metrics(float(mean_absolute_error(actual, predicted)),
                   float(root_mean_squared_error(actual, predicted)))


def run_experiment(network: Network, config: ExperimentConfig = ExperimentConfig()) -> ExperimentResult:
    observations = generate_observations(network, config)
    x = np.asarray([row.features for row in observations], dtype=float)
    y = np.asarray([row.target_minutes for row in observations], dtype=float)
    groups = np.asarray([row.group for row in observations])
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    try:
        train, test = next(splitter.split(x, y, groups))
    except ValueError as exc:
        raise ValidationError(f"ML split cannot form nonempty train/test groups: {exc}") from exc
    model = RandomForestRegressor(n_estimators=config.forest_trees, max_depth=12,
                                  min_samples_leaf=2, max_features=1.0, bootstrap=True,
                                  n_jobs=1, random_state=config.seed)
    # Fixed one-hot schema; no scaler, encoder, or other fitted preprocessing.
    model.fit(x[train], y[train])
    predicted = model.predict(x[test])
    naive = x[test, 0] * x[test, 1]
    return ExperimentResult(config, observations, tuple(map(int, train)), tuple(map(int, test)),
                            tuple(map(float, predicted)), tuple(map(float, naive)),
                            calculate_metrics(y[test], predicted), calculate_metrics(y[test], naive),
                            model, network.source_sha256)


def export_experiment(result: ExperimentResult, directory: str | Path) -> Path:
    """Write a new output folder; never replace existing files or input sources."""
    directory = Path(directory).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ValidationError(f"Choose a new ML output directory: {exc}") from exc
    try:
        report = result.report()
        (directory / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        test_set = set(result.test_indices)
        with (directory / "synthetic_observations.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[*Observation.__dataclass_fields__, "split", "synthetic"])
            writer.writeheader()
            for i, row in enumerate(result.observations):
                writer.writerow({**asdict(row), "split": "test" if i in test_set else "train", "synthetic": True})
        with (directory / "heldout_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("sample_id", "group", "target_minutes", "model_minutes", "naive_minutes"))
            for i, predicted, naive in zip(result.test_indices, result.predictions, result.naive_predictions):
                row = result.observations[i]
                writer.writerow((row.sample_id, row.group, row.target_minutes, predicted, naive))
        joblib.dump({"model": result.model, "feature_names": FEATURES, "report": report},
                    directory / "model.joblib")
    except (OSError, ValueError) as exc:
        raise ValidationError(f"ML export failed; partial files may remain in {directory}: {exc}") from exc
    return directory
