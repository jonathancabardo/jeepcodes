"""Compare deterministic and learned driving times using the same route engine."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np
import sklearn

from .data import Network, ValidationError
from .ml import FEATURES, PERIODS
from .routing import search_routes
from .scenarios import Comparison, Query, apply_scenario, compare_results


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "step4-ml" / "model.joblib"
# Fingerprint of the model generated and verified in Step 4. Check before deserializing.
VERIFIED_MODEL_SHA256 = "06a8b0f80d0ad44fb1719661bcca31e5ae37d6cdcfaa615b3976bcacbe784b2a"


@dataclass(frozen=True)
class Predictor:
    model: object
    source_sha256: str | None
    model_sha256: str


def load_predictor(network: Network, path: str | Path | None = None) -> Predictor:
    """Load the verified Step 4 bytes; never silently retrain or substitute a model."""
    path = Path(path) if path is not None else DEFAULT_MODEL
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"ML model unavailable at {path}: {exc}") from exc
    digest = sha256(data).hexdigest()
    if digest != VERIFIED_MODEL_SHA256:
        raise ValidationError("ML model fingerprint differs from the verified Step 4 artifact. Restore that model before continuing.")
    if sklearn.__version__ != "1.9.1":
        raise ValidationError("The saved Step 4 model requires its recorded scikit-learn 1.9.1 environment.")
    try:
        bundle = joblib.load(BytesIO(data))
        report = bundle["report"]
        model = bundle["model"]
        if (tuple(bundle["feature_names"]) != FEATURES or tuple(report["feature_names"]) != FEATURES
                or report["source_sha256"] != network.source_sha256 or report["synthetic"] is not True
                or report["recipe"]["period_factors"] != dict(PERIODS)
                or getattr(model, "n_features_in_", None) != len(FEATURES)
                or not callable(getattr(model, "predict", None))):
            raise ValidationError("Saved ML schema/source does not match the current workbook and feature contract.")
    except (KeyError, TypeError, ValueError, AttributeError, ImportError, OSError, EOFError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"ML model could not be loaded: {exc}") from exc
    return Predictor(model, network.source_sha256, digest)


def time_network(network: Network, period: str, predictor: Predictor | None = None) -> Network:
    """Produce a driving-time view; node values, fares, risk, statuses, and source values stay intact."""
    if period not in dict(PERIODS):
        raise ValidationError(f"Unknown ML time period: {period!r}.")
    if any(e.driving_minutes_override is not None for e in network.segments):
        raise ValidationError("Build driving-time views from source/scenario segments, not an existing time view.")
    if predictor is not None and predictor.source_sha256 != network.source_sha256:
        raise ValidationError("ML model source fingerprint does not match the network.")
    open_edges = tuple(e for e in network.segments if e.status == "OPEN")
    if predictor is None:
        times = [edge.driving_minutes * dict(PERIODS)[period] for edge in open_edges]
    elif not open_edges:
        times = []
    else:
        # Source multipliers stay separate from disruption, in exactly the Step 4 order.
        features = np.asarray([
            (edge.travel_minutes, edge.congestion_multiplier, edge.disruption_multiplier,
             *(float(period == name) for name, _ in PERIODS)) for edge in open_edges
        ], dtype=float)
        try:
            raw = np.asarray(predictor.model.predict(features))
        except (ValueError, TypeError, RuntimeError, AttributeError, OverflowError) as exc:
            raise ValidationError(f"ML driving-time prediction failed: {exc}") from exc
        if (raw.shape != (len(open_edges),) or raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all() or (raw < 0).any()):
            raise ValidationError("ML must return one finite nonnegative driving-time prediction per OPEN source segment.")
        times = list(map(float, raw))
    predictions = {edge.source_id: value for edge, value in zip(open_edges, times)}
    segments = tuple(replace(edge, driving_minutes_override=predictions.get(edge.source_id))
                     for edge in network.segments)
    # Validate deterministic overflow as well as predictions before entering search.
    for edge in segments:
        if edge.status == "OPEN":
            edge.driving_minutes
    adjacency = {place: [] for place in network.nodes}
    for edge in segments:
        adjacency[edge.origin].append(edge)
        adjacency[edge.destination].append(replace(edge, origin=edge.destination, destination=edge.origin))
    return replace(network, segments=segments,
                   adjacency=MappingProxyType({p: tuple(edges) for p, edges in adjacency.items()}))


@dataclass(frozen=True)
class RoutingStudy:
    query: Query
    period: str
    source_sha256: str | None
    model_sha256: str
    normal: Comparison
    delta: Comparison
    deterministic_reroute: Comparison
    ml_reroute: Comparison

    def report(self) -> dict:
        def path_data(path):
            if path is None:
                return None
            return {"stops": list(path.stops), "identity": [list(leg) for leg in path.identity],
                    "cost_components": asdict(path.costs), "minutes": path.costs.minutes,
                    "fare": path.costs.fare, "risk": path.costs.risk,
                    "transfers": path.costs.transfers, "balanced_score": path.costs.score}

        def changes(before, after):
            if before is None or after is None:
                return None
            return {field: getattr(after.costs, field) - getattr(before.costs, field)
                    for field in ("minutes", "fare", "risk", "transfers", "score")}

        def comparison_data(comparison):
            before, after = comparison.baseline.itineraries, comparison.current.itineraries
            return {
                "before_message": comparison.baseline.message, "after_message": comparison.current.message,
                "before_ranked_routes": [path_data(path) for path in before],
                "after_ranked_routes": [path_data(path) for path in after],
                "winner_changed": (before[0].identity != after[0].identity) if before and after else None,
                "winner_metric_deltas_after_minus_before": changes(before[0], after[0]) if before and after else None,
                "same_route_comparisons": [{"identity": [list(leg) for leg in row.identity],
                                            "before": path_data(row.before), "after": path_data(row.after),
                                            "deltas_after_minus_before": changes(row.before, row.after),
                                            "before_unavailable": row.before_unavailable,
                                            "after_unavailable": row.after_unavailable}
                                           for row in comparison.routes],
                "explanation": comparison.explanation,
            }
        return {"synthetic": True, "query": asdict(self.query), "period": self.period,
                "period_factor": dict(PERIODS)[self.period], "source_sha256": self.source_sha256,
                "model_sha256": self.model_sha256,
                "deterministic_contract": "base * source congestion * period factor * disruption factor; then shared node costs",
                "ml_contract": "predict with period/disruption inputs; use output once, then shared node costs",
                "normal_deterministic_vs_ml": comparison_data(self.normal),
                "delta_deterministic_vs_ml": comparison_data(self.delta),
                "deterministic_normal_vs_delta": comparison_data(self.deterministic_reroute),
                "ml_normal_vs_delta": comparison_data(self.ml_reroute),
                "limitations": ["Predictions are learned from synthetic classroom data, not real traffic.",
                                "ML-vs-deterministic deltas compare estimates; they are not measured time savings.",
                                "The Step 4 holdout model is reused unchanged; no retraining or new accuracy claim.",
                                "An ML Delta prediction need not be exactly 2.5 times its normal prediction."]}


def compare_routing(network: Network, predictor: Predictor, query: Query = Query(), period: str = "midday") -> RoutingStudy:
    if not isinstance(predictor, Predictor) or not callable(getattr(predictor.model, "predict", None)):
        raise ValidationError("A loaded ML predictor is required; no deterministic fallback is used.")
    source = apply_scenario(network, "S1")
    disrupted = apply_scenario(source, "S3")
    deterministic_normal = time_network(source, period)
    ml_normal = time_network(source, period, predictor)
    deterministic_delta = time_network(disrupted, period)
    ml_delta = time_network(disrupted, period, predictor)
    views = (deterministic_normal, ml_normal, deterministic_delta, ml_delta)
    results = tuple(search_routes(view, query.origin, query.destination, query.preference, query.budget)
                    for view in views)
    d0, m0, d1, m1 = results
    return RoutingStudy(query, period, network.source_sha256, predictor.model_sha256,
                        compare_results(deterministic_normal, ml_normal, d0, m0, "S1 ML"),
                        compare_results(deterministic_delta, ml_delta, d1, m1, "S3 ML"),
                        compare_results(deterministic_normal, deterministic_delta, d0, d1, "S3 deterministic"),
                        compare_results(ml_normal, ml_delta, m0, m1, "S3 ML"))


def export_routing_study(study: RoutingStudy, directory: str | Path) -> Path:
    directory = Path(directory).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ValidationError(f"Choose a new routing-comparison output directory: {exc}") from exc
    try:
        (directory / "comparison.json").write_text(
            json.dumps(study.report(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Routing comparison export failed in {directory}: {exc}") from exc
    return directory
