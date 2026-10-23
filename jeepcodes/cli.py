"""Network inspection, deterministic routing, and scenario comparisons."""

import argparse
from pathlib import Path
import sys

from .data import SOURCE_SHA256, ValidationError, load_network
from .routing import OBJECTIVES, risk_weighted_score, search_routes
from .scenarios import ASSUMPTIONS, PENDING_SCENARIOS, SCENARIOS, Query, ScenarioSession


DEFAULT_WORKBOOK = Path(__file__).resolve().parents[2] / "JeepCodes_2_AI_Route_Challenge.xlsx"


def _print_result(result, title="Baseline search"):
    print(f"{title}: {result.origin} -> {result.destination}")
    print(f"Objective ({result.preference}): {OBJECTIVES[result.preference]}")
    print("Ties: balanced score, minutes, fare, risk, transfers, then lexicographic leg identity (duplicate fields omitted).")
    print(f"Entered budget: PHP {result.entered_budget:g}; effective budget: PHP {result.effective_budget:g}")
    if result.max_transfers is not None:
        print(f"Maximum transfers: {result.max_transfers}")
    print(result.message)
    for index, path in enumerate(result.itineraries):
        label = "Recommendation" if index == 0 else f"Alternative {index}"
        costs = path.costs
        if result.preference == "Risk-weighted":
            print(f"  Approved risk-weighted score: {risk_weighted_score(costs):.9f}; original Balanced score follows.")
        print(f"{label}: {' -> '.join(path.stops)}")
        print(f"  Fare PHP {costs.fare:.6f}; minutes {costs.minutes:.6f}; synthetic risk {costs.risk:.6f}; transfers {costs.transfers}; balanced score {costs.score:.9f}")
        print(f"  Fare breakdown: boarding {costs.boarding_fare:g} + segments {costs.segment_fare:g} + transfers {costs.transfer_fare:g}")
        print(f"  Minutes breakdown: driving {costs.driving_minutes:g} + arrivals {costs.arrival_minutes:g} + transfers {costs.transfer_minutes:g}")
        print("  Score parts (time, fare, risk, transfers): " + ", ".join(f"{part:.9f}" for part in costs.score_parts))
        for leg in path.legs:
            edge = leg.segment
            print(f"    {edge.origin} -> {edge.destination}: {edge.code} (EDGE COSTS row {edge.source_row}); fare {leg.costs.fare:g}, minutes {leg.costs.minutes:g}, risk {leg.costs.risk:g}")
    print(f"Expanded path labels: {result.expanded_labels}")


def _print_comparison(comparison, title=None):
    print(title or f"Before/after comparison: S1 -> {comparison.scenario}")
    for index, row in enumerate(comparison.routes, 1):
        identity = "; ".join(f"{a}->{b} on {code}" for a, b, code in row.identity) or "Zero-leg trip"
        print(f"Comparison route {index}: {identity}")
        for label, path, reason in (("Before", row.before, row.before_unavailable),
                                    ("After", row.after, row.after_unavailable)):
            if path is None:
                print(f"  {label}: unavailable - {reason}")
            else:
                c = path.costs
                print(f"  {label}: minutes {c.minutes:.6f}; fare PHP {c.fare:g}; risk {c.risk:.6f}; transfers {c.transfers}; score {c.score:.9f}")
        if row.before is not None and row.after is not None:
            old, new = row.before.costs, row.after.costs
            print(f"  Change (after - before): minutes {new.minutes-old.minutes:+.6f}; fare {new.fare-old.fare:+g}; risk {new.risk-old.risk:+.6f}; transfers {new.transfers-old.transfers:+d}; score {new.score-old.score:+.9f}")
    print("Explanation: " + comparison.explanation)
    if comparison.uncertainty is not None:
        report = comparison.uncertainty
        print(f"S8 uncertainty: {report.simulations} simulations; seed {report.seed}; uniform [0.7, 1.3].")
        for i,row in enumerate(report.routes):
            print(f"  Route {i+1}: mean {row.mean_minutes:.6f} min; range {row.minimum_minutes:.6f} to {row.maximum_minutes:.6f}; std {row.std_minutes:.6f}; mean Balanced score {row.mean_balanced_score:.9f}; wins among returned routes {row.wins_among_returned_routes}.")
        print(report.note)


def _print_events(events):
    print("Event log (source segment changes apply in both directions):")
    for event in events:
        print(f"  Event {event.sequence}: {event.previous_scenario or 'start'} -> {event.scenario}; trigger: {event.trigger}")
        print(f"    Recalculated {event.itinerary_count} feasible itinerary/itineraries.")
        if not event.changes:
            print("    No network cost/status changes.")
        for change in event.changes:
            location = (f"{change.sheet} row {change.row}" if change.row is not None else change.sheet)
            service = f" {change.service}: {change.origin} <-> {change.destination}" if change.service else f" {change.origin or ''}"
            old = f"{change.before:.9g}" if isinstance(change.before, float) else change.before
            new = f"{change.after:.9g}" if isinstance(change.after, float) else change.after
            print(f"    {location}{service}; {change.field}: {old} -> {new}")
        print(f"    Outcome: {event.outcome}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and reroute the synthetic JeepCodes classroom network.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--code", help="Display ordered stops for an exact jeepney code, e.g. 01A.")
    parser.add_argument("--stop", help="Validate an exact stop name and show its OPEN connections.")
    parser.add_argument("--origin", help="Start a route search from this exact stop name.")
    parser.add_argument("--destination", help="Destination (defaults to Romeo when --origin is supplied).")
    parser.add_argument("--preference", choices=OBJECTIVES, default="Balanced")
    parser.add_argument("--budget", type=float, default=40, help="Nonnegative fare budget; capped at PHP 40.")
    parser.add_argument("--scenario", choices=sorted(set(SCENARIOS) | set(PENDING_SCENARIOS)),
                        help="Compare S1-S9 with stored baseline; S4/S6/S8/S9 use approved modeling assumptions.")
    parser.add_argument("--reset", action="store_true", help="Restore S1 after the selected scenario and show its log.")
    parser.add_argument("--list-scenarios", action="store_true", help="List implemented and pending scenarios.")
    parser.add_argument("--ml-experiment", action="store_true", help="Run Step 4 synthetic regression evaluation only.")
    parser.add_argument("--ml-output", type=Path, help="New directory for dataset, evaluation, predictions, and model.")
    parser.add_argument("--ml-seed", type=int, help="Reproducibility seed (default 42).")
    parser.add_argument("--ml-replicates", type=int, help="Synthetic examples per segment/condition (default 10).")
    parser.add_argument("--ml-routes", action="store_true", help="Compare deterministic/ML routing in normal and Delta conditions.")
    parser.add_argument("--ml-period", choices=("off_peak", "morning_peak", "midday", "evening_peak"),
                        help="Matched time-of-day condition for both predictors (default midday).")
    parser.add_argument("--ml-model", type=Path, help="Location of the verified Step 4 model (defaults to saved project artifact).")
    parser.add_argument("--ml-routing-output", type=Path, help="New directory for the route-comparison JSON report.")
    args = parser.parse_args(argv)
    if args.ml_routes or any(v is not None for v in (args.ml_period, args.ml_model, args.ml_routing_output)):
        if not args.ml_routes:
            print("Validation error: routing-model options require --ml-routes.", file=sys.stderr)
            return 2
        if (args.ml_experiment or args.ml_output is not None or args.ml_seed is not None
                or args.ml_replicates is not None or args.scenario is not None or args.reset
                or args.list_scenarios or args.code is not None or args.stop is not None):
            print("Validation error: --ml-routes compares S1/S3; run experiments and other scenario/lookup options separately.", file=sys.stderr)
            return 2
        try:
            from .ml_routing import compare_routing, export_routing_study, load_predictor
        except ImportError:
            print("Validation error: install requirements.txt in the project environment for ML support.", file=sys.stderr)
            return 2
        try:
            network = load_network(args.workbook, expected_sha256=SOURCE_SHA256)
            predictor = load_predictor(network, args.ml_model)
            study = compare_routing(network, predictor, Query(
                "Alpha" if args.origin is None else args.origin,
                "Romeo" if args.destination is None else args.destination,
                args.preference, args.budget), args.ml_period or "midday")
            output = export_routing_study(study, args.ml_routing_output) if args.ml_routing_output is not None else None
        except ValidationError as exc:
            print(f"Validation error: {exc}", file=sys.stderr)
            return 2
        study_report = study.report()
        print("Step 5 - SYNTHETIC classroom ML-assisted routing comparison")
        print(f"Source SHA-256: {study.source_sha256}; model SHA-256: {study.model_sha256}")
        print(f"Matched time period: {study.period}; deterministic period factor: {study_report['period_factor']}")
        print("ML uses the condition-specific prediction once; shared access/transfer costs are then added.")
        for label, comparison in (("Normal S1", study.normal), ("Delta S3", study.delta)):
            _print_result(comparison.baseline, f"{label} / deterministic")
            _print_result(comparison.current, f"{label} / ML-assisted")
            _print_comparison(comparison, f"Deterministic vs ML / {label}")
        print("Deterministic normal-to-Delta: " + study.deterministic_reroute.explanation)
        print("ML normal-to-Delta: " + study.ml_reroute.explanation)
        for label in ("normal_deterministic_vs_ml", "delta_deterministic_vs_ml",
                      "deterministic_normal_vs_delta", "ml_normal_vs_delta"):
            delta = study_report[label]["winner_metric_deltas_after_minus_before"]
            print(f"Winner metric changes / {label}: {delta}")
        for limitation in study_report["limitations"]:
            print("Limitation: " + limitation)
        if output is not None:
            print(f"Saved routing comparison: {output}")
        return 0
    if args.ml_experiment or any(value is not None for value in (args.ml_output, args.ml_seed, args.ml_replicates)):
        if not args.ml_experiment:
            print("Validation error: ML options require --ml-experiment.", file=sys.stderr)
            return 2
        if (args.scenario is not None or args.reset or args.list_scenarios or args.origin is not None
                or args.destination is not None or args.code is not None or args.stop is not None
                or args.preference != "Balanced" or args.budget != 40):
            print("Validation error: run the ML experiment separately from route/scenario options.", file=sys.stderr)
            return 2
        try:
            from .ml import ExperimentConfig, export_experiment, run_experiment
        except ImportError:
            print("Validation error: install requirements.txt in the project environment for ML support.", file=sys.stderr)
            return 2
        try:
            network = load_network(args.workbook, expected_sha256=SOURCE_SHA256)
            config = ExperimentConfig(seed=42 if args.ml_seed is None else args.ml_seed,
                                      replicates=10 if args.ml_replicates is None else args.ml_replicates)
            experiment = run_experiment(network, config)
            report = experiment.report()
            output = export_experiment(experiment, args.ml_output) if args.ml_output is not None else None
        except ValidationError as exc:
            print(f"Validation error: {exc}", file=sys.stderr)
            return 2
        print("Step 4 - SYNTHETIC classroom ML experiment; not real traffic measurements")
        print(f"Samples: {report['sample_count']}; groups: {report['group_count']}; seed: {config.seed}")
        print(f"Training: {report['train_sample_count']} samples / {len(report['train_groups'])} groups")
        print(f"Testing: {report['test_sample_count']} samples / {len(report['test_groups'])} groups")
        print("Split: all examples of each service segment remain together; no shared train/test groups.")
        print("Target: driving minutes only, excluding access and transfer delays.")
        for name, metrics in (("Random forest", experiment.model_metrics), ("Naive base x multiplier", experiment.naive_metrics)):
            print(f"{name}: MAE {metrics.mae_minutes:.6f} min; RMSE {metrics.rmse_minutes:.6f} min")
        from .ml import LIMITATIONS
        for limitation in LIMITATIONS:
            print("Limitation: " + limitation)
        if output is not None:
            print(f"Saved experiment: {output}")
        return 0
    if args.list_scenarios:
        for scenario in sorted(set(SCENARIOS) | set(PENDING_SCENARIOS)):
            print(f"{scenario}: {SCENARIOS.get(scenario) or PENDING_SCENARIOS[scenario]}")
        return 0
    try:
        network = load_network(args.workbook, expected_sha256=SOURCE_SHA256)
        session = comparison = result = None
        if args.scenario is not None or args.reset:
            session = ScenarioSession(network, Query(
                "Alpha" if args.origin is None else args.origin,
                "Romeo" if args.destination is None else args.destination,
                args.preference, args.budget))
            comparison = session.activate(args.scenario or "S1")
            result = comparison.current
        elif args.origin is not None or args.destination is not None:
            origin = "Alpha" if args.origin is None else args.origin
            destination = "Romeo" if args.destination is None else args.destination
            result = search_routes(network, origin, destination, args.preference, args.budget)
        display_network = session.network if session is not None else network
        service = display_network.service(args.code) if args.code is not None else None
        node = display_network.require_stop(args.stop) if args.stop is not None else None
    except ValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2
    print("JeepCodes 2.0 - SYNTHETIC classroom data")
    print(f"Workbook: {network.source_path}")
    print(f"SHA-256: {network.source_sha256}")
    print(f"Stops: {len(network.nodes)}")
    print(f"Services: {len(network.services)}")
    print(f"Source service segments: {len(network.segments)}")
    print(f"Directed adjacency entries: {sum(map(len, network.adjacency.values()))}")
    if service is not None:
        print(f"{service.code}: {' -> '.join(service.stops)}")
    if node is not None:
        print(f"OPEN connections from {node.place}:")
        for edge in display_network.outgoing(node.place):
            print(f"  {edge.code}: {edge.origin} -> {edge.destination} (EDGE COSTS row {edge.source_row})")
    if comparison is not None:
        _print_result(comparison.baseline, "Stored baseline search")
        _print_result(result, f"Scenario {comparison.scenario}")
        _print_comparison(comparison)
        if args.reset:
            restored = session.reset()
            _print_result(restored.current, "Reset to S1")
            _print_comparison(restored)
        _print_events(session.events)
    elif result is not None:
        _print_result(result)
    return 0
