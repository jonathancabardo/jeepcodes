"""Deterministic scenario overlays and reproducible before/after rerouting."""

from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType

from .data import Network, ValidationError
from .routing import Itinerary, SearchResult, evaluate_itinerary, rank_key, search_routes


SCENARIOS = MappingProxyType({
    "S1": "Normal weekday / reset to baseline",
    "S2": "Close Foxtrot - Golf on every service, both directions",
    "S3": "Delta congestion: driving minutes x2.5 on touching segments",
    "S4": "Hotel risk 10; approved risk-weighted objective (30/20/40/10)",
    "S5": "01A service outage",
    "S6": "42D incremental fares x2; boarding and transfer fares unchanged",
    "S7": "Accessibility: 12-minute transfer delay; at most one transfer",
    "S8": "100 uniform +/-30% driving-time simulations; seed 42; mean Balanced ranking",
    "S9": "Close India-Juliet; Kilo driving x2; Oscar node risk 10",
})
PENDING_SCENARIOS = MappingProxyType({})
ASSUMPTIONS = MappingProxyType({
    "S4": "Approved modeling assumption: Hotel node risk 10; ranking weights time 30%, fare 20%, risk 40%, transfers 10%. The original Balanced score is still shown separately. S4 uses this objective regardless of the preference control.",
    "S6": "Approved modeling assumption: double only 42D incremental segment fares; keep each boarding/transfer charge at PHP 5.",
    "S8": "Approved modeling assumption: 100 simulations, seed 42, independent uniform [0.7, 1.3] driving-time factors per service segment per simulation, shared by both directions. Rank by mean Balanced score regardless of the preference control; show time variation. This is average performance, not worst-case protection.",
    "S9": "Approved modeling assumption: close India-Juliet on every service in both directions; double driving minutes touching Kilo; set Oscar node safety risk to 10. Other costs stay unchanged.",
})


@dataclass(frozen=True)
class Change:
    sheet: str
    row: int | None
    service: str | None
    origin: str | None
    destination: str | None
    field: str
    before: str | float | int | None
    after: str | float | int | None


def network_changes(before: Network, after: Network) -> tuple[Change, ...]:
    """Record each source segment once; its two directions share the change."""
    changes = []
    previous = {edge.source_id: edge for edge in before.segments}
    for edge in after.segments:
        old = previous[edge.source_id]
        for field, old_value, new_value in (
            ("status", old.status, edge.status),
            ("driving_minutes", old.driving_minutes, edge.driving_minutes),
            ("incremental_fare", old.incremental_fare, edge.incremental_fare),
        ):
            if old_value != new_value:
                changes.append(Change("EDGE COSTS", edge.source_row, edge.code,
                                      edge.origin, edge.destination, field, old_value, new_value))
    for place, node in after.nodes.items():
        old = before.nodes[place]
        for field in ("transfer_minutes", "safety_risk"):
            if getattr(old, field) != getattr(node, field):
                changes.append(Change("NODE COSTS", node.source_row, None, place, None,
                                      field, getattr(old, field), getattr(node, field)))
    if before.scenario_id != after.scenario_id and (before.scenario_id in ASSUMPTIONS or after.scenario_id in ASSUMPTIONS):
        changes.append(Change("SCENARIO RULES", None, None, None, None, "modeling_rule",
                              ASSUMPTIONS.get(before.scenario_id, before.scenario_id),
                              ASSUMPTIONS.get(after.scenario_id, after.scenario_id)))
    if before.max_transfers != after.max_transfers:
        changes.append(Change("CONSTRAINTS", None, None, None, None, "max_transfers",
                              before.max_transfers, after.max_transfers))
    return tuple(changes)


def apply_scenario(baseline: Network, scenario: str) -> Network:
    """Always start from the supplied baseline; never compose scenario effects."""
    if baseline.scenario_id != "S1" or any(e.driving_minutes_override is not None for e in baseline.segments):
        raise ValidationError("Scenario overlays require the original S1 baseline, not an active scenario.")
    if scenario in PENDING_SCENARIOS:
        raise ValidationError(f"{scenario}: {PENDING_SCENARIOS[scenario]}")
    if scenario not in SCENARIOS:
        raise ValidationError(f"Unknown scenario: {scenario!r}.")
    if scenario == "S1":
        return baseline
    if scenario == "S8":
        from .uncertainty import mean_network
        return mean_network(baseline)
    segments = []
    for edge in baseline.segments:
        if scenario == "S2" and {edge.origin, edge.destination} == {"Foxtrot", "Golf"}:
            edge = replace(edge, status="CLOSED")
        elif scenario == "S3" and "Delta" in (edge.origin, edge.destination):
            edge = replace(edge, disruption_multiplier=2.5)
            if not isfinite(edge.travel_minutes * edge.congestion_multiplier * edge.disruption_multiplier):
                raise ValidationError("Delta-adjusted driving minutes exceed the finite numeric range.")
        elif scenario == "S5" and edge.code == "01A":
            # Retain source identities for inspection; unavailable in both directions.
            edge = replace(edge, status="CLOSED")
        elif scenario == "S6" and edge.code == "42D":
            edge = replace(edge, incremental_fare=edge.incremental_fare * 2)
            if not isfinite(edge.incremental_fare):
                raise ValidationError("Doubled fare exceeds the finite numeric range.")
        elif scenario == "S9":
            if {edge.origin, edge.destination} == {"India", "Juliet"}:
                edge = replace(edge, status="CLOSED")
            if "Kilo" in (edge.origin, edge.destination):
                edge = replace(edge, disruption_multiplier=2.0)
                edge.driving_minutes  # Validate overflow before committing the overlay.
        segments.append(edge)
    nodes = (MappingProxyType({place: replace(node, transfer_minutes=12.0)
                               for place, node in baseline.nodes.items()})
             if scenario == "S7" else baseline.nodes)
    if scenario in ("S4", "S9"):
        affected = "Hotel" if scenario == "S4" else "Oscar"
        nodes = MappingProxyType({place: replace(node, safety_risk=10.0) if place == affected else node
                                  for place, node in nodes.items()})
    adjacency = {place: [] for place in nodes}
    for edge in segments:
        adjacency[edge.origin].append(edge)
        adjacency[edge.destination].append(replace(edge, origin=edge.destination,
                                                   destination=edge.origin))
    return replace(baseline, nodes=nodes, segments=tuple(segments),
                   adjacency=MappingProxyType({place: tuple(edges)
                                               for place, edges in adjacency.items()}),
                   max_transfers=1 if scenario == "S7" else baseline.max_transfers,
                   scenario_id=scenario)


@dataclass(frozen=True)
class Query:
    origin: str = "Alpha"
    destination: str = "Romeo"
    preference: str = "Balanced"
    budget: float = 40.0


@dataclass(frozen=True)
class RouteComparison:
    identity: tuple[tuple[str, str, str], ...]
    before: Itinerary | None
    after: Itinerary | None
    before_unavailable: str = ""
    after_unavailable: str = ""


@dataclass(frozen=True)
class Comparison:
    scenario: str
    baseline: SearchResult
    current: SearchResult
    routes: tuple[RouteComparison, ...]
    explanation: str
    uncertainty: object | None = None


@dataclass(frozen=True)
class ScenarioEvent:
    sequence: int
    scenario: str
    previous_scenario: str | None
    trigger: str
    source_sha256: str | None
    query: Query
    changes: tuple[Change, ...]
    outcome: str
    itinerary_count: int


def _reprice(network: Network, path: Itinerary, budget: float):
    edges = {(edge.origin, edge.destination, edge.code): edge
             for candidates in network.adjacency.values() for edge in candidates}
    try:
        mapped = tuple(edges[identity] for identity in path.identity)
        result = evaluate_itinerary(network, path.origin, mapped)
        if result.costs.fare > budget:
            return None, f"Fare PHP {result.costs.fare:g} exceeds budget PHP {budget:g}."
        return result, ""
    except (KeyError, ValidationError) as exc:
        return None, str(exc)


def _metrics(path: Itinerary) -> str:
    c = path.costs
    return (f"{c.minutes:.2f} minutes, PHP {c.fare:g}, risk {c.risk:.2f}, "
            f"{c.transfers} transfer(s), balanced score {c.score:.9f}")


def compare_results(baseline_network: Network, active_network: Network,
                    baseline: SearchResult, current: SearchResult, scenario: str) -> Comparison:
    """Compare the union of both top-three sets, including the old winner after change."""
    candidates = {path.identity: path for path in baseline.itineraries}
    candidates.update({path.identity: path for path in current.itineraries})
    routes = []
    for identity, path in candidates.items():
        old, old_reason = _reprice(baseline_network, path, baseline.effective_budget)
        new, new_reason = _reprice(active_network, path, current.effective_budget)
        routes.append(RouteComparison(identity, old, new, old_reason, new_reason))
    if not baseline.itineraries:
        explanation = "No baseline recommendation was feasible. " + current.message
    elif not current.itineraries:
        explanation = "The scenario has no feasible recommendation. " + current.message
        old = routes[0]
        explanation += " Original winner is unavailable: " + old.after_unavailable
    else:
        old_winner, winner = baseline.itineraries[0], current.itineraries[0]
        changed = old_winner.identity != winner.identity
        explanation = ("Recommendation changed. " if changed else "Recommendation unchanged. ")
        explanation += f"Baseline winner: {_metrics(old_winner)}. "
        old_after = next(row for row in routes if row.identity == old_winner.identity)
        explanation += (f"Original winner under {scenario}: {_metrics(old_after.after)}. "
                        if old_after.after is not None else
                        f"Original winner under {scenario} is unavailable: {old_after.after_unavailable}. ")
        explanation += f"Current winner: {_metrics(winner)}. "
        explanation += f"{current.preference} is the active objective; lower values rank first. "
        for index, alternative in enumerate(current.itineraries[1:], 1):
            primary = rank_key(alternative, current.preference)[0]
            winning_primary = rank_key(winner, current.preference)[0]
            reason = (f"its primary value {primary:.9f} is higher than {winning_primary:.9f}"
                      if primary > winning_primary else "the documented tie rules rank it later")
            explanation += f"Alternative {index}: {_metrics(alternative)}; {reason}. "
        if not current.itineraries[1:]:
            explanation += "No additional feasible itinerary exists."
    return Comparison(scenario, baseline, current, tuple(routes), explanation)


class ScenarioSession:
    """A fixed-query session with an immutable baseline and append-only event history."""

    def __init__(self, baseline: Network, query: Query = Query()):
        if baseline.scenario_id != "S1" or any(e.driving_minutes_override is not None for e in baseline.segments):
            raise ValidationError("A session must start from the original S1 baseline.")
        self._baseline = baseline
        self._query = query
        self._baseline_result = search_routes(baseline, query.origin, query.destination,
                                              query.preference, query.budget)
        self._network = baseline
        self._result = self._baseline_result
        self._scenario = "S1"
        self._events = (ScenarioEvent(1, "S1", None, "Initial baseline calculation",
                                     baseline.source_sha256, query, (), self._result.message,
                                     len(self._result.itineraries)),)

    @property
    def baseline(self) -> Network:
        return self._baseline

    @property
    def baseline_result(self) -> SearchResult:
        return self._baseline_result

    @property
    def network(self) -> Network:
        return self._network

    @property
    def result(self) -> SearchResult:
        return self._result

    @property
    def events(self) -> tuple[ScenarioEvent, ...]:
        return self._events

    def activate(self, scenario: str, trigger: str = "User selected scenario") -> Comparison:
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValidationError("A recalculation trigger is required.")
        updated = apply_scenario(self._baseline, scenario)
        query = self._query
        preference = "Risk-weighted" if scenario == "S4" else "Balanced" if scenario == "S8" else query.preference
        result = search_routes(updated, query.origin, query.destination, preference, query.budget)
        comparison = compare_results(self._baseline, updated, self._baseline_result, result, scenario)
        if scenario in ASSUMPTIONS:
            comparison = replace(comparison, explanation=ASSUMPTIONS[scenario] + " " + comparison.explanation)
        if scenario == "S8":
            from .uncertainty import summarize
            comparison = replace(comparison, uncertainty=summarize(self._baseline, result))
        changes = network_changes(self._network, updated)
        event = ScenarioEvent(len(self._events) + 1, scenario, self._scenario, trigger,
                              self._baseline.source_sha256, query, changes,
                              comparison.explanation, len(result.itineraries))
        # Commit only once every calculation and validation has succeeded.
        self._network, self._result, self._scenario = updated, result, scenario
        self._events += (event,)
        return comparison

    def reset(self) -> Comparison:
        return self.activate("S1", trigger="User reset to baseline")
