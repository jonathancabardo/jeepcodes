"""Baseline cost rules and uniform-cost search over complete path labels."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite

from .data import Network, Segment, ValidationError, _number


OBJECTIVES = {
    "Balanced": "Minimize 0.40*time/120 + 0.30*fare/40 + 0.20*risk/100 + 0.10*transfers/4.",
    "Risk-weighted": "Approved S4 objective: minimize 0.30*time/120 + 0.20*fare/40 + 0.40*risk/100 + 0.10*transfers/4; original Balanced score remains separate.",
    "Fare": "Minimize total fare, including first boarding and transfers.",
    "Time": "Minimize total minutes, including driving, arrivals, and transfers.",
    "Safety": "Minimize synthetic risk points: edge risk + 0.1*destination safety risk per leg.",
}


@dataclass(frozen=True)
class Costs:
    boarding_fare: float = 0.0
    segment_fare: float = 0.0
    transfer_fare: float = 0.0
    driving_minutes: float = 0.0
    arrival_minutes: float = 0.0
    transfer_minutes: float = 0.0
    risk: float = 0.0
    transfers: int = 0

    @property
    def fare(self) -> float:
        return self.boarding_fare + self.segment_fare + self.transfer_fare

    @property
    def minutes(self) -> float:
        return self.driving_minutes + self.arrival_minutes + self.transfer_minutes

    @property
    def score_parts(self) -> tuple[float, float, float, float]:
        return (0.40 * (self.minutes / 120), 0.30 * (self.fare / 40),
                0.20 * (self.risk / 100), 0.10 * (self.transfers / 4))

    @property
    def score(self) -> float:
        return sum(self.score_parts)

    def plus(self, other: "Costs") -> "Costs":
        result = Costs(**{name: getattr(self, name) + getattr(other, name)
                          for name in self.__dataclass_fields__})
        if not all(isfinite(value) for value in (
                result.fare, result.minutes, result.risk, result.score)):
            raise ValidationError("Calculated trip costs exceed the supported finite numeric range.")
        return result


@dataclass(frozen=True)
class Leg:
    segment: Segment
    costs: Costs


@dataclass(frozen=True)
class Itinerary:
    origin: str
    legs: tuple[Leg, ...] = ()
    costs: Costs = Costs()

    @property
    def stops(self) -> tuple[str, ...]:
        return (self.origin,) + tuple(leg.segment.destination for leg in self.legs)

    @property
    def identity(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((leg.segment.origin, leg.segment.destination, leg.segment.code)
                     for leg in self.legs)


@dataclass(frozen=True)
class SearchResult:
    origin: str
    destination: str
    preference: str
    entered_budget: float
    effective_budget: float
    itineraries: tuple[Itinerary, ...]
    expanded_labels: int
    max_transfers: int | None = None

    @property
    def message(self) -> str:
        if not self.itineraries:
            limit = (f" and maximum {self.max_transfers} transfer(s)"
                     if self.max_transfers is not None else "")
            return f"No feasible route meets the effective budget of PHP {self.effective_budget:g}{limit}."
        if self.origin == self.destination:
            return "Zero-leg trip: no boarding or travel costs; no alternatives are needed."
        alternatives = len(self.itineraries) - 1
        if alternatives < 2:
            return f"Found the best feasible route and {alternatives} alternative(s); fewer than two alternatives exist."
        return "Found the best feasible route and two distinct alternatives."


def leg_costs(network: Network, edge: Segment, current_service: str | None) -> Costs:
    """Calculate one actual OPEN traversal; arrival costs belong to its destination."""
    if edge not in network.outgoing(edge.origin):
        raise ValidationError("The leg must be an OPEN service segment in this network.")
    if current_service is not None:
        if edge.origin not in network.service(current_service).stops:
            raise ValidationError("A service change must occur at a common stop.")
    changing = current_service is not None and current_service != edge.code
    destination = network.require_stop(edge.destination)
    costs = Costs(
        boarding_fare=5.0 if current_service is None else 0.0,
        segment_fare=edge.incremental_fare,
        transfer_fare=5.0 if changing else 0.0,
        driving_minutes=edge.driving_minutes,
        arrival_minutes=destination.access_minutes,
        transfer_minutes=network.nodes[edge.origin].transfer_minutes if changing else 0.0,
        risk=edge.risk + 0.1 * destination.safety_risk,
        transfers=int(changing),
    )
    return Costs().plus(costs)


def evaluate_itinerary(network: Network, origin: str, segments: tuple[Segment, ...]) -> Itinerary:
    """Recalculate a simple itinerary and reject disconnected or invalid legs."""
    network.require_stop(origin)
    place, service = origin, None
    visited = {origin}
    legs, total = [], Costs()
    for edge in segments:
        if edge.origin != place or edge.destination in visited:
            raise ValidationError("An itinerary must be connected and cannot revisit stops.")
        costs = leg_costs(network, edge, service)
        legs.append(Leg(edge, costs))
        total = total.plus(costs)
        if network.max_transfers is not None and total.transfers > network.max_transfers:
            raise ValidationError(f"This scenario permits at most {network.max_transfers} transfer(s).")
        visited.add(edge.destination)
        place, service = edge.destination, edge.code
    return Itinerary(origin, tuple(legs), total)


def rank_key(itinerary: Itinerary, preference: str) -> tuple:
    if preference not in OBJECTIVES:
        raise ValidationError(f"Unknown preference: {preference!r}. Choose {', '.join(OBJECTIVES)}.")
    costs = itinerary.costs
    values = {"score": costs.score, "minutes": costs.minutes, "fare": costs.fare,
              "risk": costs.risk, "transfers": costs.transfers,
              "weighted": risk_weighted_score(costs)}
    primary = {"Balanced": "score", "Risk-weighted": "weighted", "Fare": "fare", "Time": "minutes", "Safety": "risk"}[preference]
    fields = tuple(dict.fromkeys((primary, "score", "minutes", "fare", "risk", "transfers")))
    return tuple(values[field] for field in fields) + (itinerary.identity,)


def risk_weighted_score(costs: Costs) -> float:
    return sum((0.30 * (costs.minutes / 120), 0.20 * (costs.fare / 40),
                0.40 * (costs.risk / 100), 0.10 * (costs.transfers / 4)))


def search_routes(network: Network, origin: str = "Alpha", destination: str = "Romeo",
                  preference: str = "Balanced", budget: float = 40) -> SearchResult:
    """Settle the top three simple paths, without merging competing arrival labels.

    All incremental ranking metrics are nonnegative. Every label keeps its own
    history, including visited stops and service. No heuristic cutoff or
    (stop, service) dominance rule removes a potentially feasible alternative.
    """
    network.require_stop(origin)
    network.require_stop(destination)
    entered = _number(budget, "Budget")
    effective = min(entered, 40.0)
    if network.max_transfers is not None and (
            type(network.max_transfers) is not int or network.max_transfers < 0):
        raise ValidationError("Maximum transfers must be a nonnegative integer or None.")
    initial = Itinerary(origin)
    initial_key = rank_key(initial, preference)
    serial = count()
    queue = [(initial_key, next(serial), initial, frozenset((origin,)))]
    winners = []
    expanded = 0
    while queue and len(winners) < 3:
        _, _, path, visited = heappop(queue)
        place = path.legs[-1].segment.destination if path.legs else origin
        if place == destination:
            winners.append(path)
            continue
        expanded += 1
        service = path.legs[-1].segment.code if path.legs else None
        for edge in network.outgoing(place):
            if edge.destination in visited:
                continue
            increment = leg_costs(network, edge, service)
            total = path.costs.plus(increment)
            if total.fare > effective:
                continue
            if network.max_transfers is not None and total.transfers > network.max_transfers:
                continue
            candidate = Itinerary(origin, path.legs + (Leg(edge, increment),), total)
            heappush(queue, (rank_key(candidate, preference), next(serial), candidate,
                             visited | {edge.destination}))
    return SearchResult(origin, destination, preference, entered, effective, tuple(winners),
                        expanded, network.max_transfers)
