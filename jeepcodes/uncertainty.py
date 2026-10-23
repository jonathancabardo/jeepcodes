"""Approved S8 simulation; exact mean-score search using additive sampled costs."""
from dataclasses import dataclass, replace
from math import fsum
from random import Random
from statistics import pstdev
from types import MappingProxyType

from .data import ValidationError

SEED = 42
SIMULATIONS = 100


def draws(network):
    if network.scenario_id != "S1" or any(e.driving_minutes_override is not None for e in network.segments):
        raise ValidationError("S8 simulation requires the original S1 baseline.")
    rng = Random(SEED)
    ordered = sorted(network.segments, key=lambda e: e.source_id)
    samples = {e.source_id: [] for e in ordered}
    for _ in range(SIMULATIONS):
        for edge in ordered:
            samples[edge.source_id].append(rng.uniform(0.7, 1.3))
    return {key: tuple(values) for key, values in samples.items()}


def mean_network(network):
    samples = draws(network)
    segments = tuple(replace(e, disruption_multiplier=fsum(samples[e.source_id])/SIMULATIONS)
                     for e in network.segments)
    adjacency = {place: [] for place in network.nodes}
    for edge in segments:
        edge.driving_minutes  # Reject nonfinite simulated totals atomically.
        adjacency[edge.origin].append(edge)
        adjacency[edge.destination].append(replace(edge, origin=edge.destination,destination=edge.origin))
    return replace(network, segments=segments, scenario_id="S8",
                   adjacency=MappingProxyType({place:tuple(edges) for place,edges in adjacency.items()}))


@dataclass(frozen=True)
class RouteVariation:
    identity: tuple
    mean_minutes: float
    minimum_minutes: float
    maximum_minutes: float
    std_minutes: float
    mean_balanced_score: float
    wins_among_returned_routes: int


@dataclass(frozen=True)
class UncertaintyReport:
    seed: int
    simulations: int
    routes: tuple[RouteVariation, ...]
    note: str = "Ranking is exact for mean Balanced score over all feasible simple itineraries, because that score is linear in driving time. Win counts compare only the returned routes in each simulation; they are not global ranking-stability probabilities."


def summarize(baseline, result):
    samples = draws(baseline)
    source = {e.source_id:e for e in baseline.segments}
    times, scores = [], []
    for path in result.itineraries:
        fixed = path.costs.arrival_minutes + path.costs.transfer_minutes
        path_times = tuple(fixed + fsum(source[l.segment.source_id].driving_minutes * samples[l.segment.source_id][i]
                                       for l in path.legs) for i in range(SIMULATIONS))
        path_scores = tuple(0.4*(t/120) + 0.3*(path.costs.fare/40) +
                            0.2*(path.costs.risk/100) + 0.1*(path.costs.transfers/4) for t in path_times)
        times.append(path_times)
        scores.append(path_scores)
    wins = [0]*len(times)
    for i in range(SIMULATIONS):
        if scores:
            # Match the documented tie order, including sampled time and leg identity.
            winner = min(range(len(scores)), key=lambda j:(scores[j][i],times[j][i],result.itineraries[j].costs.fare,
                         result.itineraries[j].costs.risk,result.itineraries[j].costs.transfers,result.itineraries[j].identity))
            wins[winner] += 1
    return UncertaintyReport(SEED,SIMULATIONS,tuple(
        RouteVariation(path.identity,fsum(times[j])/SIMULATIONS,min(times[j]),max(times[j]),pstdev(times[j]),
                       fsum(scores[j])/SIMULATIONS,wins[j]) for j,path in enumerate(result.itineraries)))
