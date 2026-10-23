"""Streamlit presentation for the existing deterministic and ML route services."""

from pathlib import Path

import streamlit as st

from .cli import DEFAULT_WORKBOOK
from .data import SOURCE_SHA256, ValidationError, load_network
from .routing import OBJECTIVES, risk_weighted_score
from .route_graph import route_graph_html, stop_colors
from .scenarios import ASSUMPTIONS, PENDING_SCENARIOS, SCENARIOS, Query, ScenarioSession


MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "step4-ml" / "model.joblib"
SCENARIO_LABELS = {
    "S1": "Normal conditions", "S2": "Foxtrot–Golf road closure", "S3": "Delta congestion",
    "S4": "Hotel safety incident", "S5": "01A service outage",
    "S6": "42D fare surge", "S7": "Accessibility · at most one transfer",
    "S8": "Forecast uncertainty · 100 simulations", "S9": "Mixed disruption",
}
PERIOD_LABELS = {"off_peak": "Off-peak", "morning_peak": "Morning peak",
                 "midday": "Midday", "evening_peak": "Evening peak"}


def load_source():
    return load_network(DEFAULT_WORKBOOK, SOURCE_SHA256)


def ml_comparison(network, query, period):
    from .ml_routing import compare_routing, load_predictor
    return compare_routing(network, load_predictor(network, MODEL_PATH), query, period)


def metric_row(label, path):
    if path is None:
        return {"Condition": label, "Time (min)": "Unavailable", "Fare": "—",
                "Risk points": "—", "Transfers": "—", "Balanced score": "—"}
    c = path.costs
    return {"Condition": label, "Time (min)": f"{c.minutes:.2f}", "Fare": f"₱{c.fare:g}",
            "Risk points": f"{c.risk:.2f}", "Transfers": str(c.transfers),
            "Balanced score": f"{c.score:.6f}"}


def leg_rows(path):
    previous = None
    rows = []
    for leg in path.legs:
        e, c = leg.segment, leg.costs
        action = "Board" if previous is None else "Stay aboard" if previous == e.code else "Transfer"
        rows.append({"From": e.origin, "To": e.destination, "Jeepney": e.code,
                     "Action": action, "Fare (₱)": c.fare, "Minutes": round(c.minutes, 2)})
        previous = e.code
    return rows


def render_path(path, title, expanded=False, preference="Balanced", colors=None, network=None):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(" → ".join(path.stops))
        c = path.costs
        if preference == "Risk-weighted":
            st.caption(f"S4 risk-weighted score: {risk_weighted_score(c):.6f}. Lower is better. Original Balanced score shown below.")
        columns = st.columns(5)
        for col, label, value in zip(columns,
                                    ("Fare", "Time (min)", "Transfers", "Risk points", "Balanced score"),
                                    (f"₱{c.fare:g}", f"{c.minutes:.2f}", str(c.transfers),
                                     f"{c.risk:.1f}", f"{c.score:.4f}")):
            col.metric(label, value)
        if not path.legs:
            st.info("You are already at your destination. No boarding, travel, or fare is needed.")
            with st.expander(f"{title} · leg-by-leg details", expanded=expanded):
                st.html(route_graph_html(path, colors if colors is not None else stop_colors(path.stops), network))
            return
        boarded = []
        previous = None
        for leg in path.legs:
            if leg.segment.code != previous:
                verb = "Board" if previous is None else "Change to"
                boarded.append(f"{verb} {leg.segment.code} at {leg.segment.origin}")
                previous = leg.segment.code
        st.write(". ".join(boarded) + ".")
        with st.expander(f"{title} · leg-by-leg details", expanded=expanded):
            st.dataframe(leg_rows(path), hide_index=True, width="stretch")
            st.caption(f"Fare = ₱{c.boarding_fare:g} first boarding + ₱{c.segment_fare:g} segment fares "
                       f"+ ₱{c.transfer_fare:g} transfers. Time = {c.driving_minutes:.2f} driving "
                       f"+ {c.arrival_minutes:g} access + {c.transfer_minutes:g} transfer minutes.")
            st.html(route_graph_html(path, colors if colors is not None else stop_colors(path.stops), network))


def render_routes(result, colors=None, network=None):
    if not result.itineraries:
        st.warning(result.message)
        st.write("Try a larger budget up to ₱40, a different trip, or reset to normal conditions.")
        return
    phrases = {"Balanced": "the lowest combined score", "Risk-weighted": "the lowest approved risk-weighted score (30/20/40/10)", "Fare": "the lowest fare",
               "Time": "the shortest estimated time", "Safety": "the lowest synthetic risk"}
    st.caption(f"Ranking: {phrases[result.preference]}. Ties use the documented balanced-score and cost rules.")
    if colors is None:
        colors = stop_colors(network.nodes if network is not None else
                             (stop for path in result.itineraries for stop in path.stops))
    st.caption("Graphs include all network stops in the reference arrangement. Orange arrows highlight this route; "
               "gray connections show the rest of the network. Each stop has its own color. "
               "Connections show jeepney codes and total leg minutes. Spacing does not represent distance.")
    render_path(result.itineraries[0], "Recommended route", expanded=True, preference=result.preference, colors=colors, network=network)
    if result.origin == result.destination:
        return
    st.subheader("Other ways to get there")
    if len(result.itineraries) < 3:
        st.info(result.message)
    best = result.itineraries[0].costs
    for index, path in enumerate(result.itineraries[1:], 1):
        render_path(path, f"Alternative {index}", preference=result.preference, colors=colors, network=network)
        c = path.costs
        st.caption(f"Compared with the recommendation: fare {c.fare-best.fare:+g}, "
                   f"time {c.minutes-best.minutes:+.2f} min, risk {c.risk-best.risk:+.2f}, "
                   f"transfers {c.transfers-best.transfers:+d}, score {c.score-best.score:+.6f}.")
    st.caption("Alternatives can visit the same stops using different jeepney services.")


def render_comparison(comparison, before_label="Normal", after_label="Selected scenario"):
    old, new = comparison.baseline.itineraries, comparison.current.itineraries
    st.dataframe([metric_row(before_label, old[0] if old else None),
                  metric_row(after_label, new[0] if new else None)], hide_index=True, width="stretch")
    if old and new:
        same = old[0].identity == new[0].identity
        st.write("The recommended itinerary stays the same." if same else "The recommended itinerary changed.")
        st.write(f"The winning trip's time is {old[0].costs.minutes:.2f} → {new[0].costs.minutes:.2f} minutes; "
                 f"fare is ₱{old[0].costs.fare:g} → ₱{new[0].costs.fare:g}. "
                 "The active preference decides how time, fare, risk, and transfers are compared.")
    elif not new:
        st.warning(comparison.current.message)
    with st.expander("Why this choice? Numerical reasoning"):
        st.write(comparison.explanation)
    for index, row in enumerate(comparison.routes, 1):
        with st.expander(f"Itinerary {index} · costs under both conditions"):
            st.write(" → ".join((row.before or row.after).stops))
            st.dataframe([metric_row(before_label, row.before), metric_row(after_label, row.after)],
                         hide_index=True, width="stretch")
            if row.before is None:
                st.caption(f"{before_label}: {row.before_unavailable}")
            if row.after is None:
                st.caption(f"{after_label}: {row.after_unavailable}")


def render_activity(session):
    st.caption("History belongs to this submitted trip. A different trip starts a new history.")
    st.dataframe([{"Event": event.sequence, "Scenario": SCENARIO_LABELS[event.scenario],
                   "Trigger": event.trigger, "Routes returned": event.itinerary_count,
                   "Changed values": len(event.changes)} for event in session.events],
                 hide_index=True, width="stretch")
    labels = {"driving_minutes": "Driving minutes", "status": "Availability",
              "incremental_fare": "Segment fare", "safety_risk": "Node safety risk", "modeling_rule": "Approved modeling rule",
              "transfer_minutes": "Transfer delay (min)", "max_transfers": "Maximum transfers"}
    for event in reversed(session.events):
        with st.expander(f"Event {event.sequence} · {SCENARIO_LABELS[event.scenario]}"):
            st.write(event.trigger)
            if event.changes:
                st.dataframe([{
                    "Jeepney / stop": change.service or change.origin or "All routes",
                    "Segment": f"{change.origin} ↔ {change.destination}" if change.destination else "—",
                    "Change": labels[change.field], "Before": str(change.before), "After": str(change.after),
                    "Workbook reference": f"{change.sheet} row {change.row}" if change.row else "Scenario rule",
                } for change in event.changes], hide_index=True, width="stretch")
                st.caption("Each source segment change applies in both travel directions.")
            else:
                st.caption("No network values changed in this event.")
            st.write(event.outcome)


def request_reset():
    st.session_state["scenario"] = "S1"
    st.session_state["reset_requested"] = True


def main():
    st.set_page_config(page_title="JeepCodes 2.0", page_icon="🚐", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 4rem; max-width: 1200px;}
        h1 {letter-spacing: -.04em;} h3 {letter-spacing: -.02em;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #dce7e2;}
        [data-testid="stSidebarUserContent"] {padding-top: 1.5rem;}
        </style>""", unsafe_allow_html=True)
    st.caption("JEEPCODES 2.0 / CLASSROOM ROUTE LAB")
    st.title("Plan your trip. Understand the choice.")
    st.write("Compare affordable jeepney journeys and see what changes when traffic slows.")
    st.caption("Synthetic teaching routes and costs. These are not actual transit directions or fares.")
    try:
        network = load_source()
    except ValidationError as exc:
        st.error(f"The workbook could not be loaded. {exc}")
        st.button("Retry loading workbook", on_click=lambda: None)
        return

    with st.sidebar:
        st.header("Your trip")
        places = list(network.nodes)
        origin = st.selectbox("Origin", places, index=places.index("Alpha"), key="origin")
        destination = st.selectbox("Destination", places, index=places.index("Romeo"), key="destination")
        preference = st.selectbox("Preference", list(OBJECTIVES), key="preference")
        budget = st.number_input("Fare budget (₱)", value=40.0, step=1.0, key="budget")
        st.caption("Maximum allowed fare: ₱40. You can choose a smaller budget, including zero.")
        scenario = st.selectbox("Scenario", list(SCENARIO_LABELS),
                                format_func=lambda value: SCENARIO_LABELS[value], key="scenario")
        if scenario in ASSUMPTIONS:
            st.info(ASSUMPTIONS[scenario])
        include_ml = st.checkbox("Include ML comparison (normal / Delta)", key="include_ml")
        find = st.button("Find routes", type="primary", width="stretch", key="find")
        st.button("Reset to baseline", on_click=request_reset, width="stretch", key="reset")
        with st.expander("ML time-of-day setting"):
            period = st.selectbox("Time of day for ML comparison", list(PERIOD_LABELS), index=2,
                                  format_func=lambda value: PERIOD_LABELS[value], disabled=not include_ml,
                                  key="period")
        st.caption(f"{len(network.nodes)} stops · {len(network.services)} jeepney services")
        with st.expander("Approved scenario assumptions"):
            for code, message in ASSUMPTIONS.items():
                st.write(f"**{code}** — {message}")

    query = Query(origin, destination, preference, budget)
    signature = (query, scenario, include_ml, period if include_ml else None, network.source_sha256)
    reset = st.session_state.pop("reset_requested", False)
    initial = "submitted_signature" not in st.session_state and "attempted" not in st.session_state
    if find or reset or initial:
        st.session_state["attempted"] = True
        st.session_state["error"] = ""
        try:
            previous = st.session_state.get("controller")
            same_query = (previous is not None and st.session_state.get("submitted_query") == query
                          and previous.baseline.source_sha256 == network.source_sha256)
            controller = previous if same_query else ScenarioSession(network, query)
            with st.spinner("Comparing valid journeys within your budget…"):
                comparison = controller.reset() if reset else controller.activate(scenario, "Trip search submitted")
                study, ml_error = None, ""
                if include_ml:
                    if scenario not in ("S1", "S3"):
                        ml_error = "ML comparison supports normal and Delta only. Choose one of these scenarios to compare estimates."
                    else:
                        try:
                            study = ml_comparison(network, query, period)
                        except (ValidationError, ImportError) as exc:
                            ml_error = f"ML comparison unavailable: {exc}. Workbook-rule results are shown separately."
            st.session_state.update(controller=controller, comparison=comparison, study=study,
                                    ml_error=ml_error, submitted_query=query, submitted_signature=signature)
        except ValidationError as exc:
            st.session_state["error"] = str(exc)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    current = st.session_state.get("submitted_signature") == signature and not st.session_state.get("error")
    if not current:
        st.info("Trip inputs changed or need correction. Select Find routes to calculate this request.")
    else:
        st.subheader(f"{origin} → {destination}")
        st.caption(f"{SCENARIO_LABELS[scenario]} · {st.session_state['controller'].result.preference} ranking · "
                   f"effective budget ₱{st.session_state['controller'].result.effective_budget:g}")
    routes_tab, changes_tab, ml_tab, activity_tab, lookup_tab = st.tabs(
        ["Routes", "What changed", "ML comparison", "Activity", "Code lookup"])
    with routes_tab:
        if current:
            st.caption("S8 times and scores are means across 100 simulations." if scenario == "S8" else
                       "These routes use the workbook's driving-time rules with the selected scenario.")
            render_routes(st.session_state["controller"].result, stop_colors(network.nodes),
                          st.session_state["controller"].network)
            report = st.session_state["comparison"].uncertainty
            if report is not None:
                st.subheader("Travel-time variation across 100 simulations")
                st.caption(f"Seed {report.seed}; uniform factors 0.7–1.3; shared by both directions of each service segment.")
                st.dataframe([{"Route": "Recommended" if i == 0 else f"Alternative {i}",
                               "Mean minutes": row.mean_minutes, "Minimum minutes": row.minimum_minutes,
                               "Maximum minutes": row.maximum_minutes, "Standard deviation": row.std_minutes,
                               "Mean Balanced score": row.mean_balanced_score,
                               "Wins among these routes": row.wins_among_returned_routes}
                              for i,row in enumerate(report.routes)], hide_index=True, width="stretch")
                st.info(report.note)
        else:
            st.write("Your updated routes will appear here after a successful search.")
    with changes_tab:
        if current:
            render_comparison(st.session_state["comparison"], "Normal conditions", SCENARIO_LABELS[scenario])
        else:
            st.write("Submit your trip to see its normal and selected-scenario costs.")
    with ml_tab:
        if current and st.session_state.get("ml_error"):
            st.warning(st.session_state["ml_error"])
        study = st.session_state.get("study") if current else None
        if study is None:
            st.info("Enable Include ML comparison, choose normal or Delta conditions, and select Find routes.")
        else:
            st.subheader("Two ways to estimate the same trip")
            st.caption(f"Matched period: {PERIOD_LABELS[study.period]}. "
                       f"The deterministic comparator includes this simulation's period factor "
                       f"({study.report()['period_factor']:g}) in driving time. The Routes tab uses original workbook rules.")
            st.info("These are simulated forecasts, not measured time savings or real-traffic accuracy.")
            for title, comparison in (("Normal conditions", study.normal), ("Delta congestion", study.delta)):
                st.markdown(f"#### {title}")
                render_comparison(comparison, "Matched deterministic", "ML-assisted")
                with st.expander(f"{title} · ML routes and alternatives"):
                    render_routes(comparison.current, stop_colors(network.nodes), network)
            st.markdown("#### What Delta changes in the ML estimate")
            render_comparison(study.ml_reroute, "Normal ML", "Delta ML")
    with activity_tab:
        if current:
            render_activity(st.session_state["controller"])
        else:
            st.write("Submit the current trip to view its activity history.")
    with lookup_tab:
        st.subheader("Where does this jeepney go?")
        code = st.text_input("Jeepney code", value="01A", key="lookup_code")
        st.caption("Codes include their leading zero. Examples: 01A, 02B, 20A.")
        if st.button("Look up code", key="lookup"):
            try:
                service = network.service(code)
                st.success(f"{service.code}: " + " → ".join(service.stops))
            except ValidationError as exc:
                st.error(str(exc))
    with st.expander("How to read these results"):
        st.write("A transfer means changing jeepney codes at a shared stop. First boarding costs ₱5; "
                 "each transfer adds ₱5 and the stop's transfer delay. Each traveled segment adds its own fare.")
        st.write("Balanced combines time, fare, risk, and transfers. A lower score is better. "
                 "Risk points are fictional teaching values. A score may exceed 1.")
        st.caption("Balanced score = 0.40 × minutes/120 + 0.30 × fare/40 + "
                   "0.20 × risk/100 + 0.10 × transfers/4. Ranking uses unrounded values.")
        st.write("Scenario history is kept for this browser session. Reopening the app starts a fresh session. "
                 "The original workbook and trained model are read without changes.")


if __name__ == "__main__":
    main()
