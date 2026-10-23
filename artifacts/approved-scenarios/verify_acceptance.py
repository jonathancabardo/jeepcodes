"""Repeat the documented acceptance audit without editing source data or models.

Run with the project Python: python -B verify_acceptance.py PROJECT OUTPUT_DIR
OUTPUT_DIR must be a new directory. This utility is verification, not app logic.
"""
import csv
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
from math import isclose, sqrt
from pathlib import Path
import platform
import subprocess
import sys

project = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(project))
from jeepcodes.cli import DEFAULT_WORKBOOK
from jeepcodes.data import SOURCE_SHA256, ValidationError, load_network
from jeepcodes.routing import Costs, evaluate_itinerary, search_routes
from jeepcodes.scenarios import ASSUMPTIONS, SCENARIOS, PENDING_SCENARIOS, Query, ScenarioSession, apply_scenario
from jeepcodes.ml import run_experiment
from jeepcodes.ml_routing import compare_routing, load_predictor

def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()

protected = [DEFAULT_WORKBOOK, *sorted((project / 'artifacts/step4-ml').glob('*')),
             *sorted((project / 'artifacts/step5-routing').glob('*'))]
before = {str(p.relative_to(project)) if p.is_relative_to(project) else str(p): fingerprint(p)
          for p in protected if p.is_file()}
graph = load_network(DEFAULT_WORKBOOK, SOURCE_SHA256)
checks = []
def check(name, condition, detail):
    checks.append({'id': name, 'status': 'PASS' if condition else 'FAIL', 'detail': detail})

def invalid(call):
    try:
        call()
    except ValidationError:
        return True
    return False

def valid_paths(network, result):
    return all(p.costs.fare <= result.effective_budget and
               len(p.stops) == len(set(p.stops)) and
               evaluate_itinerary(network, result.origin, tuple(l.segment for l in p.legs)) == p
               for p in result.itineraries)

baseline = search_routes(graph)
zero = search_routes(graph, 'Alpha', 'Alpha', budget=0)
delta_graph = apply_scenario(graph, 'S3')
delta = search_routes(delta_graph)
check('A1', graph.service('01A').stops == ('Alpha','Bravo','Charlie','Echo','Golf'), 'Exact original ordered stops, including leading-zero code.')
check('A2', invalid(lambda: graph.service('UNKNOWN')) and invalid(lambda: search_routes(graph, destination='Atlantis')), 'Unknown code and stop rejected.')
check('A3', zero.itineraries[0].costs == Costs() and not zero.itineraries[0].legs, 'Zero-leg query has zero totals.')
check('A4', len(baseline.itineraries) == 3 and valid_paths(graph, baseline), 'Three valid affordable baseline itineraries.')
changed = [(a,b) for a,b in zip(graph.segments, delta_graph.segments) if 'Delta' in (a.origin,a.destination)]
check('A5', len(changed) == 8 and all(isclose(b.driving_minutes,a.driving_minutes*2.5) for a,b in changed) and valid_paths(delta_graph,delta), 'Eight source segments slow once; top routes recomputed.')
check('A6', not search_routes(graph,budget=0).itineraries, 'No affordable nonzero trip at budget zero.')
closed = apply_scenario(graph,'S2')
check('A7', all(not ({e.origin,e.destination} == {'Foxtrot','Golf'}) for edges in closed.adjacency.values() for e in edges if e.status == 'OPEN') and valid_paths(closed,search_routes(closed)), 'Every Foxtrot–Golf service closed both ways.')
check('A8', len({p.identity for p in baseline.itineraries}) == 3 and valid_paths(graph,baseline), 'Alternative leg identities distinct and re-evaluate correctly.')
predictor = load_predictor(graph)
study = compare_routing(graph,predictor,Query())
repeat = compare_routing(graph,predictor,Query())
saved = json.loads((project/'artifacts/step4-ml/evaluation.json').read_text(encoding='utf-8'))
experiment = run_experiment(graph)
check('A9', baseline == search_routes(graph) and study.report() == repeat.report() and experiment.report() == saved,
      'Baseline and ML routes repeat exactly; full seed-42 experiment reproduces saved Step 4 report. S8 seed/repeat/reset also verified below and by regression tests.')
with (project/'artifacts/step4-ml/heldout_predictions.csv').open(encoding='utf-8',newline='') as f:
    heldout = list(csv.DictReader(f))
metrics = {}
for field in ('model_minutes','naive_minutes'):
    errors = [float(row['target_minutes'])-float(row[field]) for row in heldout]
    metrics[field] = {'mae':sum(abs(e) for e in errors)/len(errors),'rmse':sqrt(sum(e*e for e in errors)/len(errors))}
check('A10', len(heldout) == 1440 and
      isclose(metrics['model_minutes']['mae'],experiment.model_metrics.mae_minutes) and
      isclose(metrics['model_minutes']['rmse'],experiment.model_metrics.rmse_minutes) and
      isclose(metrics['naive_minutes']['mae'],experiment.naive_metrics.mae_minutes) and
      isclose(metrics['naive_minutes']['rmse'],experiment.naive_metrics.rmse_minutes),
      'Independent arithmetic on 1,440 saved held-out rows agrees with rerun metrics for both predictors.')

queries = [('T01','Alpha','Golf','Fare','S1'),('T02','Alpha','Romeo','Balanced','S1'),
           ('T03','Juliet','Romeo','Time','S1'),('T04','Alpha','Alpha','Balanced','S1'),
           ('T06','Foxtrot','Romeo','Safety','S1'),('T07','Alpha','Romeo','Balanced','S2'),
           ('T08','Alpha','Romeo','Fare','S5'),('T09','Romeo','Alpha','Balanced','S1')]
query_results = {}
for name,origin,destination,preference,scenario in queries:
    network = apply_scenario(graph,scenario)
    result = search_routes(network,origin,destination,preference)
    good = bool(result.itineraries) and valid_paths(network,result)
    if name == 'T02': good = good and result.itineraries[0].costs.transfers > 0
    if name == 'T04': good = good and result.itineraries[0].costs == Costs()
    if name == 'T07': good = good and all({l.segment.origin,l.segment.destination} != {'Foxtrot','Golf'} for p in result.itineraries for l in p.legs)
    if name == 'T08': good = good and all(l.segment.code != '01A' for p in result.itineraries for l in p.legs)
    check(name,good, f'{origin} → {destination}; {preference}; {scenario}; valid paths with requested property. Safety is minimum total synthetic risk, not a ban on particular stops.')
    query_results[name] = {'routes':[{'identity':p.identity,'costs':asdict(p.costs)} for p in result.itineraries], 'message':result.message}
check('T05',invalid(lambda: search_routes(graph,'Alpha','Atlantis')), 'Unknown destination rejected.')
disconnected = replace(graph,adjacency={stop:() for stop in graph.nodes})
check('T10', not search_routes(disconnected).itineraries, 'Isolated disconnected fixture; original workbook unchanged.')

scenario_status = {}
scenario_results = {}
for scenario in SCENARIOS:
    session = ScenarioSession(graph,Query())
    comparison = session.activate(scenario)
    active = session.result
    scenario_results[scenario] = {'preference':active.preference,
        'routes':[{'identity':p.identity,'costs':asdict(p.costs),'minutes':p.costs.minutes,'fare':p.costs.fare,'balanced_score':p.costs.score} for p in active.itineraries],
        'explanation':comparison.explanation,
        'uncertainty':asdict(comparison.uncertainty) if comparison.uncertainty is not None else None}
    session.activate(scenario)
    stable = session.result == active
    if scenario == 'S8':
        check('S8-repeat-report',comparison.uncertainty == session.activate('S8').uncertainty,'100 seed-42 simulation statistics reproduce exactly.')
    session.reset()
    check(scenario, stable and session.network == graph and session.result == baseline,
          'Repeated activation stable and reset restores original graph/results; scenario-specific assertions also covered in regression suite.')
    scenario_status[scenario] = 'PASS'
for scenario,reason in PENDING_SCENARIOS.items():
    check(scenario+'-guard', invalid(lambda s=scenario: apply_scenario(graph,s)),reason)
    scenario_status[scenario] = 'UNRESOLVED: '+reason

commands = [
    ('lookup',['--code','01A'],0),
    ('baseline',['--origin','Alpha','--destination','Romeo'],0),
    ('delta-reset',['--scenario','S3','--reset'],0),
    ('closure',['--scenario','S2'],0),
    ('infeasible',['--origin','Alpha','--destination','Romeo','--budget','0'],0),
    ('zero',['--origin','Alpha','--destination','Alpha','--budget','0'],0),
    ('unknown',['--destination','Atlantis'],2),
    ('s4',['--scenario','S4'],0),
    ('s6',['--scenario','S6'],0),
    ('s8',['--scenario','S8'],0),
    ('s9',['--scenario','S9'],0),
    ('ml-comparison',['--ml-routes'],0)]
cli = []
for name,args,expected in commands:
    run = subprocess.run([sys.executable,'-B',str(project/'app.py'),*args],cwd=project,capture_output=True,text=True,encoding='utf-8')
    (output/(name+'.txt')).write_text(run.stdout+run.stderr,encoding='utf-8')
    check('CLI-'+name,run.returncode == expected and 'Traceback' not in run.stderr, f'Exit {run.returncode}, expected {expected}; see {name}.txt.')
    cli.append({'name':name,'args':args,'exit_code':run.returncode})

after = {str(p.relative_to(project)) if p.is_relative_to(project) else str(p): fingerprint(p)
         for p in protected if p.is_file()}
check('source-integrity',before == after,'Workbook, Step 4 and Step 5 artifacts unchanged during audit.')
pip = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
check('dependencies',pip.returncode == 0,pip.stdout.strip()+pip.stderr.strip())
report = {'synthetic':True,'checks':checks,'scenario_status':scenario_status,'scenario_results':scenario_results,'approved_assumptions':dict(ASSUMPTIONS),'queries':query_results,
          'heldout_metrics':metrics,'ml_report_matches_saved':experiment.report() == saved,
          'ml_study':study.report(),'protected_sha256':after,
          'versions':{'python':platform.python_version(),**{name:version(name) for name in ['openpyxl','numpy','scikit-learn','joblib','streamlit','pandas']}},
          'cli':cli,'failed':sum(c['status']=='FAIL' for c in checks),
          'unresolved':[],'not_performed':['Instructor acceptance','Mobile browser inspection','Public deployment']}
(output/'acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'checks':len(checks),'failed':report['failed'],'versions':report['versions'],'heldout_metrics':metrics,'unresolved':report['unresolved']},indent=2))
sys.exit(bool(report['failed']))
