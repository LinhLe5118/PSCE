"""Development-only audit of EH-aligned stochastic-tabular OEH proposals."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'code'))
from oeh_certificate_adapter import trajectory_statistics
from oeh_eh_aligned import estimate_eh_aligned, oracle_eh_aligned
from scripts.analysis.run_oeh_development_smoke import development_inputs
from evaluation.io import reserve_output_dir, verified_dataset_path, sha256_file, write_json, write_csv_rows


def _ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i+1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i+j-1)/2
        i = j
    return ranks


def _spearman(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 2:
        return None
    x, y = _ranks(first[keep]), _ranks(second[keep])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary(rows):
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    def cluster_ci(key):
        groups = {}
        for row in rows:
            groups.setdefault(row['mdp_id'], []).append(float(row[key]))
        values = list(groups.values())
        if not values:
            return None
        rng = np.random.default_rng(20260919)
        draws = np.empty(10000)
        for i in range(len(draws)):
            picked = rng.integers(0, len(values), len(values))
            draws[i] = np.mean([item for j in picked for item in values[j]])
        return [float(x) for x in np.quantile(draws, [.025, .975])]
    return dict(
        cases=len(rows),
        point_evidence_nonflat=sum(r['point_evidence_range'] > 1e-9 for r in rows),
        eb_evidence_nonflat=sum(r['eb_evidence_range'] > 1e-9 for r in rows),
        eb_evidence_range_above_0_01=sum(r['eb_evidence_range'] > .01 for r in rows),
        point_score_nonflat=sum(r['point_raw_range'] > 1e-9 for r in rows),
        eb_score_nonflat=sum(r['eb_raw_range'] > 1e-9 for r in rows),
        planning_regret_nonflat=sum(r['planning_regret_range'] > 1e-9 for r in rows),
        multiple_policies=sum(r['unique_policies'] > 1 for r in rows),
        candidate_success_available=sum(r['candidate_first_success_depth'] is not None for r in rows),
        point_selector_available=sum(r['point_selected_depth'] is not None for r in rows),
        eb_selector_available=sum(r['eb_selected_depth'] is not None for r in rows),
        any_identified_depth=sum(r['identified_depths'] > 0 for r in rows),
        point_conditional_success=sum(r['point_selected_regret'] is not None and r['point_selected_regret'] <= .05+1e-9 for r in rows),
        eb_conditional_success=sum(r['eb_selected_regret'] is not None and r['eb_selected_regret'] <= .05+1e-9 for r in rows),
        point_fallback_success=sum(r['point_fallback_regret'] <= .05+1e-9 for r in rows),
        eb_fallback_success=sum(r['eb_fallback_regret'] <= .05+1e-9 for r in rows),
        fixed100_success=sum(r['fixed100_regret'] <= .05+1e-9 for r in rows),
        mean_point_conditional_regret=mean('point_selected_regret'),
        mean_eb_conditional_regret=mean('eb_selected_regret'),
        mean_point_fallback_regret=mean('point_fallback_regret'),
        mean_eb_fallback_regret=mean('eb_fallback_regret'),
        mean_fixed0_regret=mean('fixed0_regret'),
        mean_fixed100_regret=mean('fixed100_regret'),
        mean_candidate_best_regret=mean('candidate_best_regret'),
        mean_point_gain_vs_fixed100=mean('point_gain_vs_fixed100'),
        mean_eb_gain_vs_fixed100=mean('eb_gain_vs_fixed100'),
        point_gain_ci95_cluster_mdp=cluster_ci('point_gain_vs_fixed100'),
        eb_gain_ci95_cluster_mdp=cluster_ci('eb_gain_vs_fixed100'),
        point_better_worse_tied_vs_fixed100=[
            sum(r['point_gain_vs_fixed100'] > 1e-12 for r in rows),
            sum(r['point_gain_vs_fixed100'] < -1e-12 for r in rows),
            sum(abs(r['point_gain_vs_fixed100']) <= 1e-12 for r in rows)],
        eb_better_worse_tied_vs_fixed100=[
            sum(r['eb_gain_vs_fixed100'] > 1e-12 for r in rows),
            sum(r['eb_gain_vs_fixed100'] < -1e-12 for r in rows),
            sum(abs(r['eb_gain_vs_fixed100']) <= 1e-12 for r in rows)],
        point_exact_first_success=sum(r['point_depth_error'] == 0 for r in rows if r['point_depth_error'] is not None),
        eb_exact_first_success=sum(r['eb_depth_error'] == 0 for r in rows if r['eb_depth_error'] is not None),
        point_within2_first_success=sum(r['point_depth_error'] is not None and r['point_depth_error'] <= 2 for r in rows),
        eb_within2_first_success=sum(r['eb_depth_error'] is not None and r['eb_depth_error'] <= 2 for r in rows),
        mean_point_depth_error=mean('point_depth_error'),
        mean_eb_depth_error=mean('eb_depth_error'),
        mean_point_oracle_spearman=mean('point_oracle_spearman'),
        mean_eb_oracle_spearman=mean('eb_oracle_spearman'))


def _md_report(report):
    lines = [
        "# EH-aligned OEH development audit (2026-09-19)", "",
        "Status: **%s**. Scope: development pilots only; all oracle quantities are post-estimation audit labels." % report['status'], "",
        "## Formula tested", "",
        r"For QVI index $k$, the author-aligned point term is $k+\log_A(1/\hat\Delta_k^2)$. "
        r"The stochastic extension replaces $1/\hat\Delta_k^2$ by the total transition count required by an empirical-Bernstein radius, using the Bellman-target variance and behavior frequency of the greedy/runner-up actions. "
        r"A depth is admissible only when every supported empirical greedy policy reaches at least 95% of the empirical optimal return, measured from the empirical minimum return, matching the approximate-solvability convention in the original paper.", "",
        "The occupancy threshold is 1/(10|S|), so these are expected-return, behavior-supported diagnostics rather than the original paper's uniform all-state EH. The Bernstein term is a plug-in extrapolation, not a finite-sample certificate.", "",
        "## Results", ""]
    for name in ('all', 'stochastic'):
        s = report[name]
        lines += [f"### {name.capitalize()} ({s['cases']} pilots)", "",
            f"- Non-flat evidence: point {s['point_evidence_nonflat']}/{s['cases']}; Bernstein {s['eb_evidence_nonflat']}/{s['cases']} (range > 0.01: {s['eb_evidence_range_above_0_01']}). Non-flat finite total score: point {s['point_score_nonflat']}; Bernstein {s['eb_score_nonflat']}.",
            f"- At least one identified depth: {s['any_identified_depth']}/{s['cases']}; multiple candidate policies: {s['multiple_policies']}/{s['cases']}; a candidate reaches benchmark NR >= 0.95 in {s['candidate_success_available']}/{s['cases']}.",
            f"- Success with fixed-depth-100 fallback on abstention: point {s['point_fallback_success']}/{s['cases']}; Bernstein {s['eb_fallback_success']}/{s['cases']}; fixed depth 100 {s['fixed100_success']}/{s['cases']}.",
            f"- Mean benchmark normalized regret (1-NR), including fallback: point {s['mean_point_fallback_regret']}; Bernstein {s['mean_eb_fallback_regret']}; fixed depth 100 {s['mean_fixed100_regret']}; candidate oracle {s['mean_candidate_best_regret']}.",
            f"- Mean gain over fixed depth 100: point {s['mean_point_gain_vs_fixed100']} (cluster CI {s['point_gain_ci95_cluster_mdp']}); Bernstein {s['mean_eb_gain_vs_fixed100']} (cluster CI {s['eb_gain_ci95_cluster_mdp']}).",
            f"- Mean depth error against the first successful candidate: point {s['mean_point_depth_error']}; Bernstein {s['mean_eb_depth_error']}.",
            f"- Mean within-curve Spearman correlation with the exact-model counterpart: point {s['mean_point_oracle_spearman']}; Bernstein {s['mean_eb_oracle_spearman']}.", ""]
    lines += ["## Interpretation", "",
        "The thresholded all-one multiplier floor is absent: an identified depth has a gap/sample requirement, while an unresolved empirical tie causes abstention. This removes the old mechanical saturation but does not make every evidence curve non-flat.", "",
        "The Bernstein selector has a small positive development-only mean gain with the registered depth-100 fallback, but exact-model rank alignment is weak and many pilots abstain. The result is evidence for continued validation, not a confirmed predictive or publication claim.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-mdps', type=int)
    args = parser.parse_args(argv)
    manifest, records, labels, index = development_inputs(args.dataset_dir, args.limit_mdps)
    bench = manifest['effective_protocol']['benchmark']
    if bench['gamma'] != .95 or bench['reward_range'] != [0., 1.]:
        raise ValueError('Protocol requires gamma .95 and rewards [0,1]')
    reserve_output_dir(args.output_dir)
    sources = ['code/oeh_eh_aligned.py', 'code/oeh_selective.py',
        'code/oeh_certificate_adapter.py', 'code/exact_solver.py',
        'tests/test_oeh_eh_aligned.py', 'scripts/analysis/run_oeh_eh_aligned.py',
        'scripts/analysis/run_oeh_development_smoke.py',
        'environments/benchmark_environments.py', 'evaluation/io.py']
    plan = dict(status='exploratory_development', dataset_dir=str(args.dataset_dir.resolve()),
        source_sha256={p:sha256_file(ROOT/p) for p in sources},
        manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'),
        index_sha256=sha256_file(index),
        design=dict(gamma=.95, reward_bound=1, epsilon=.05, delta=.05, k_max=100,
            qvi_indexing='Q^1=Q^behavior; depth column is qvi_index-1',
            state_scope='discounted occupancy >= 1/(10|S|), nonterminal, >=2 observed actions',
            point_requirement='max critical pair 1/gap^2',
            bernstein_requirement='max pair/action max(128*variance*ell/(d*gap^2), (1+56*ell/(3*gap))/d)',
            ell='log(4*S*A*101/delta)',
            admissibility='worst supported empirical greedy policy reaches 95% of empirical optimum measured from empirical minimum',
            selector='minimum k+log_A(requirement), earliest on exact ties'),
        limitations=['Development-only formula audit; no evaluation split accessed.',
            'Empirical-Bernstein quantity is a plug-in sample requirement, not a confidence certificate.',
            'State scope adapts the original uniform EH to the expected-return offline objective.',
            'Pilot cases share MDPs and are not independent statistical units.'],
        runtime=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}))
    write_json(args.output_dir/'plan.json', plan)
    started = time.perf_counter()
    summaries, curves, errors, cache = [], [], [], {}
    for i, rec in enumerate(records, 1):
        label = labels[rec['dataset_id']]
        identity = dict(dataset_id=rec['dataset_id'], mdp_id=label['instance_id'],
            policy_family=label['policy_family'], pilot_multiplier=rec['size_multiplier'],
            num_transitions=rec['size'])
        try:
            with np.load(verified_dataset_path(args.dataset_dir, rec), allow_pickle=False) as z:
                arrays = {k:z[k][:rec['size']] for k in z.files}
            estimate = estimate_eh_aligned(arrays, rec['initial_state_distribution'],
                num_states=rec['num_states'], num_actions=rec['num_actions'])
            counts3, _, _, _ = trajectory_statistics(arrays,
                num_states=rec['num_states'], num_actions=rec['num_actions'], reward_bound=1.)
            counts = counts3.sum(2)
            # Oracle access happens only after the complete data-only profile exists.
            from environments.benchmark_environments import make_benchmark_mdp
            from exact_solver import value_iteration, evaluate_policy
            instance = identity['mdp_id']
            if instance not in cache:
                spec = next(s for s in manifest['instances'] if s['instance_id'] == instance)
                mdp = make_benchmark_mdp(type('Spec', (), spec)())
                best = float(mdp.initial_state_distribution@value_iteration(mdp).values)
                if not np.isclose(best, spec['optimal_return'], atol=1e-8, rtol=1e-8):
                    raise ValueError('Oracle mismatch')
                cache[instance] = mdp, best, spec['family']
            mdp, best, family = cache[instance]
            oracle = oracle_eh_aligned(mdp, counts, estimate['behavior'])
            random_policy = np.full((mdp.num_states, mdp.num_actions), 1/mdp.num_actions)
            random_return = evaluate_policy(mdp, random_policy).expected_return
            benchmark_range = best-random_return
            if benchmark_range <= 1e-12:
                raise ValueError('Nonpositive benchmark normalization range')
            value_cache, regrets, bound_regrets = {}, [], []
            for policy in estimate['policies']:
                key = policy.tobytes()
                if key not in value_cache:
                    value_cache[key] = evaluate_policy(mdp, policy).expected_return
                value = value_cache[key]
                regrets.append(max(0., (best-value)/benchmark_range))
                bound_regrets.append(max(0., .05*(best-value)))
            regrets = np.asarray(regrets)
            bound_regrets = np.asarray(bound_regrets)
            successes = np.flatnonzero(regrets <= .05+1e-9)
            first_success = int(successes[0]) if len(successes) else None
            point_depth, eb_depth = estimate['point_selected_depth'], estimate['eb_selected_depth']
            erows, orows = estimate['records'], oracle['records']
            point_ev = np.array([np.nan if r['point_evidence'] is None else r['point_evidence'] for r in erows])
            eb_ev = np.array([np.nan if r['eb_evidence'] is None else r['eb_evidence'] for r in erows])
            point_raw = np.array([np.nan if r['point_raw'] is None else r['point_raw'] for r in erows])
            eb_raw = np.array([np.nan if r['eb_raw'] is None else r['eb_raw'] for r in erows])
            planning = np.array([r['planning_regret'] for r in erows])
            point_oracle = np.array([np.nan if r['point_evidence'] is None else r['point_evidence'] for r in orows])
            eb_oracle = np.array([np.nan if r['eb_evidence'] is None else r['eb_evidence'] for r in orows])
            def finite_range(values):
                values = values[np.isfinite(values)]
                return float(np.ptp(values)) if len(values) else 0.
            row = {**identity, 'structural_family':family,
                'unique_policies':len({p.tobytes() for p in estimate['policies']}),
                'point_evidence_range':finite_range(point_ev),
                'eb_evidence_range':finite_range(eb_ev),
                'point_raw_range':finite_range(point_raw),
                'eb_raw_range':finite_range(eb_raw),
                'planning_regret_range':float(np.ptp(planning)),
                'admissible_depths':sum(r['admissible'] for r in erows),
                'identified_depths':sum(r['identified'] for r in erows),
                'point_selected_depth':point_depth, 'eb_selected_depth':eb_depth,
                'oracle_point_selected_depth':oracle['point_selected_depth'],
                'oracle_eb_selected_depth':oracle['eb_selected_depth'],
                'candidate_first_success_depth':first_success,
                'candidate_best_depth':int(regrets.argmin()),
                'candidate_best_regret':float(regrets.min()),
                'candidate_best_public_bound_regret':float(bound_regrets.min()),
                'fixed0_regret':float(regrets[0]), 'fixed100_regret':float(regrets[-1]),
                'fixed0_public_bound_regret':float(bound_regrets[0]),
                'fixed100_public_bound_regret':float(bound_regrets[-1]),
                'point_selected_regret':None if point_depth is None else float(regrets[point_depth]),
                'eb_selected_regret':None if eb_depth is None else float(regrets[eb_depth]),
                'point_fallback_depth':100 if point_depth is None else point_depth,
                'eb_fallback_depth':100 if eb_depth is None else eb_depth,
                'point_fallback_regret':float(regrets[-1] if point_depth is None else regrets[point_depth]),
                'eb_fallback_regret':float(regrets[-1] if eb_depth is None else regrets[eb_depth]),
                'point_gain_vs_fixed100':float(0 if point_depth is None else regrets[-1]-regrets[point_depth]),
                'eb_gain_vs_fixed100':float(0 if eb_depth is None else regrets[-1]-regrets[eb_depth]),
                'point_depth_error':None if point_depth is None or first_success is None else abs(point_depth-first_success),
                'eb_depth_error':None if eb_depth is None or first_success is None else abs(eb_depth-first_success),
                'point_oracle_spearman':_spearman(point_ev, point_oracle),
                'eb_oracle_spearman':_spearman(eb_ev, eb_oracle)}
            summaries.append(row)
            for depth, (est, truth) in enumerate(zip(erows, orows)):
                curves.append({**identity, 'structural_family':family,
                    **{f'est_{k}':v for k,v in est.items()},
                    'oracle_point_evidence':truth['point_evidence'],
                    'oracle_eb_evidence':truth['eb_evidence'],
                    'oracle_planning_regret':truth['planning_regret'],
                    'true_candidate_regret':float(regrets[depth]),
                    'true_candidate_public_bound_regret':float(bound_regrets[depth])})
            print(f"[{i}/{len(records)}] {instance} x{rec['size_multiplier']}: "
                  f"evidence ranges point={row['point_evidence_range']:.3g}, EB={row['eb_evidence_range']:.3g}; "
                  f"selected={point_depth}/{eb_depth}, first-success={first_success}", flush=True)
        except Exception as exc:
            errors.append({**identity, 'error':repr(exc)})
            print(f"[{i}/{len(records)}] ERROR {exc!r}", flush=True)
        write_json(args.output_dir/'progress.json', dict(processed=i,total=len(records),errors=errors))
    if summaries:
        write_csv_rows(args.output_dir/'summary.csv', summaries, list(summaries[0]))
    if curves:
        write_csv_rows(args.output_dir/'curves.csv', curves, list(curves[0]))
    all_summary = _summary(summaries)
    stochastic_summary = _summary([r for r in summaries if r['structural_family'] != 'lock_structured'])
    report = {**plan, 'status':'complete' if not errors else 'completed_with_errors',
        'all':all_summary, 'stochastic':stochastic_summary, 'errors':errors,
        'num_curve_rows':len(curves), 'wall_seconds':time.perf_counter()-started}
    write_json(args.output_dir/'report.json', report)
    (args.output_dir/'REPORT.md').write_text(_md_report(report), encoding='utf-8')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
