# PSCE: estimating the sample complexity of an offline RL learner before training

This repository accompanies the article *Quantifying Data Efficiency in Offline RL: Estimating Sample Complexity Prior to Policy Training* by Thi Hai Linh Le, Trong Hung Do, Quoc Viet Nguyen, Quoc Viet Tran and Duy Hung Phan (FPT University, 2026). It contains the method, the data, the sealed evidence of the three final tests and the command that reproduces the results (release v1.1; v1.1 renames the method's parts in the documentation, and the code and data are those of v1.0).

**Names.** The article calls the method **PSCE** (pre-training sample complexity estimator). The code and the artifacts keep the names used during development:

| In the article | In this repository |
|---|---|
| PSCE (the final method) | OEH, version 2: `oeh_v2_spec.json`, `scripts/analysis/oeh_v2_forecast.py` (`v2_law`) |
| full head | OEH-F: `oeh_final`, `oeh_full_head` |
| robust head | `prior_oeh_matched` |
| Baseline | B2: `b2_matched`, `FEATURES_B2` |
| sampled-MDP training (Block 2) | OEH-B: `oeh_b_groups`, `direct_oehb` |
| effective horizon `h_H` (Block 3) | OEH-H: `horizon`, `oehh` |
| pilot learning curve (Block 4) | legacy diagnostics, D: feature groups G7–G8 |
| novelty gate and distance `d` (Block 6) | shift gate: `v2_law`; `lambda_max`, `eps_max`, `d0` in `oeh_v2_spec.json` |
| empirical sample complexity (the label) | data requirement, `event` |

**Reproduce the results** on a new machine (Python 3.12; about five minutes after installation):

```shell
pip install -r requirements.txt
python -m scripts.reproduce --out reproduce_out
```

**Large files.** Five development dataset indexes (about 355 MB) are not needed to reproduce the results and are attached to release v1.0 instead of the repository; their paths are given by their file names (`__` stands for `/`).

**Licence.** Code: MIT (`LICENSE`). Data and experiment records: CC BY 4.0 (`LICENSE-DATA.md`). Authors and contributions: `AUTHORS.md`.

---

# Difficulty-Aware Offline Reinforcement Learning

**Forecasting, from a small pilot log, how many logged transitions a given offline RL learner needs.**

An offline RL learner is trained on a fixed log of transitions. Whether the log is large enough is usually found out only after the data are collected and the learner is trained. This repository defines that question as a forecasting problem and contains the method of the project, **OEH (Offline Effective Horizon)**, the data, the sealed evidence of its three final tests, and a command that reproduces the final results on a new machine.

This page is the starting point; the full description of the method, the data, the final tests and the route to them is in [docs/final](docs/final/README.md). Everything listed under "Final method and results" belongs to the final pipeline; the rest of the repository is development history, kept unchanged because the evidence rules forbid deleting tried candidates, failed runs or negative results.

## The problem in one paragraph

A *problem* is a small tabular MDP (at most 256 state–action pairs) together with the behavior policy that logs it. A *learner* is one offline RL algorithm with fixed settings: tabular behavior cloning (BC), conservative fitted Q-iteration (CFQI) or implicit Q-learning (IQL). A *budget* `xb` means `b·|S||A|` logged transitions, on the grid `x1, x2, x5, x10, x20, x50`. The *data requirement* of a learner is the first budget from which at least 8 of 10 independent training runs reach a normalized return of at least 0.95 and keep doing so at every larger budget; if no budget qualifies, the requirement is `>x50`. From a *pilot* log of `x10` transitions and the public facts `|S|`, `|A|`, the start distribution and the discount factor, OEH forecasts a probability for each of the seven classes. Forecasts are scored by the integrated Brier score over the six budgets (IBS) and the seven-class log loss (LL), averaged within each MDP and then across MDPs.

## Final method: the six blocks of OEH

| Block (thesis name) | What it does | Code |
|---|---|---|
| 1. Generic statistics | 18 statistics of the pilot: size, coverage, returns, transition randomness, bootstrap spreads | `scripts/analysis/run_oeh_eighth_cohort_20261014.py` (`pilot_features`), field list `FEATURES_B2` in `scripts/analysis/compare_difficulty.py` |
| 2. Sampled-MDP training | Dirichlet posterior over the MDP and behavior; repeats the labeling experiment with the actual learner in 64 drawn MDPs | `code/oeh/posterior.py`, `code/oeh/predictive.py`, `code/oeh/tabular.py` |
| 3. Effective horizon | Offline counterpart of the effective horizon (`h_H`) in the posterior mean model | `code/oeh/horizon.py` |
| 4. Pilot learning curve | The learner cross-fitted on folds of the pilot prefixes (learning curve) and an action-identification risk | computed with Block 1 in `pilot_features` (fields `oeh3_*` and `lcoeh4_*`, listed in `OEH_MECHANISM` of `scripts/analysis/develop_oeh_difficulty.py`) |
| 5. Forecasting heads | Full head (108 features), finite-only head, robust head (generic statistics + pilot learning curve), each an average of three XGBoost models | `scripts/analysis/oeh_final_forecast.py` (`matched_forecast`) |
| 6. Novelty gate | Combines the heads by the distance of the pilot to the meta-training MDPs | `scripts/analysis/oeh_v2_forecast.py` (`v2_law`) |

**Single entry point:** [`scripts/analysis/oeh_pipeline.py`](scripts/analysis/oeh_pipeline.py). With `--dataset` it computes Blocks 1–4 from a dataset bundle (`pilot_inputs`); with `--query` it starts from an existing table of pilot inputs. It then fits the heads on the 720 meta-training MDPs and applies the frozen specification [`oeh_v2_spec.json`](artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json):

```shell
python -m scripts.analysis.oeh_pipeline --query <pilot inputs .csv/.csv.gz> --out <new folder>
```

## Final results

Three final tests on 660 new MDPs; every forecast was saved and hashed before any learner was run on the test data.

| Test | MDPs | Artifact folder | Registered comparison | Verdict |
|---|---|---|---|---|
| Test 1 | 168 from the project generator, 72 Colosseum, 108 SPIBB, 48 forest | [`artifacts/oeh_v2_sealed_test_2026-10-14`](artifacts/oeh_v2_sealed_test_2026-10-14/) | three steps: OEH vs robust head (external), vs full head, vs six baselines | GO |
| Test 2 | 36 Gymnasium FrozenLake, 36 gambler's problem | [`artifacts/oeh_fifth_panel_2026-10-15`](artifacts/oeh_fifth_panel_2026-10-15/) | OEH vs robust head | **NO-GO** |
| Test 3 | 96 MDP Playground, 96 labmaze | [`artifacts/oeh_panel6_2026-10-15`](artifacts/oeh_panel6_2026-10-15/) | OEH vs Baseline + prefix/FQE, then vs the Baseline, on budgets from the pilot size up | GO |

The numbers are produced by the reproduction command below (`4_tables.md` holds Tables 10–18 of the thesis). The NO-GO of Test 2 is a result, not an error: the advantage of OEH over the robust head depends on the source, and every claim is limited to small tabular MDPs.

## Reproduce the results

This recomputes the final results from the committed files: it refits the forecasting heads, regenerates the forecasts of the three tests from their sealed pilot inputs and checks them against the sealed forecasts, reruns each test's registered analysis on a copy of its files, and recomputes every printed number of Tables 11–18 of the thesis. It does not rerun the learners (that took several days and needs the raw transitions, which are not in the repository).

Requirements: Git with [Git LFS](https://git-lfs.com/), Python 3.12 and about 2 GB of disk (checkout, environment and output). Cloning downloads about 400 MB and took half an hour on a home connection; installing the packages takes a few minutes and the reproduction about five.

```shell
git lfs install
git clone https://github.com/LinhLe5118/Devoted.git
cd Devoted
python -m venv .venv
.venv\Scripts\activate                    # Windows; on Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements-test.txt -r requirements-difficulty.txt
python -m scripts.reproduce --out reproduce_out
```

Expected output, about five minutes with eight worker processes:

| Step | What it checks | Output file |
|---|---|---|
| 1. seals | protocols, pre-outcome seals and labels of the three tests are the committed bytes | `1_seals.json` |
| 2. forecast | heads refitted; regenerated forecasts of OEH, full head, robust head and the Baseline equal the sealed ones (largest difference about 1e-14) | `2_forecast_check.json`, `2_forecasts.csv.gz` |
| 3. registered | each test's registered analysis, rerun on a copy, gives the committed `result.json`, summaries and scores | `3_registered_check.json` |
| 4. tables | every printed number of Tables 11–18 of the thesis is recomputed and compared | `4_tables.md`, `4_tables.json` |

Read `reproduce_out/4_tables.md` first. Two printed numbers differ in the fourth decimal and are documented there: two lower bounds of post hoc bootstrap intervals in Table 17 whose original run is not on record. A step whose output exists is skipped, so an interrupted run can be resumed with the same command; `--steps seals,forecast` runs only some steps. The command never writes inside `artifacts/`.

Notes:
- The pins in `requirements-difficulty.txt` matter: another XGBoost or scikit-learn version gives different heads, and step 2 then stops.
- On Windows, long multi-process jobs can be moved to efficiency cores; if step 2 takes much longer than five minutes, run it with fewer workers (`--workers 4`) or keep the terminal in the foreground.

## Tests

```shell
python -m scripts.testing.run_full_tests      # full suite; any skipped test is a failure
python -m scripts.analysis.verify_oeh         # audit of the published evidence and unchanged definitions
```

`tests/test_sealed_integrity.py` fails if any file whose SHA-256 appears in a seal or manifest under `artifacts/` has changed.

## Repository map

| Path | Contents | Part of the final pipeline |
|---|---|---|
| `scripts/analysis/oeh_pipeline.py` | Single entry point of OEH | yes |
| `scripts/reproduce/` | Reproduction command of the final results | yes |
| `code/oeh/` | Blocks 2 and 3: posterior, sampled-MDP training, tabular learners with exact evaluation, effective horizon | yes |
| `scripts/analysis/run_oeh_v2_sealed_test_20261014.py`, `run_oeh_fifth_panel_20261015.py`, `run_oeh_panel6_20261015.py` | Runners of Tests 1–3: freeze the protocol, generate the data, seal the forecasts, open the labels, run the registered analysis | yes |
| `scripts/analysis/posthoc_20261028/`, `posthoc_20261101/` | The post hoc scripts that first produced the ablation, net-cost, learner-level and calibration numbers of the thesis; `scripts/reproduce` recomputes them | record |
| `artifacts/` | Protocols, seals, forecasts, labels and scores, one folder per experiment; see [artifacts/README.md](artifacts/README.md) for which folders the final pipeline reads | partly |
| `benchmarks/bridge/` | BRIDGE transfer runs (development) | no |
| `environments/`, `evaluation/`, `pilot/`, `config/`, `tmp/`, `protocol_config.json` | Restored runtime of the earlier routing study; byte-identical sources referenced by hashes in old protocols | no, keep unchanged |
| `docs/final/` | Final documentation: method, data, the three final tests, history | yes |
| `paper/` | Source of the IEEE paper and supplement, built and checked by CI; the final PDF is added here after every claim has been checked (H6) | writing |
| `docs/week8/` and the rest of `docs/` | Decision records, protocols and handovers ([week 8 index](docs/week8/README.md)); many are hash-recorded in protocols and manifests | records |
| `proposed_methodology.md` | First method design (superseded by docs/final) | records |
| `tests/` | Test suite | yes |

Earlier sealed tests, failed cohorts and abandoned candidates stay in `artifacts/` and `docs/` because the evidence rules forbid deleting them; they are development history, not part of the final method. The development of the final method and the mapping between code names and the names used in writing are in [docs/week8/OEH_V2_DEVELOPMENT_2026-10-14.md](docs/week8/OEH_V2_DEVELOPMENT_2026-10-14.md), section 6.

## Names in the code and in the writing

Names in writing changed twice. The current names are those of the thesis and of the journal article (where the method
is called PSCE); older records use the earlier names given in brackets.

| Writing (thesis, article; earlier name) | Code and artifacts |
|---|---|
| OEH in the thesis, PSCE in the journal article | `v2` in Test 1, `oeh` in Tests 2–3; specification `oeh_v2_spec.json` |
| Full head | `oeh_final` (Test 1), `oeh_full_head` (Tests 2–3); field set `oeh_final` |
| Robust head | `legacy` (Test 1), `robust` (Tests 2–3); field set `prior_oeh_matched` |
| Baseline (B2) | `b2_matched` |
| Sampled-MDP training, Block 2 (learner simulation) | OEH-B, `oeh_b_groups`, `direct_oehb` |
| Effective horizon `h_H`, Block 3 (planning risk) | OEH-H, `horizon`, `oehh` |
| Pilot learning curve, Block 4 (pilot diagnostics) | legacy diagnostics, D, feature groups G7–G8 |
| Novelty gate and distance `d`, Block 6 (shift gate, shift distance) | `v2_law`, gate settings `lambda_max`, `eps_max`, `d0` in `oeh_v2_spec.json` |
| Test 1 / Test 2 / Test 3 | four-panel sealed test / fifth panel / panel 6 |

## Evidence rules

- Definitions, comparators and decision rules are frozen in a protocol before the data of a test exist; seals record the SHA-256 of every forecast before any learner outcome exists.
- Sealed files are never modified; runners refuse to overwrite outputs, and every experiment writes to a new dated folder.
- Results on opened data are labeled exploratory or post hoc; negative and mixed results are kept and reported.
- Censored outcomes are never replaced by a finite budget; undefined values stay missing with a reason.

## Team

Le Thi Hai Linh (statistical evaluation and research protocol), Do Trong Hung (OEH core), Tran Quoc Viet (benchmark and oracle), Nguyen Quoc Viet (offline RL and data generation). Supervisor: Assoc. Prof. Phan Duy Hung, FPT University.
