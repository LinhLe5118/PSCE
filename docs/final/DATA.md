# The data

## Why the data were generated

Checking a forecast of the data requirement needs, for each problem, a pilot, separate training data at increasing
sizes, repeated runs of every learner and exact evaluation of the trained policies. Published offline RL datasets
(for example D4RL or the Minari FourRooms data) provide fixed logs, but neither an exact model of the environment nor
repeated logs of one environment at several sizes. The project therefore collected its own transitions on MDPs whose
models are known, and ran the learners to measure the labels.

## Sources

| Source | Used in | What the project uses or changes |
|---|---|---|
| Project generator: Garnet, layered and bottleneck MDPs | meta-training, development, Test 1 | implements the generators and samples new stochastic MDPs |
| Project generator: deterministic chains | meta-training only, as controls | a wrong action returns the agent to the start |
| Colosseum | development, Test 1 | the authors' sampler and transitions; rewards replaced by their means |
| SPIBB benchmark | development, Test 1 | the authors' random MDPs together with their behavior policies |
| MDPToolbox forest model | Test 1 | varies the parameters; rewards divided by their maximum (pre-outcome amendment 1) |
| Gymnasium FrozenLake | Test 2 | random-map generator with slippery dynamics |
| Gambler's problem | Test 2 | implements the published problem; varies target capital and win probability |
| MDP Playground | Test 3 | sets size, transition noise and reward density; exports the MDPs |
| labmaze | Test 3 | generated layouts plus four-way moves with slipping and reward 1 at the goal |

All MDPs share one format (transition probabilities, rewards, start distribution) with `gamma = 0.95`; terminal states
restart from the start distribution. Before any data were collected, each candidate MDP was kept only if its optimal
return exceeds the random return by at least 0.2; this screen uses neither learner results nor forecasts. All MDPs
have at most 256 state–action pairs.

## Behavior policies

Except for SPIBB, which keeps its authors' policy, each MDP is logged by five behavior policies, each with three
parameter levels assigned in a fixed rotation: softmax of the optimal action values, an epsilon-soft optimal policy of
the MDP with noisy rewards, an epsilon-soft greedy policy of a random low-rank score, a mixture of the best and the
worst action of a random score, and an epsilon-soft optimal policy of an MDP with partly exchanged transitions. The
optimal values are used only to choose the logged actions and are never given to OEH.

## From an MDP to a label

For each MDP and behavior policy the project collects one pilot and ten independent training streams, with separate
seeds and episodes of at most 100 steps. Each transition stores state, action, reward, next state, episode, step and
end flags.

- The **pilot** has `10|S||A|` transitions; its first `2|S||A|` and `5|S||A|` transitions also serve the pilot
  diagnostics. It is the only input of the forecast, with the learner name and `|S|`, `|A|`, the start distribution and
  `gamma`.
- The first `b|S||A|` transitions of a training stream form the sample at budget `xb`, so larger budgets contain the
  smaller ones. BC, CFQI and IQL are trained at each of the six budgets on all ten streams, and each policy is
  evaluated exactly. No pilot transition enters a training sample.
- The label follows the rule in [METHOD.md](METHOD.md). One MDP logged by five behavior policies gives 15 cases (5
  pilots × 3 learners); an SPIBB MDP gives 3.

## Parts of the dataset

- **Meta-training** (the heads are fitted on it): 720 MDPs of the project generator from seven opened cohorts;
  504 stochastic MDPs give the 7,560 cases the heads use, and 216 deterministic chains (3,240 cases) are controls. The
  folders are listed in [artifacts/README.md](../../artifacts/README.md#meta-training-inputs-read-when-the-heads-are-fitted).
- **Development** (choice of settings and of the gate; no final test reuses it): two further generator cohorts, two
  Colosseum sets drawn with different sampler seeds, one SPIBB set, and a generator cohort logged by alternative
  behavior policies, which was examined and not adopted.
- **Final tests**: 660 new MDPs and 8,604 cases in three tests, listed with their seals in
  [FINAL_TESTS.md](FINAL_TESTS.md). Training and validation splits keep all cases of an MDP together, and test kernels
  were checked against earlier ones for duplicates.

## What is in the repository

For every final test, each source folder holds the pilot-only inputs (`query_preoutcome.csv.gz`), the forecasts, the
seal, the learner outcomes (`outcomes.csv.gz`) and the labels (`labels.csv`). The raw transitions are not committed
(`datasets/` is local). The runner of each test regenerates them with `prepare --panel <source>` and the labels with
`evaluate --panel <source>`; the frozen data generator is restored from checked-in bytes by
[`scripts/restore_frozen_runtime.py`](../../scripts/restore_frozen_runtime.py). Regenerating the labels takes days of
learner runs and is not needed to reproduce the results (see the root README).
