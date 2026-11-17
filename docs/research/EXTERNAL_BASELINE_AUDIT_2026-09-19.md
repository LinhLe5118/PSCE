# External baseline audit — 19 September 2026

## Scope rule

The research target is offline RL on stochastic, finite, discrete-state and
discrete-action MDPs. A repository is a direct comparator only when its learner
can consume the same logged transitions and its policy can be scored under the
same exact normalized-return contract. Standard-task reproductions on Atari or
D4RL are provenance/runtime checks, not same-data quality comparisons.

## Repository decision matrix

| Method | Primary source and repository | Compatibility | Decision |
|---|---|---|---|
| Pi_b-SPIBB | [PMLR paper](https://proceedings.mlr.press/v97/laroche19a.html), [author repository](https://github.com/RomainLaroche/SPIBB) | Finite tabular MDP; discrete actions | Included. Author core is imported unmodified at commit `833f32b...`; the complete 75-group, 4,500-run development grid is finished. |
| VI-LCB | [Annals of Statistics paper](https://arxiv.org/abs/2204.05275), [author repository](https://github.com/Laixishi/Model-based-VI-LCB) | The paper covers discounted tabular MDPs, but the repository only implements the finite-horizon Gambler experiment | Included as an explicitly labeled paper-spec port of discounted Algorithm 1/equation (28), pinned to author commit `37858d3...`. Both `c_b=0.05` from the author notebook and the paper's example theoretical constant `c_b=144` are fixed before the full run. |
| Discrete BCQ | [ICML paper](https://proceedings.mlr.press/v97/fujimoto19a.html), [author repository](https://github.com/sfujim/BCQ) | Discrete-action fully connected implementation accepts the same transition tuples | Included in a fixed-budget neural robustness panel. The author learner is imported unmodified at commit `4876f7e...`; one-hot state encoding and CUDA-resident replay are adapter operations. |
| Discrete CQL | [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html), [author repository](https://github.com/aviralkumar2907/CQL) | Original discrete code is coupled to Atari replay and QR-DQN | Included through [d3rlpy](https://github.com/takuseno/d3rlpy), a documented JMLR benchmark implementation, at commit `38c34b6...`. This is not described as the untouched original CQL repository. |
| IQL | [official repository](https://github.com/ikostrikov/implicit_q_learning) | Original implementation assumes continuous actions and D4RL tasks | Excluded from direct neural comparison. The project's registered tabular IQL remains in the same-data panel; the official JAX repository is retained only as a compatibility/runtime probe. |
| APVI | [NeurIPS paper](https://proceedings.neurips.cc/paper/2021/hash/212ab20dbdf4191cbcdcf015511783f4-Abstract.html) | Finite-horizon, stage-indexed formulation; no author implementation was found in the paper or author publication page | Excluded from this discounted stationary benchmark. Porting it would change the problem and would not be an author-code comparison. |

## Neural protocol fixed before the full run

- Development split only; the 120 evaluation MDPs remain sealed.
- Budget `10|S||A|`, all 75 MDP–behavior groups, registered data seed 0.
- One-hot states, exact logged next state, exact project policy scoring.
- Two hidden layers of 256 units, batch size 64, 500 gradient updates.
- BCQ: threshold 0.3, Adam learning rate `3e-4`, Polyak `tau=0.005`.
- CQL: alpha 1.0, learning rate `6.25e-5`, target synchronization every 50 updates.
- RTX 3050 Ti Laptop GPU (`cuda:0`); BCQ and CQL run concurrently. The panel is exploratory because it uses one data/model seed and a compute-limited update budget.

## Interpretation limits

The tabular methods define the main scientific comparison because they match the
paper's state/action assumptions and full learning-curve protocol. The neural
panel tests whether conclusions at the selected budget are obviously overturned
by standard discrete deep offline-RL learners. It cannot support neural SOTA
claims, and it does not turn the tabular paper into a deep-RL benchmark.
