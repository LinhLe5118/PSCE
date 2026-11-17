# Exact tabular solver

`exact_solver.py` is oracle-only infrastructure for the finite-MDP benchmark.
It computes exact optimal values, exact policy values, normalized-return labels,
and discounted occupancies. It must not be imported by the data-only OEH
estimator.

An MDP object must expose:

- `P`: transition tensor with shape `(S,A,S)`;
- `R` or `r`: reward array with shape `(S,A)` or `(S,A,S)`;
- `gamma`: discount factor in `[0,1)`;
- `rho0` or `initial_state_distribution`: initial state distribution.

Example:

```python
from exact_solver import evaluate_policy, value_iteration

solution = value_iteration(mdp)
evaluation = evaluate_policy(mdp, solution.policy)

print(solution.values)                 # V*
print(solution.q_values)               # Q*
print(solution.optimal_action_mask)    # every optimal action, including ties
print(evaluation.expected_return)      # rho0 @ V^pi
```

For K4, pass the exact population/support restriction through
`value_iteration(mdp, action_mask=supported_actions)`. Unsupported actions are
never included in a Bellman maximum or returned as optimal.

## Exact k-step backup path

`k_step_backups` implements the frozen recurrence in `PLAN_Final.md`:

```python
from exact_solver import k_step_backups

path = k_step_backups(
    mdp,
    behavior_policy,
    k_max=100,
    action_mask=supported_actions,
)

q_at_depth_0 = path.q_values[0]       # Q^(0) = Q^(beta^K)
q_at_depth_1 = path.q_values[1]       # one supported optimality backup
gap_at_depth_1 = path.action_gaps[1]  # best - second-best supported action
policy_at_depth_1 = path.greedy_policies[1]
```

The behavior policy is renormalized over `supported_actions` before computing
`Q^(0)`. A state with only one supported action receives an action gap of
`NaN`, because no second-best supported action exists.

Run the tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```
