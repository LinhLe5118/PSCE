"""Môi trường thăm dò có ngẫu nhiên thật, để KIỂM ĐỊNH NGHĨA — không phải benchmark.

Cả 14 fixture trong ``environments/tabular_environments.py`` đều deterministic. Trong một
MDP deterministic, mọi bootstrap replicate còn giữ đủ ``A_K`` cho CÙNG một
``P_hat`` và ``r_hat`` (không có transition nào khác để lấy mẫu lại tới), nên
``sigma_hat_Delta = 0`` CHÍNH XÁC ở mọi trạng thái và mọi độ sâu — ``z_k`` là
vô cực, ``m_k`` là 0, đường cong độ sâu-bằng chứng phẳng tuyệt đối và không
mang thông tin (K2 thành "0 = 0", K5 tự động hỏng vì lý do thiết kế, không
vì công thức sai).

Ba MDP dưới đây có ``slip probability`` khác 0, chỉ để estimator
(``code/oeh_estimator.py``) và đường cong chính xác (``code/exact_evidence_curve.py``)
có phương sai thật để đo. Đây KHÔNG phải benchmark 120+15+3 của mục 5 PLAN_Final —
benchmark thật nằm ở ``environments/benchmark_environments.py`` (120+15+3 thể hiện).
Dùng đúng dataclass ``TabularMDP`` đã có trong ``environments/tabular_environments.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.tabular_environments import TabularMDP  # noqa: E402


FloatArray = NDArray[np.float64]


def _empty_mdp(num_states: int, num_actions: int) -> tuple[FloatArray, FloatArray]:
    return (
        np.zeros((num_states, num_actions, num_states), dtype=np.float64),
        np.zeros((num_states, num_actions, num_states), dtype=np.float64),
    )


def make_river_swim(
    num_states: int = 6, *, slip_probability: float = 0.1, gamma: float = 0.95
) -> TabularMDP:
    """RiverSwim: bơi ngược dòng có xác suất trượt, họ hard-exploration kinh điển.

    Hai hành động: ``LEFT`` (xuôi dòng, tất định, thưởng nhỏ ở đầu chuỗi) và
    ``RIGHT`` (ngược dòng): với xác suất ``1 - slip_probability`` tiến lên một
    bước, với xác suất ``slip_probability`` bị trượt lùi lại. Thưởng lớn chỉ ở
    trạng thái cuối cùng khi đi ``RIGHT`` thành công.
    """

    if num_states < 3:
        raise ValueError("num_states phải >= 3.")
    if not 0.0 < slip_probability < 1.0:
        raise ValueError("slip_probability phải thuộc (0, 1).")

    LEFT, RIGHT = 0, 1
    num_actions = 2
    P, R = _empty_mdp(num_states, num_actions)

    for state in range(num_states):
        left_next = max(state - 1, 0)
        P[state, LEFT, left_next] = 1.0
        if state == 0:
            R[state, LEFT, left_next] = 0.0
        else:
            R[state, LEFT, left_next] = 0.0

        if state == num_states - 1:
            P[state, RIGHT, state] = 1.0 - slip_probability
            P[state, RIGHT, max(state - 1, 0)] = slip_probability
            R[state, RIGHT, state] = 1.0
        else:
            right_next = state + 1
            P[state, RIGHT, right_next] = 1.0 - slip_probability
            P[state, RIGHT, max(state - 1, 0)] = slip_probability

    R[0, LEFT, 0] = 0.005

    initial = np.zeros(num_states)
    initial[0] = 1.0
    mdp = TabularMDP(
        environment_id=f"probe_river_swim_{num_states}",
        P=P,
        R=R,
        initial_state_distribution=initial,
        terminal_states=(),
        gamma=gamma,
        metadata={
            "family": "probe_river_swim",
            "num_states": num_states,
            "slip_probability": slip_probability,
            "transition_type": "stochastic",
            "actions": {"left": LEFT, "right": RIGHT},
            "purpose": "definition_probe_not_benchmark",
        },
    )
    mdp.validate()
    return mdp


def make_stochastic_combination_lock(
    chain_length: int,
    *,
    slip_probability: float = 0.1,
    seed: int = 0,
    gamma: float = 0.95,
) -> TabularMDP:
    """Combination lock giữ cấu trúc chuỗi nhưng có slip_probability > 0.

    Mỗi trạng thái có một hành động "đúng" duy nhất (cố định theo seed). Đi
    đúng: với xác suất ``1 - slip_probability`` tiến lên, với xác suất
    ``slip_probability`` bị trượt về trạng thái đầu. Đi sai: luôn về trạng
    thái đầu. Đây là phần phá tính deterministic của
    ``tabular_environments.make_combination_lock``.
    """

    if chain_length < 2:
        raise ValueError("chain_length phải >= 2.")
    if not 0.0 < slip_probability < 1.0:
        raise ValueError("slip_probability phải thuộc (0, 1).")

    num_states, num_actions = chain_length + 1, 2
    terminal = chain_length
    P, R = _empty_mdp(num_states, num_actions)
    rng = np.random.default_rng(seed)
    correct_actions = rng.integers(0, num_actions, size=chain_length).tolist()

    for state, correct_action in enumerate(correct_actions):
        for action in range(num_actions):
            if action == correct_action:
                advance = state + 1
                P[state, action, advance] = 1.0 - slip_probability
                P[state, action, 0] += slip_probability
                if advance == terminal:
                    R[state, action, advance] = 1.0
            else:
                P[state, action, 0] = 1.0
    P[terminal, :, terminal] = 1.0

    initial = np.zeros(num_states)
    initial[0] = 1.0
    mdp = TabularMDP(
        environment_id=f"probe_stochastic_lock_{chain_length}",
        P=P,
        R=R,
        initial_state_distribution=initial,
        terminal_states=(terminal,),
        gamma=gamma,
        metadata={
            "family": "probe_stochastic_combination_lock",
            "chain_length": chain_length,
            "correct_actions": correct_actions,
            "slip_probability": slip_probability,
            "transition_type": "stochastic",
            "seed": seed,
            "purpose": "definition_probe_not_benchmark",
        },
    )
    mdp.validate()
    return mdp


def make_stochastic_gridworld(
    size: int = 5, *, slip_probability: float = 0.1, gamma: float = 0.95
) -> TabularMDP:
    """GridWorld vuông với slip_probability: hành động đôi khi trượt sang cạnh.

    Start ở góc trên trái, goal ở góc dưới phải (khớp
    ``tabular_environments.make_gridworld``). Với xác suất
    ``1 - slip_probability`` di chuyển đúng hướng; với xác suất
    ``slip_probability`` bị chia đều sang hai hướng vuông góc với hướng dự định.
    """

    if size < 2:
        raise ValueError("size phải >= 2.")
    if not 0.0 < slip_probability < 1.0:
        raise ValueError("slip_probability phải thuộc (0, 1).")

    UP, RIGHT, DOWN, LEFT = range(4)
    deltas = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
    perpendicular = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

    coordinates = [(row, col) for row in range(size) for col in range(size)]
    state_of = {coordinate: state for state, coordinate in enumerate(coordinates)}
    num_states, num_actions = len(coordinates), 4
    start, terminal = state_of[(0, 0)], state_of[(size - 1, size - 1)]
    P, R = _empty_mdp(num_states, num_actions)
    step_cost = -0.01

    for state, (row, col) in enumerate(coordinates):
        if state == terminal:
            P[state, :, state] = 1.0
            continue
        for action, (d_row, d_col) in deltas.items():
            intended = state_of.get((row + d_row, col + d_col), state)
            side_a, side_b = perpendicular[action]
            side_a_state = state_of.get(
                (row + deltas[side_a][0], col + deltas[side_a][1]), state
            )
            side_b_state = state_of.get(
                (row + deltas[side_b][0], col + deltas[side_b][1]), state
            )
            P[state, action, intended] += 1.0 - slip_probability
            P[state, action, side_a_state] += slip_probability / 2.0
            P[state, action, side_b_state] += slip_probability / 2.0
            for next_state in (intended, side_a_state, side_b_state):
                reward = 1.0 if next_state == terminal else step_cost
                R[state, action, next_state] = reward

    initial = np.zeros(num_states)
    initial[start] = 1.0
    mdp = TabularMDP(
        environment_id=f"probe_stochastic_gridworld_{size}x{size}",
        P=P,
        R=R,
        initial_state_distribution=initial,
        terminal_states=(terminal,),
        gamma=gamma,
        metadata={
            "family": "probe_stochastic_gridworld",
            "size": size,
            "slip_probability": slip_probability,
            "transition_type": "stochastic",
            "actions": {"up": UP, "right": RIGHT, "down": DOWN, "left": LEFT},
            "purpose": "definition_probe_not_benchmark",
        },
    )
    mdp.validate()
    return mdp


PROBE_ENVIRONMENTS: dict[str, Callable[[], TabularMDP]] = {
    "probe_river_swim_6": lambda: make_river_swim(6),
    "probe_stochastic_lock_5": lambda: make_stochastic_combination_lock(5, seed=0),
    "probe_stochastic_gridworld_5x5": lambda: make_stochastic_gridworld(5),
}


def make_probe_env(environment_id: str) -> TabularMDP:
    try:
        factory = PROBE_ENVIRONMENTS[environment_id]
    except KeyError as error:
        available = ", ".join(sorted(PROBE_ENVIRONMENTS))
        raise KeyError(f"Môi trường thăm dò không tồn tại. Các ID hợp lệ: {available}") from error
    return factory()


if __name__ == "__main__":
    for env_id, factory in PROBE_ENVIRONMENTS.items():
        env = factory()
        env.validate()
        print(f"- {env_id}: S={env.num_states}, A={env.num_actions}, gamma={env.gamma}")
