"""Bộ môi trường MDP dạng bảng; xem docs/design/ENVIRONMENTS.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
UP, RIGHT, DOWN, LEFT = range(4)


@dataclass(frozen=True)
class TabularMDP:
    """Biểu diễn chuẩn được dùng chung cho mọi môi trường."""

    environment_id: str
    P: FloatArray
    R: FloatArray
    initial_state_distribution: FloatArray
    terminal_states: tuple[int, ...]
    gamma: float
    metadata: dict[str, Any]

    @property
    def num_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.P.shape[1])

    @property
    def expected_rewards(self) -> FloatArray:
        """E[R | s,a], shape (S, A), tương thích solver dùng reward 2-D."""

        return np.sum(self.P * self.R, axis=2)

    def validate(self, atol: float = 1e-12) -> None:
        """Ném ValueError nếu MDP vi phạm schema hoặc xác suất."""

        s, a, s_next = self.P.shape
        if s != s_next or self.R.shape != self.P.shape:
            raise ValueError("P và R phải có shape (S, A, S).")
        if self.initial_state_distribution.shape != (s,):
            raise ValueError("Phân phối trạng thái đầu phải có shape (S,).")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma phải thuộc [0, 1).")
        if not np.all(np.isfinite(self.P)) or not np.all(np.isfinite(self.R)):
            raise ValueError("P và R không được chứa NaN hoặc infinity.")
        if np.any(self.P < 0.0):
            raise ValueError("Xác suất chuyển không được âm.")
        if not np.allclose(self.P.sum(axis=2), 1.0, atol=atol):
            raise ValueError("Tổng P[s,a,:] phải bằng 1.")
        if np.any(self.initial_state_distribution < 0.0) or not np.isclose(
            self.initial_state_distribution.sum(), 1.0, atol=atol
        ):
            raise ValueError("Phân phối trạng thái đầu không hợp lệ.")
        for terminal in self.terminal_states:
            if terminal < 0 or terminal >= s:
                raise ValueError("Terminal state nằm ngoài không gian trạng thái.")
            for action in range(a):
                expected = np.zeros(s)
                expected[terminal] = 1.0
                if not np.allclose(self.P[terminal, action], expected, atol=atol):
                    raise ValueError("Terminal state phải absorbing.")
                if not np.allclose(self.R[terminal, action], 0.0, atol=atol):
                    raise ValueError("Thưởng sau terminal phải bằng 0.")


def _empty_mdp(num_states: int, num_actions: int) -> tuple[FloatArray, FloatArray]:
    return (
        np.zeros((num_states, num_actions, num_states), dtype=np.float64),
        np.zeros((num_states, num_actions, num_states), dtype=np.float64),
    )


def make_combination_lock(
    chain_length: int, *, seed: int = 0, gamma: float = 0.99
) -> TabularMDP:
    """Tạo Combination Lock với chuỗi hành động đúng tái lập theo seed."""

    if chain_length < 2:
        raise ValueError("chain_length phải >= 2.")
    num_states, num_actions = chain_length + 1, 2
    terminal = chain_length
    P, R = _empty_mdp(num_states, num_actions)
    rng = np.random.Generator(np.random.PCG64(seed))
    correct_actions = rng.integers(0, num_actions, size=chain_length).tolist()

    for state, correct_action in enumerate(correct_actions):
        for action in range(num_actions):
            next_state = state + 1 if action == correct_action else 0
            P[state, action, next_state] = 1.0
            if next_state == terminal:
                R[state, action, next_state] = 1.0
    P[terminal, :, terminal] = 1.0

    initial = np.zeros(num_states)
    initial[0] = 1.0
    environment_id = f"combination_lock_length_{chain_length}"
    mdp = TabularMDP(
        environment_id,
        P,
        R,
        initial,
        (terminal,),
        gamma,
        {
            "family": "combination_lock",
            "chain_length": chain_length,
            "correct_actions": correct_actions,
            "transition_type": "deterministic",
            "wrong_action": "reset_to_start",
            "seed": seed,
        },
    )
    mdp.validate()
    return mdp


def make_gridworld(
    size: int,
    *,
    walls: frozenset[tuple[int, int]] = frozenset(),
    seed: int = 0,
    gamma: float = 0.99,
    step_cost: float = -0.01,
) -> TabularMDP:
    """Tạo GridWorld vuông, start góc trên trái và goal góc dưới phải."""

    if size < 2:
        raise ValueError("size phải >= 2.")
    start_coord, goal_coord = (0, 0), (size - 1, size - 1)
    if start_coord in walls or goal_coord in walls:
        raise ValueError("Tường không được che start hoặc goal.")
    coordinates = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in walls
    ]
    state_of = {coordinate: state for state, coordinate in enumerate(coordinates)}
    num_states, num_actions = len(coordinates), 4
    start, terminal = state_of[start_coord], state_of[goal_coord]
    P, R = _empty_mdp(num_states, num_actions)
    deltas = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

    for state, (row, col) in enumerate(coordinates):
        if state == terminal:
            P[state, :, state] = 1.0
            continue
        for action, (d_row, d_col) in deltas.items():
            candidate = (row + d_row, col + d_col)
            next_state = state_of.get(candidate, state)
            P[state, action, next_state] = 1.0
            R[state, action, next_state] = 1.0 if next_state == terminal else step_cost

    initial = np.zeros(num_states)
    initial[start] = 1.0
    environment_id = f"gridworld_{size}x{size}"
    mdp = TabularMDP(
        environment_id,
        P,
        R,
        initial,
        (terminal,),
        gamma,
        {
            "family": "gridworld",
            "size": size,
            "walls": sorted(walls),
            "start_coordinate": start_coord,
            "goal_coordinate": goal_coord,
            "state_coordinates": coordinates,
            "actions": {"up": UP, "right": RIGHT, "down": DOWN, "left": LEFT},
            "transition_type": "deterministic",
            "step_cost": step_cost,
            "seed": seed,
        },
    )
    mdp.validate()
    return mdp


def make_linear_chain(
    length: int, *, seed: int = 0, gamma: float = 0.99
) -> TabularMDP:
    """Tạo chuỗi có reward nhỏ gần bên trái và reward lớn bên phải."""

    if length < 5 or length % 2 == 0:
        raise ValueError("Linear Chain cần length lẻ và >= 5 để có start chính giữa.")
    num_states, num_actions = length, 2
    left_action, right_action = 0, 1
    start, left_terminal, right_terminal = length // 2, 0, length - 1
    P, R = _empty_mdp(num_states, num_actions)

    for state in range(1, length - 1):
        left_state, right_state = state - 1, state + 1
        P[state, left_action, left_state] = 1.0
        P[state, right_action, right_state] = 1.0
        if left_state == left_terminal:
            R[state, left_action, left_state] = 0.1
        if right_state == right_terminal:
            R[state, right_action, right_state] = 1.0
    P[left_terminal, :, left_terminal] = 1.0
    P[right_terminal, :, right_terminal] = 1.0

    initial = np.zeros(num_states)
    initial[start] = 1.0
    environment_id = f"linear_chain_{length}"
    mdp = TabularMDP(
        environment_id,
        P,
        R,
        initial,
        (left_terminal, right_terminal),
        gamma,
        {
            "family": "linear_chain",
            "length": length,
            "start_state": start,
            "left_reward": 0.1,
            "right_reward": 1.0,
            "actions": {"left": left_action, "right": right_action},
            "transition_type": "deterministic",
            "seed": seed,
        },
    )
    mdp.validate()
    return mdp


GRID_WALLS: dict[int, frozenset[tuple[int, int]]] = {
    4: frozenset({(1, 1), (2, 1)}),
    6: frozenset({(1, 1), (1, 2), (2, 4), (3, 2), (4, 2), (4, 4)}),
    8: frozenset(
        {(1, 1), (1, 2), (2, 5), (3, 2), (3, 3), (4, 5), (5, 1), (5, 5), (6, 3)}
    ),
}


EnvironmentFactory = Callable[[int], TabularMDP]


def _lock_factory(length: int) -> EnvironmentFactory:
    return lambda seed=0: make_combination_lock(length, seed=seed)


def _grid_factory(size: int) -> EnvironmentFactory:
    return lambda seed=0: make_gridworld(size, walls=GRID_WALLS[size], seed=seed)


def _chain_factory(length: int) -> EnvironmentFactory:
    return lambda seed=0: make_linear_chain(length, seed=seed)


ENVIRONMENT_REGISTRY: dict[str, EnvironmentFactory] = {
    **{
        f"combination_lock_length_{length}": _lock_factory(length)
        for length in (2, 3, 4, 5, 6, 8, 10, 12)
    },
    **{f"gridworld_{size}x{size}": _grid_factory(size) for size in (4, 6, 8)},
    **{f"linear_chain_{length}": _chain_factory(length) for length in (5, 11, 21)},
}


def make_env(environment_id: str, *, seed: int = 0) -> TabularMDP:
    """Tạo một trong 14 cấu hình chuẩn bằng ID."""

    try:
        factory = ENVIRONMENT_REGISTRY[environment_id]
    except KeyError as error:
        available = ", ".join(sorted(ENVIRONMENT_REGISTRY))
        raise KeyError(f"Môi trường không tồn tại. Các ID hợp lệ: {available}") from error
    return factory(seed)


def run_self_tests() -> None:
    """Kiểm tra schema, ngữ nghĩa cơ bản và seed trên đủ 14 cấu hình."""

    if len(ENVIRONMENT_REGISTRY) != 14:
        raise AssertionError("Registry phải chứa đúng 14 cấu hình.")

    for environment_id in ENVIRONMENT_REGISTRY:
        first = make_env(environment_id, seed=42)
        second = make_env(environment_id, seed=42)
        first.validate()
        if not np.array_equal(first.P, second.P) or not np.array_equal(first.R, second.R):
            raise AssertionError(f"{environment_id} không tái lập được theo seed.")

    lock = make_combination_lock(5, seed=7)
    state, total_reward = 0, 0.0
    for correct_action in lock.metadata["correct_actions"]:
        next_state = int(np.argmax(lock.P[state, correct_action]))
        total_reward += lock.R[state, correct_action, next_state]
        state = next_state
    if state != lock.terminal_states[0] or not np.isclose(total_reward, 1.0):
        raise AssertionError("Đi đúng Combination Lock phải đến terminal và nhận thưởng 1.")

    wrong_action = 1 - lock.metadata["correct_actions"][0]
    if int(np.argmax(lock.P[0, wrong_action])) != 0:
        raise AssertionError("Đi sai Combination Lock phải quay về start.")

    grid = make_env("gridworld_4x4")
    start = int(np.argmax(grid.initial_state_distribution))
    if int(np.argmax(grid.P[start, UP])) != start or int(np.argmax(grid.P[start, LEFT])) != start:
        raise AssertionError("GridWorld không được đi xuyên biên.")

    chain = make_env("linear_chain_5")
    chain_start = int(np.argmax(chain.initial_state_distribution))
    if int(np.argmax(chain.P[chain_start, 0])) != chain_start - 1:
        raise AssertionError("Linear Chain di chuyển trái không đúng.")
    if int(np.argmax(chain.P[chain_start, 1])) != chain_start + 1:
        raise AssertionError("Linear Chain di chuyển phải không đúng.")


if __name__ == "__main__":
    run_self_tests()
    print("PASS: 14/14 environments valid; reproducibility and semantic tests passed.")
    for env_id in sorted(ENVIRONMENT_REGISTRY):
        env = make_env(env_id, seed=42)
        print(f"- {env_id}: S={env.num_states}, A={env.num_actions}, P/R={env.P.shape}")
