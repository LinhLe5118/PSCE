"""Collector quỹ đạo tối giản, để kiểm định nghĩa OEH.

Đây là bản thay thế TẠM THỜI, tối giản cho ``scripts/data/generate_offline_datasets.py``
(bản cũ trên nhánh này): bộ sinh dữ liệu cũ không có trường ``trajectory_id``
nên bootstrap ở mức quỹ đạo của mục 3.3 PLAN_Final.md không cài được trên dữ
liệu nó sinh ra, và nó lọc transition hậu kỳ để ép độ phủ về mức mục tiêu —
điều mục 5.3 PLAN_Final.md cấm ("Lọc hậu kỳ phá huỷ diễn giải chính sách hành
vi mà bộ ước lượng dựa vào"). Vì vậy collector này không dùng nó làm nguồn.

Collector này thu quỹ đạo TỰ NHIÊN từ phân bố trạng thái đầu của MDP dưới một
chính sách hành vi cho trước — không exploring starts, không audit transition,
không lọc/chèn/xoá/nhân bản transition sau khi thu. Lúc merge với nhánh Việt
Nguyễn, file này bị thay bằng bộ sinh dữ liệu chính thức của họ
(``scripts/data/generate_offline_datasets.py`` bản mới, năm họ behavior policy).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.tabular_environments import TabularMDP  # noqa: E402


FloatArray = NDArray[np.float64]

HORIZON = 100  # PLAN_Final.md Phụ lục A, đồng bộ với K_max = 100.


@dataclass(frozen=True)
class Transition:
    """Một transition, đúng các trường bắt buộc của mục Đầu mục 3."""

    state: int
    action: int
    reward: float
    next_state: int
    terminal: bool
    truncated: bool
    trajectory_id: int
    timestep: int


@dataclass(frozen=True)
class PilotDataset:
    """Tập thí điểm D0: danh sách transition cộng chỉ mục theo quỹ đạo."""

    transitions: tuple[Transition, ...]
    environment_id: str

    @property
    def num_trajectories(self) -> int:
        return len({t.trajectory_id for t in self.transitions})

    def trajectory_ids(self) -> tuple[int, ...]:
        seen: list[int] = []
        seen_set: set[int] = set()
        for t in self.transitions:
            if t.trajectory_id not in seen_set:
                seen.append(t.trajectory_id)
                seen_set.add(t.trajectory_id)
        return tuple(seen)

    def by_trajectory(self) -> dict[int, tuple[Transition, ...]]:
        grouped: dict[int, list[Transition]] = {}
        for t in self.transitions:
            grouped.setdefault(t.trajectory_id, []).append(t)
        return {tid: tuple(ts) for tid, ts in grouped.items()}


def collect_pilot_dataset(
    mdp: TabularMDP,
    behavior_policy: FloatArray,
    *,
    num_transitions: int,
    horizon: int = HORIZON,
    seed: int = 0,
) -> PilotDataset:
    """Thu quỹ đạo tự nhiên dưới ``behavior_policy`` tới khi đạt ``num_transitions``.

    ``behavior_policy`` có shape ``(S, A)``, mỗi hàng là một phân phối xác
    suất. Trạng thái đầu mỗi quỹ đạo lấy mẫu từ ``mdp.initial_state_distribution``
    của chính MDP — không có exploring starts. Quỹ đạo kết thúc khi chạm
    terminal state (``terminal=True``) hoặc chạm ``horizon`` bước
    (``truncated=True``). Không transition nào bị lọc, chèn, xoá hay nhân bản
    sau khi thu thập; hàm dừng ngay khi đạt đủ ``num_transitions``, kể cả giữa
    một quỹ đạo (quỹ đạo cuối cùng thu được đánh dấu ``truncated=True``).
    """

    if num_transitions <= 0:
        raise ValueError("num_transitions phải dương.")
    if horizon <= 0:
        raise ValueError("horizon phải dương.")
    if behavior_policy.shape != (mdp.num_states, mdp.num_actions):
        raise ValueError("behavior_policy phải có shape (S, A).")
    if not np.allclose(behavior_policy.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Mỗi hàng của behavior_policy phải là một phân phối xác suất.")

    rng = np.random.default_rng(seed)
    terminal_states = set(mdp.terminal_states)
    transitions: list[Transition] = []
    trajectory_id = 0

    while len(transitions) < num_transitions:
        state = int(rng.choice(mdp.num_states, p=mdp.initial_state_distribution))
        for timestep in range(horizon):
            action = int(rng.choice(mdp.num_actions, p=behavior_policy[state]))
            next_state = int(rng.choice(mdp.num_states, p=mdp.P[state, action]))
            reward = float(mdp.R[state, action, next_state])
            is_terminal = next_state in terminal_states
            is_last_step = timestep == horizon - 1
            truncated = is_last_step and not is_terminal

            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminal=is_terminal,
                    truncated=truncated,
                    trajectory_id=trajectory_id,
                    timestep=timestep,
                )
            )

            if len(transitions) >= num_transitions:
                break
            if is_terminal or truncated:
                break
            state = next_state

        trajectory_id += 1

    return PilotDataset(
        transitions=tuple(transitions),
        environment_id=mdp.environment_id,
    )


def uniform_random_behavior_policy(mdp: TabularMDP) -> FloatArray:
    """Chính sách hành vi ngẫu nhiên đều — dùng cho kiểm định nghĩa tối giản."""

    return np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_envs import PROBE_ENVIRONMENTS

    for env_id, factory in PROBE_ENVIRONMENTS.items():
        env = factory()
        beta = uniform_random_behavior_policy(env)
        dataset = collect_pilot_dataset(env, beta, num_transitions=500, seed=42)
        print(
            f"- {env_id}: {len(dataset.transitions)} transitions, "
            f"{dataset.num_trajectories} trajectories"
        )
