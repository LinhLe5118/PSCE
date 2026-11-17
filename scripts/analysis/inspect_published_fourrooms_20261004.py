"""Read-only schema and outcome audit of the pinned published Minari data."""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/published_minari/D4RL/minigrid/fourrooms-random-v0/data/main_data.hdf5"


def main():
    count = 0
    successes = 0
    steps = 0
    unique = set()
    with DATA.open("rb") as handle, h5py.File(handle, "r") as file:
        for key in sorted(file):
            if not key.startswith("episode_"):
                continue
            episode = file[key]
            rewards = episode["rewards"][:]
            count += 1
            successes += int(np.any(rewards > 0))
            steps += len(rewards)
            if count <= 200:
                obs = episode["observations"]
                for image, direction in zip(obs["image"][:], obs["direction"][:]):
                    unique.add(hashlib.blake2b(image.tobytes() + bytes([int(direction)]), digest_size=8).digest())
            if count % 1000 == 0:
                print(count, successes, steps, flush=True)
    print("total", count, "successes", successes, "steps", steps, "unique_obs_first_200", len(unique), flush=True)


if __name__ == "__main__":
    main()
