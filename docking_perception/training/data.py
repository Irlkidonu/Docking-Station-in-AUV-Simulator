"""Torch dataset over the frozen P1 shards.

Runs in the training venv. Loads ``.npz`` shards written by the capture side --
the two interpreters share nothing else.

Turbidity is applied here, at load time, from the stored *raw* frame. That is
what lets the held-out c = 2.0 condition be evaluated without ever having been
captured, and it keeps a single physics run serving every water condition.

Labels are reached through an explicit ``with_labels`` flag. An evaluation or
training loader sets it; nothing on the perception path does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..observation.acoustic import encode
from ..observation.optical import to_network_input

__all__ = ["DockingDataset", "Normalisation", "compute_normalisation"]


@dataclass
class Normalisation:
    """Label statistics. **Computed on the training split only.**

    Using train+val+test would leak test-set structure into every model's output
    scaling, which is subtle and would survive every other check in this project.
    """
    mean: np.ndarray                      # (3,) x, y, z
    std: np.ndarray                       # (3,)

    def encode(self, pose: np.ndarray) -> np.ndarray:
        out = pose.copy()
        out[..., :3] = (pose[..., :3] - self.mean) / self.std
        return out

    def decode(self, pose: np.ndarray) -> np.ndarray:
        out = pose.copy()
        out[..., :3] = pose[..., :3] * self.std + self.mean
        return out

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Normalisation":
        return cls(np.array(data["mean"], np.float32),
                   np.array(data["std"], np.float32))


def _shard_paths(root: Path, split: str) -> list[Path]:
    paths = []
    for path in sorted(Path(root).glob("*.npz")):
        with np.load(path, allow_pickle=True) as shard:
            if str(shard["split"]) == split:
                paths.append(path)
    return paths


def compute_normalisation(root: Path) -> Normalisation:
    poses = []
    for path in _shard_paths(root, "train"):
        with np.load(path, allow_pickle=True) as shard:
            poses.append(shard["label_pose"][:, :3])
    stacked = np.concatenate(poses).astype(np.float32)
    std = stacked.std(axis=0)
    # Floor at 5 cm. The closed-loop controller pulls the vehicle onto the dock
    # axis quickly, so y and z spend most samples near zero and their raw std
    # can fall to a few millimetres. Dividing by that would amplify label noise
    # by two orders of magnitude and make the loss chase quantisation. 5 cm is
    # also the scale below which position error stops mattering for docking.
    std = np.maximum(std, 0.05)
    return Normalisation(stacked.mean(axis=0).astype(np.float32),
                         std.astype(np.float32))


class DockingDataset(Dataset):
    """Frames from one split, with turbidity applied on the fly.

    ``override_c`` forces every sample to a single turbidity, which is how the
    per-condition and held-out evaluations are produced from the same episodes.
    """

    def __init__(self, root: Path, split: str,
                 normalisation: Normalisation | None = None,
                 with_labels: bool = True, override_c: float | None = None,
                 cache: bool = True,
                 exclude_frame_indices: tuple = ()) -> None:
        self.root = Path(root)
        self.split = split
        self.with_labels = with_labels
        self.override_c = override_c
        self.normalisation = normalisation
        self._episodes = []

        for path in _shard_paths(self.root, split):
            with np.load(path, allow_pickle=True) as shard:
                episode = json.loads(str(shard["episode"]))
                entry = {
                    "camera_raw": shard["camera_raw"] if cache else None,
                    "fls_range": shard["fls_range"],
                    "fls_valid": shard["fls_valid"],
                    "standoff_m": shard["standoff_m"],
                    "turbidity_c": float(episode["turbidity_c"]),
                    "family": episode["family"],
                    "episode_id": episode["episode_id"],
                    "path": path,
                    "n": int(shard["t_sim"].shape[0]),
                    "t_sim": shard["t_sim"] - shard["t_sim"][0],
                }
                if with_labels:
                    entry["label_pose"] = shard["label_pose"].astype(np.float32)
            self._episodes.append(entry)

        #: Frame indices dropped from this view. Empty by default, so training
        #: and every existing caller are unaffected; held-out metric evaluation
        #: passes evaluation.heldout.exclusion_for_split(split).
        self.excluded_frame_indices = tuple(exclude_frame_indices)
        drop = set(self.excluded_frame_indices)
        self._index = [(e, i) for e, entry in enumerate(self._episodes)
                       for i in range(entry["n"]) if i not in drop]

    def __len__(self) -> int:
        return len(self._index)

    def turbidity_of(self, index: int) -> float:
        episode, _ = self._index[index]
        entry = self._episodes[episode]
        return (self.override_c if self.override_c is not None
                else entry["turbidity_c"])

    def __getitem__(self, index: int):
        episode, frame = self._index[index]
        entry = self._episodes[episode]
        turbidity = (self.override_c if self.override_c is not None
                     else entry["turbidity_c"])

        image = to_network_input(entry["camera_raw"][frame], turbidity,
                                 float(entry["standoff_m"][frame]))
        scan = encode(entry["fls_range"][frame], entry["fls_valid"][frame])

        sample = {"image": torch.from_numpy(image),
                  "scan": torch.from_numpy(scan)}
        if self.with_labels:
            pose = entry["label_pose"][frame].copy()
            target = pose.copy()
            if self.normalisation is not None:
                target = self.normalisation.encode(target)
            sample["target"] = torch.from_numpy(target.astype(np.float32))
            sample["pose_raw"] = torch.from_numpy(pose.astype(np.float32))
            sample["turbidity"] = torch.tensor(turbidity, dtype=torch.float32)
            sample["range"] = torch.tensor(float(np.linalg.norm(pose[:3])),
                                           dtype=torch.float32)
            sample["t_episode"] = torch.tensor(
                float(entry["t_sim"][frame]), dtype=torch.float32)
        return sample
