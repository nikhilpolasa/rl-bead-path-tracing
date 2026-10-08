"""Neural-network components and compatibility helpers for PPO checkpoints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EluMlpExtractor(BaseFeaturesExtractor):
    """Two 256-unit ELU layers used by the historical PPO experiments."""

    def __init__(self, observation_space, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def ppo_policy_kwargs() -> dict:
    """Policy kwargs matching the archived PPO checkpoint architecture."""
    return {
        "features_extractor_class": EluMlpExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [],
        "activation_fn": torch.nn.ELU,
    }


def load_archived_ppo(path: str | Path, *, device: str = "cpu"):
    """Load a historical PPO checkpoint without unpickling its stale class.

    The original PPO checkpoints were saved from a training script where
    ``EluMlpExtractor`` lived in ``__main__``. Stable-Baselines3 therefore
    cloudpickled that class inside ``policy_kwargs``. Across Python/cloudpickle
    versions, reconstructing the embedded class can fail even though the
    checkpoint weights are valid (for example, ``TypeError: super() takes no
    keyword arguments``).

    Supplying the equivalent, repository-owned policy kwargs via
    ``custom_objects`` prevents SB3 from deserializing that fragile embedded
    class while preserving the exact network topology required by the saved
    state dict. Newly trained checkpoints use the importable class above and do
    not need this compatibility path.
    """
    from stable_baselines3 import PPO

    return PPO.load(
        str(path),
        device=device,
        custom_objects={"policy_kwargs": ppo_policy_kwargs()},
    )
