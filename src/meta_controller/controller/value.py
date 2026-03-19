from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn


class ValueEstimator(nn.Module):
    """PPO Critic 网络。"""

    def __init__(self, obs_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: Iterable[float] | Tensor) -> Tensor:
        if torch.is_tensor(observation):
            tensor = observation.detach().clone().float()
        else:
            tensor = torch.as_tensor(observation, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        feature_dim = tensor.shape[-1]
        if feature_dim < self.obs_dim:
            padding = torch.zeros((tensor.shape[0], self.obs_dim - feature_dim), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=-1)
        elif feature_dim > self.obs_dim:
            tensor = tensor[..., : self.obs_dim]
        return self.network(tensor).squeeze(-1)

    def estimate(self, observation: Iterable[float]) -> float:
        with torch.no_grad():
            return float(self.forward(observation).item())


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    next_values: Sequence[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """计算 GAE 优势值和回报。

    这里显式传入 `next_values`，是为了让每一步 TD 误差都可追踪，不依赖隐式数组偏移。
    """

    advantages = []
    gae = 0.0
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_values[index] * mask - values[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)
    advantages.reverse()
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
    values_tensor = torch.as_tensor(values, dtype=torch.float32)
    returns_tensor = advantages_tensor + values_tensor
    return advantages_tensor, returns_tensor


def explained_variance(targets: Tensor, predictions: Tensor) -> float:
    target_var = torch.var(targets)
    if float(target_var) <= 1e-8:
        return 0.0
    return float(1.0 - torch.var(targets - predictions) / target_var)
