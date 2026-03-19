from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class MetaPolicy(nn.Module):
    """PPO Actor 网络。

    这里保留 `act()` 作为线上推理入口，保证现有
    `InferenceService -> MetaPolicy.act -> SafeActionMapper`
    这条链路不被破坏。

    同时额外提供：
    1. `sample_action()`：训练时采样动作并记录对数概率；
    2. `evaluate_actions()`：PPO 更新时重新计算 log_prob / entropy。
    """

    def __init__(self, obs_dim: int = 7, action_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # PPO 中把标准差设成可学习参数，是连续动作任务里最常见、也最稳的一种做法。
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _ensure_batch(self, observation: Iterable[float] | Tensor, target_dim: int) -> Tuple[Tensor, bool]:
        if torch.is_tensor(observation):
            tensor = observation.detach().clone().float()
        else:
            tensor = torch.as_tensor(observation, dtype=torch.float32)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        feature_dim = tensor.shape[-1]
        if feature_dim < target_dim:
            # 兼容旧接口时允许自动补零；动作张量也会走同一逻辑，但 target_dim 会切到 action_dim。
            padding = torch.zeros((tensor.shape[0], target_dim - feature_dim), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=-1)
        elif feature_dim > target_dim:
            tensor = tensor[..., :target_dim]
        return tensor, single

    def forward(self, observation: Iterable[float] | Tensor) -> Tensor:
        obs_tensor, _ = self._ensure_batch(observation, self.obs_dim)
        return self.backbone(obs_tensor)

    def _distribution(self, observation: Iterable[float] | Tensor) -> Tuple[Normal, Tensor]:
        mean = self.forward(observation)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        return Normal(mean, std), mean

    @staticmethod
    def _atanh(value: Tensor) -> Tensor:
        clipped = value.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw_action: Tensor, action: Tensor) -> Tensor:
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def sample_action(self, observation: Iterable[float] | Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """训练时采样动作。

        返回值约定：
        1. `action`：已经过 tanh 压缩到 [-1, 1]；
        2. `log_prob`：PPO 比率项需要的旧策略对数概率；
        3. `entropy`：策略熵，用于鼓励探索。
        """

        distribution, _ = self._distribution(observation)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(distribution, raw_action, action)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, observation: Iterable[float] | Tensor, action: Iterable[float] | Tensor) -> Tuple[Tensor, Tensor]:
        obs_tensor, _ = self._ensure_batch(observation, self.obs_dim)
        action_tensor, _ = self._ensure_batch(action, self.action_dim)
        distribution, _ = self._distribution(obs_tensor)
        raw_action = self._atanh(action_tensor)
        log_prob = self._squashed_log_prob(distribution, raw_action, action_tensor)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def act(self, observation: Iterable[float], deterministic: bool = True) -> list[float]:
        """线上推理默认走确定性动作。

        这样做有两个现实好处：
        1. 推理链路稳定，不会让同一状态每次给出完全不同的调度建议；
        2. 现有 `test_seed_determinism` 更容易稳定通过。
        """

        with torch.no_grad():
            if deterministic:
                action = torch.tanh(self.forward(observation))
            else:
                action, _, _ = self.sample_action(observation)
            return action.squeeze(0).cpu().tolist()
