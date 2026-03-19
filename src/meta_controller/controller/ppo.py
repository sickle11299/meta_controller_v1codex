from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from meta_controller.controller.policy import MetaPolicy
from meta_controller.controller.value import ValueEstimator, compute_gae, explained_variance


@dataclass
class RolloutStep:
    """单步轨迹记录。

    PPO 训练依赖这些字段构造 surrogate objective。
    这里用显式 dataclass，而不是散乱 dict，是为了后续排查更稳。
    """

    observation: List[float]
    action: List[float]
    log_prob: float
    reward: float
    done: bool
    value: float
    next_value: float


class PPOTrainer:
    """标准 PPO 训练器。"""

    def __init__(self, policy: MetaPolicy, value_estimator: ValueEstimator, config: Dict[str, object]) -> None:
        ppo_config = config["ppo"]
        self.policy = policy
        self.value_estimator = value_estimator
        self.gamma = float(ppo_config["gamma"])
        self.gae_lambda = float(ppo_config["gae_lambda"])
        self.clip_eps = float(ppo_config["clip_eps"])
        self.target_kl = float(ppo_config["target_kl"])
        self.entropy_coef = float(ppo_config["entropy_coef"])
        self.value_coef = float(ppo_config["value_coef"])
        self.max_grad_norm = float(ppo_config["max_grad_norm"])
        self.update_epochs = int(ppo_config["update_epochs"])
        self.mini_batch_size = int(ppo_config["mini_batch_size"])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(ppo_config["policy_lr"]))
        self.value_optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=float(ppo_config["value_lr"]))

    def train_epoch(self, trajectories: Sequence[RolloutStep]) -> Dict[str, float]:
        if not trajectories:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "explained_variance": 0.0,
            }

        observations = torch.as_tensor([step.observation for step in trajectories], dtype=torch.float32)
        actions = torch.as_tensor([step.action for step in trajectories], dtype=torch.float32)
        old_log_probs = torch.as_tensor([step.log_prob for step in trajectories], dtype=torch.float32)
        rewards = [step.reward for step in trajectories]
        dones = [step.done for step in trajectories]
        values = [step.value for step in trajectories]
        next_values = [step.next_value for step in trajectories]
        advantages, returns = compute_gae(rewards, values, dones, next_values, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        kls: List[float] = []
        early_stopped = False
        sample_count = len(trajectories)

        for _ in range(self.update_epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, self.mini_batch_size):
                indices = permutation[start : start + self.mini_batch_size]
                obs_batch = observations[indices]
                action_batch = actions[indices]
                adv_batch = advantages[indices]
                return_batch = returns[indices]
                old_log_prob_batch = old_log_probs[indices]

                new_log_prob, entropy = self.policy.evaluate_actions(obs_batch, action_batch)
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                surrogate = torch.minimum(ratio * adv_batch, clipped_ratio * adv_batch)
                policy_loss = -surrogate.mean()

                self.policy_optimizer.zero_grad()
                actor_loss = policy_loss - self.entropy_coef * entropy.mean()
                actor_loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                value_prediction = self.value_estimator(obs_batch)
                value_loss = F.mse_loss(value_prediction, return_batch)
                self.value_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                clip_grad_norm_(self.value_estimator.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                approx_kl = float((old_log_prob_batch - new_log_prob).mean().item())
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.mean().item()))
                kls.append(approx_kl)

                if approx_kl > self.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break

        with torch.no_grad():
            final_values = self.value_estimator(observations)
        return {
            "policy_loss": sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
            "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "approx_kl": sum(kls) / len(kls) if kls else 0.0,
            "explained_variance": explained_variance(returns, final_values),
        }
