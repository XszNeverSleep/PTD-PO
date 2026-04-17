# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        advantages[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo", "sapo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        tau_positive: (float)
            The temperature for control the positive tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        tau_negative: (float)
            The temperature for control the negative tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    elif loss_type == "sapo":
        positive_token_mask =  (advantages >= 0).float()
        negative_token_mask =  (advantages < 0).float()
        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))
        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
    kl_direction: str = "forward_kl",
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor — actor (student) log probs, shape [B, T]
        ref_log_probs: torch.Tensor — reference (teacher) log probs, shape [B, T]
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")
        kl_direction: str ("reverse_kl", "forward_kl", "jsd_kl")
            - reverse_kl: KL(actor || ref)  (default, same as original behavior)
            - forward_kl: KL(ref || actor)  — swap P and Q
            - jsd_kl: JSD(actor, ref) = 0.5*KL(actor||M) + 0.5*KL(ref||M), M = 0.5*(actor+ref)

    Returns:
        kl_div: torch.Tensor, shape [B, T]

    """
    # Handle kl_direction by swapping arguments
    if kl_direction == "forward_kl":
        # KL(ref || actor): swap so ref is "P" and actor is "Q"
        log_probs, ref_log_probs = ref_log_probs, log_probs
    elif kl_direction == "jsd_kl":
        # JSD: compute midpoint M = 0.5*(P+Q), then 0.5*KL(P||M) + 0.5*KL(Q||M)
        # Per-token: p = exp(log_probs), q = exp(ref_log_probs), m = 0.5*(p+q)
        log_probs_f, ref_log_probs_f = log_probs.float(), ref_log_probs.float()
        p = log_probs_f.exp()   # [B, T]
        q = ref_log_probs_f.exp()   # [B, T]
        m = 0.5 * (p + q)  # [B, T] midpoint distribution
        log_m = m.clamp(min=1e-10).log()
        # JSD = 0.5 * p * log(p/m) + 0.5 * q * log(q/m)
        jsd = 0.5 * p * (log_probs_f - log_m) + 0.5 * q * (ref_log_probs_f - log_m)
        return jsd
    elif kl_direction != "reverse_kl":
        raise ValueError(f"Unknown kl_direction: {kl_direction}")

    # Existing kl_penalty logic (unchanged) — now operates on possibly-swapped args
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")


def _topk_match_and_gather(
    query_ids: torch.Tensor,
    key_topk_log_probs: torch.Tensor,
    key_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """For each query index, look it up in key's stored top-K and return log-prob.

    Indices not found in key's top-K are approximated as uniform over key's remaining
    tail mass: log(tail_mass / (approx_vocab_size - M)).

    Args:
        query_ids:          [B, T, K] token indices to look up
        key_topk_log_probs: [B, T, M] key's stored top-M log-softmax values
        key_topk_ids:       [B, T, M] key's stored top-M token indices

    Returns:
        [B, T, K] key's log-prob at each query index
    """
    M = key_topk_ids.shape[-1]

    # Match: [B, T, K, 1] == [B, T, 1, M] -> [B, T, K, M]
    match = query_ids.unsqueeze(-1).eq(key_topk_ids.unsqueeze(-2))
    # Sum matched log-probs over M dim: 0 for unmatched, log-prob for matched
    lp_matched = (match.float() * key_topk_log_probs.unsqueeze(-2)).sum(-1)  # [B, T, K]
    has_match = match.any(-1)  # [B, T, K]

    # Uniform approximation for miss indices not in key's stored top-M
    key_tail_mass = (1.0 - key_topk_log_probs.exp().sum(-1)).clamp(min=1e-10)  # [B, T]
    vocab_miss_count = max(1, 150000 - M)
    miss_lp = (key_tail_mass.log() - math.log(vocab_miss_count)).unsqueeze(-1)  # [B, T, 1]

    return torch.where(has_match, lp_matched, miss_lp.expand_as(lp_matched))  # [B, T, K]


def compute_topk_kl(
    student_topk_log_probs: torch.Tensor,
    student_topk_probs: torch.Tensor,
    student_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    kl_direction: str = "forward_kl",
) -> torch.Tensor:
    """KL divergence using independent top-K sets with tail compensation.

    Student and teacher each have their own top-K. Cross-reference via matching:
    for tokens in student's top-K that are missing from teacher's top-K, approximate
    teacher's probability as uniform over the tail.

    Args:
        student_topk_log_probs: [B, T, K] student's top-K log-softmax values (with grad)
        student_topk_probs:     [B, T, K] student's top-K softmax probs (with grad)
        student_topk_ids:       [B, T, K] student's top-K token indices
        teacher_topk_log_probs: [B, T, M] teacher's top-K log-softmax values (detached)
        teacher_topk_ids:       [B, T, M] teacher's top-K token indices (detached)
        kl_direction: 'forward_kl', 'reverse_kl', or 'jsd_kl'

    Returns:
        kl: [B, T] per-token KL divergence
    """
    # Ensure teacher tensors carry no grad (defensive; they should already be detached)
    teacher_topk_log_probs = teacher_topk_log_probs.detach()
    teacher_topk_ids = teacher_topk_ids.detach()

    # Gather teacher log-probs at student's top-K positions: [B, T, K]
    teacher_lp_at_student = _topk_match_and_gather(
        student_topk_ids, teacher_topk_log_probs, teacher_topk_ids
    ).detach()  # detach: no grad through teacher branch
    teacher_probs_at_student = teacher_lp_at_student.exp()  # [B, T, K]

    # Tail mass in fp32 for numerical stability
    p_tail_s = (1.0 - student_topk_probs.sum(-1)).to(torch.float32).clamp(min=1e-8)  # [B, T]
    p_tail_t = (1.0 - teacher_probs_at_student.sum(-1)).to(torch.float32).clamp(min=1e-8)  # [B, T]

    if kl_direction == "forward_kl":
        # KL(student || teacher) = sum_k p_s[k] * (log p_s[k] - log p_t[k]) + p_tail_s * (log p_tail_s - log p_tail_t)
        topk_kl = (student_topk_probs * (student_topk_log_probs - teacher_lp_at_student)).sum(-1)  # [B, T]
        tail_kl = (p_tail_s * (p_tail_s.log() - p_tail_t.log())).to(student_topk_probs.dtype)  # [B, T]
        return topk_kl + tail_kl

    elif kl_direction == "reverse_kl":
        # KL(teacher || student): same student top-K support, swap P and Q
        topk_kl = (teacher_probs_at_student * (teacher_lp_at_student - student_topk_log_probs)).sum(-1)  # [B, T]
        tail_kl = (p_tail_t * (p_tail_t.log() - p_tail_s.log())).to(student_topk_probs.dtype)  # [B, T]
        return topk_kl + tail_kl

    elif kl_direction == "jsd_kl":
        # JSD = 0.5 * KL(S || M) + 0.5 * KL(T || M), M = 0.5 * (S + T)
        # Both terms evaluated on the same student top-K support
        m_probs = 0.5 * (student_topk_probs + teacher_probs_at_student)  # [B, T, K]
        m_log_probs = m_probs.clamp(min=1e-10).log()
        m_tail = 0.5 * (p_tail_s + p_tail_t)  # [B, T]
        m_tail_log = m_tail.clamp(min=1e-10).log()

        # KL(S || M): stop-grad on p_s weight to bound gradient when p_s -> 0
        kl_s_m = (student_topk_probs.detach() * (student_topk_log_probs - m_log_probs)).sum(-1) \
                 + (p_tail_s.detach() * (p_tail_s.log() - m_tail_log)).to(student_topk_probs.dtype)
        # KL(T || M): teacher already detached, no extra stop-grad needed
        kl_t_m = (teacher_probs_at_student * (teacher_lp_at_student - m_log_probs)).sum(-1) \
                 + (p_tail_t * (p_tail_t.log() - m_tail_log)).to(student_topk_probs.dtype)

        return 0.5 * (kl_s_m + kl_t_m)

    raise ValueError(f"Unknown kl_direction: {kl_direction}")


def compute_pid_mask(
    accuracy_scores: torch.Tensor,
    group_index: np.ndarray,
    pid_threshold: float,
    pid_all_trajectories: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine which samples get PID distillation.

    PID activates for groups where group accuracy < pid_threshold.
    Within such groups, only incorrect trajectories receive PID loss by default,
    but ALL samples in these groups skip standard ref KL.

    When pid_all_trajectories=True, ALL trajectories in PID groups (including
    correct ones) receive PID distillation loss.

    Args:
        accuracy_scores: [B] per-sample accuracy (0.0 or 1.0)
        group_index: [B] group uid (np.ndarray of object)
        pid_threshold: float — group accuracy threshold
        pid_all_trajectories: bool — if True, PID loss applies to all trajectories in PID groups

    Returns:
        pid_mask: [B] bool. True = receives PID loss
        pid_group_mask: [B] bool. True = in PID group (skips ref KL, both correct and incorrect)
    """
    bsz = accuracy_scores.shape[0]
    pid_mask = torch.zeros(bsz, dtype=torch.bool)
    pid_group_mask = torch.zeros(bsz, dtype=torch.bool)

    id2indices = defaultdict(list)
    id2acc = defaultdict(list)
    for i in range(bsz):
        id2indices[group_index[i]].append(i)
        id2acc[group_index[i]].append(accuracy_scores[i].item())

    for uid in id2indices:
        group_accuracy = sum(1 for a in id2acc[uid] if a == 1.0) / len(id2acc[uid])
        if group_accuracy < pid_threshold:
            for idx, acc in zip(id2indices[uid], id2acc[uid]):
                pid_group_mask[idx] = True
                if pid_all_trajectories or acc != 1.0:
                    pid_mask[idx] = True

    return pid_mask, pid_group_mask
