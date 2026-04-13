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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss, compute_topk_kl
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, pid_top_k: int = 0
    ) -> torch.Tensor:
        """Forward pass for a micro-batch.

        Args:
            micro_batch: dict with input_ids, attention_mask, position_ids, responses, etc.
            temperature: sampling temperature for logits scaling
            pid_top_k: if > 0, also compute student top-K distribution for PID.

        Returns:
            log_probs: [B, resp_len] when pid_top_k == 0
            (log_probs, topk_log_probs, topk_probs, topk_ids) when pid_top_k > 0
                topk_log_probs: [B, resp_len, K] — top-K log-softmax values (with grad)
                topk_probs:     [B, resp_len, K] — top-K softmax probs (with grad)
                topk_ids:       [B, resp_len, K] — top-K token indices (detached)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            if pid_top_k > 0:
                # Compute student top-K in unpadded space to avoid materializing (bsz, seqlen, V).
                logits_for_topk = logits_rmpad
                if self.config.ulysses_size > 1:
                    logits_for_topk = gather_outputs_and_unpad(
                        logits_for_topk, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )  # (total_nnz, V)

                # Use gradient checkpoint so the (total_nnz, V) softmax output is NOT
                # kept in memory for backward — it will be recomputed on demand.
                k = pid_top_k

                def _compact_fn(logits):
                    log_probs_full = torch.nn.functional.log_softmax(logits.float(), dim=-1)  # (N, V)
                    _, idx = torch.topk(logits.detach(), k=k, dim=-1)  # (N, K)
                    topk_lp = torch.gather(log_probs_full, -1, idx)  # (N, K)
                    topk_p = topk_lp.exp()  # (N, K) — no full-vocab probs tensor
                    return topk_lp, topk_p, idx

                topk_lp_rmpad, topk_p_rmpad, topk_idx_rmpad = torch.utils.checkpoint.checkpoint(
                    _compact_fn, logits_for_topk, use_reentrant=False
                )

                # Pad back — (total_nnz, K) tensors, small
                full_topk_lp = pad_input(
                    hidden_states=topk_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                )  # (bsz, seqlen, K)
                full_topk_p = pad_input(
                    hidden_states=topk_p_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                )  # (bsz, seqlen, K)
                full_topk_idx = pad_input(
                    hidden_states=topk_idx_rmpad.float(), indices=indices, batch=batch_size, seqlen=seqlen
                ).long()  # (bsz, seqlen, K)

                # Slice to response portion: [B, resp_len, K]
                topk_log_probs = full_topk_lp[:, -response_length - 1 : -1, :]
                topk_probs = full_topk_p[:, -response_length - 1 : -1, :]
                topk_ids = full_topk_idx[:, -response_length - 1 : -1, :]
                return log_probs, topk_log_probs, topk_probs, topk_ids
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

            if pid_top_k > 0:
                # Checkpoint: avoid keeping (B, resp_len, V) softmax output for backward
                k = pid_top_k

                def _compact_fn(lgts):
                    log_probs_full = torch.nn.functional.log_softmax(lgts.float(), dim=-1)  # [B, T, V]
                    _, idx = torch.topk(lgts.detach(), k=k, dim=-1)  # [B, T, K]
                    topk_lp = torch.gather(log_probs_full, -1, idx)  # [B, T, K]
                    topk_p = topk_lp.exp()  # [B, T, K] — no full-vocab probs tensor
                    return topk_lp, topk_p, idx

                topk_log_probs, topk_probs, topk_ids = torch.utils.checkpoint.checkpoint(
                    _compact_fn, logits, use_reentrant=False
                )
                return log_probs, topk_log_probs, topk_probs, topk_ids

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @torch.no_grad()
    def _forward_teacher(
        self,
        hint_input_ids: torch.Tensor,
        hint_attention_mask: torch.Tensor,
        hint_position_ids: torch.Tensor,
        responses: torch.Tensor,
        temperature: float,
        top_k: int = 0,
    ):
        """Teacher forward: hint_prompt + response -> log_probs or own top-K (detached).

        The teacher is the same actor model but with hint-augmented prompt context.
        Uses standard padded forward (not padding-free) for simplicity.

        Args:
            hint_input_ids: [P, hint_len] — left-padded hint prompt token ids
            hint_attention_mask: [P, hint_len] — attention mask for hint prompt
            hint_position_ids: [P, hint_len] or [P, 4, hint_len] (mRoPE) — position ids
            responses: [P, resp_len] — response token ids
            temperature: sampling temperature
            top_k: if > 0, teacher computes its own top-K independently and returns
                (topk_log_probs [P, resp_len, K], topk_ids [P, resp_len, K]).
                If 0, return per-token log_probs [P, resp_len].

        Returns:
            (topk_log_probs, topk_ids) when top_k > 0 — both detached
            log_probs [P, resp_len] when top_k == 0 — detached
        """
        resp_len = responses.size(-1)
        # Concatenate hint prompt + response tokens: [P, hint_len + resp_len]
        teacher_ids = torch.cat([hint_input_ids, responses], dim=-1)
        teacher_attn = torch.cat(
            [hint_attention_mask, torch.ones_like(responses)], dim=-1
        )  # [P, hint_len + resp_len]

        # Build position_ids from attention_mask
        if hint_position_ids.dim() == 3:
            # mRoPE: [P, 4, hint_len] -> extend with response positions
            max_hint_pos = hint_position_ids.max(dim=-1, keepdim=True).values  # [P, 4, 1]
            resp_positions = torch.arange(1, resp_len + 1, device=responses.device)  # [resp_len]
            resp_positions = resp_positions.unsqueeze(0).unsqueeze(0).expand(
                hint_position_ids.size(0), hint_position_ids.size(1), -1
            )  # [P, 4, resp_len]
            resp_positions = resp_positions + max_hint_pos  # [P, 4, resp_len]
            teacher_pos = torch.cat([hint_position_ids, resp_positions], dim=-1)  # [P, 4, hint_len + resp_len]
            teacher_pos = teacher_pos.transpose(0, 1)  # (4, P, hint_len + resp_len)
        else:
            teacher_pos = torch.clip(teacher_attn.cumsum(dim=-1) - 1, min=0)  # [P, hint_len + resp_len]

        output = self.actor_module(
            input_ids=teacher_ids,
            attention_mask=teacher_attn,
            position_ids=teacher_pos,
            use_cache=False,
        )
        # Clone only the response slice [P, resp_len, V] — avoids keeping the full
        # [P, hint_len+resp_len, V] allocation alive via a view into output.logits.
        logits = output.logits[:, -resp_len - 1 : -1, :].detach().clone()  # [P, resp_len, V]
        del output  # free full [P, hint_len+resp_len, V] tensor
        logits.div_(temperature)

        if top_k > 0:
            # Compute top-K indices first (BF16 logits, no large allocation)
            _, topk_ids = torch.topk(logits, k=top_k, dim=-1)  # [P, resp_len, K]
            # Memory-efficient log_softmax at K positions only:
            # log_softmax(x_i) = x_i - logsumexp(x_all) — avoids full [P, T, V] float32
            log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)  # [P, resp_len, 1]
            topk_log_probs = (torch.gather(logits, -1, topk_ids) - log_Z).float()  # [P, resp_len, K]
            del logits  # free [P, resp_len, V]
            torch.cuda.empty_cache()  # return pool fragments to CUDA before next micro-batch
            return topk_log_probs, topk_ids  # both detached
        else:
            log_probs = self.log_probs_from_logits(logits, responses)  # [P, resp_len] — detached
            del logits
            torch.cuda.empty_cache()
            return log_probs

    @torch.no_grad()
    def compute_teacher_log_prob(self, data: DataProto) -> DataProto:
        """Compute teacher log_probs / top-K for all samples using old-policy weights.

        Processes ALL samples (not just PID ones) to guarantee every FSDP rank executes
        the same number of forward passes — avoids deadlock from uneven PID distribution.
        Non-PID outputs are zeroed out by pid_mask and never read downstream.

        Called once by fsdp_workers before update_policy, analogous to compute_log_prob for ref.

        Returns DataProto with tensors:
          pid_top_k > 0: "teacher_topk_log_probs" [B, T, K], "teacher_topk_ids" [B, T, K]
          pid_top_k == 0: "teacher_log_probs" [B, T]
        """
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]

        pid_mask = data.batch["pid_mask"]   # [B]
        K   = self.config.pid_top_k
        mb  = self.config.micro_batch_size_per_device_for_experience

        # Iterate over ALL samples — same as compute_log_prob — so every FSDP rank
        # runs the same number of _forward_teacher calls (avoids all-gather deadlock).
        micro_batches = data.select(
            ["hint_input_ids", "hint_attention_mask", "hint_position_ids", "responses"]
        ).split(mb)

        if K > 0:
            all_lp, all_ids = [], []
            for micro_batch in micro_batches:
                mb_b = micro_batch.batch
                t_lp, t_ids = self._forward_teacher(
                    mb_b["hint_input_ids"], mb_b["hint_attention_mask"],
                    mb_b["hint_position_ids"], mb_b["responses"],
                    temperature, top_k=K,
                )  # [mb, T, K], [mb, T, K] — both detached
                all_lp.append(t_lp)
                all_ids.append(t_ids)

            teacher_lp  = torch.cat(all_lp,  dim=0)  # [B, T, K]
            teacher_ids = torch.cat(all_ids, dim=0)  # [B, T, K]
            # Zero out non-PID positions (never read, but keeps tensors clean)
            mask = pid_mask[:, None, None].float()    # [B, 1, 1]
            return DataProto.from_dict(tensors={
                "teacher_topk_log_probs": teacher_lp  * mask,        # [B, T, K]
                "teacher_topk_ids":       teacher_ids * mask.long(),  # [B, T, K]
            })
        else:
            all_lp = []
            for micro_batch in micro_batches:
                mb_b = micro_batch.batch
                t_lp = self._forward_teacher(
                    mb_b["hint_input_ids"], mb_b["hint_attention_mask"],
                    mb_b["hint_position_ids"], mb_b["responses"],
                    temperature,
                )  # [mb, T] — detached
                all_lp.append(t_lp)

            teacher_lp = torch.cat(all_lp, dim=0)   # [B, T]
            mask = pid_mask[:, None].float()         # [B, 1]
            return DataProto.from_dict(tensors={
                "teacher_log_probs": teacher_lp * mask,         # [B, T]
            })

    def _compute_pid_loss(
        self,
        model_inputs: dict,
        pid_mask: torch.Tensor,
        student_log_probs: torch.Tensor,
        student_topk: tuple = None,
        response_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute PID distillation loss for masked samples.

        Teacher outputs are read from pre-computed keys in model_inputs (populated by
        _precompute_teacher_outputs before the training loop — old-policy snapshot).

        Two modes controlled by self.config.pid_top_k:
        - pid_top_k > 0: Top-K KL on independent top-K sets (reads teacher_topk_log_probs/ids)
        - pid_top_k == 0: Single-token KL using per-token log-probs (reads teacher_log_probs)

        Args:
            model_inputs: dict with pre-computed teacher outputs and response mask
            pid_mask: [B] bool — True for PID-active samples
            student_log_probs: [B, T] — student log probs (with grad)
            student_topk: (topk_log_probs [B,T,K], topk_probs [B,T,K], topk_ids [B,T,K]) or None
            response_mask: [B, T] — response mask

        Returns:
            pid_loss: scalar tensor
        """
        pid_idx = pid_mask.nonzero(as_tuple=True)[0]
        if len(pid_idx) == 0:
            return torch.tensor(0.0, device=student_log_probs.device)

        pid_resp_mask = response_mask[pid_idx]  # [P, T]

        use_topk = self.config.pid_top_k > 0

        if use_topk:
            # Top-K mode: read pre-computed teacher top-K from model_inputs
            s_topk_lp, s_topk_p, s_topk_ids = student_topk
            s_topk_lp  = s_topk_lp[pid_idx]   # [P, T, K] — with grad
            s_topk_p   = s_topk_p[pid_idx]     # [P, T, K] — with grad
            s_topk_ids = s_topk_ids[pid_idx]   # [P, T, K] — detached

            t_topk_lp  = model_inputs["teacher_topk_log_probs"][pid_idx]  # [P, T, K] detached
            t_topk_ids = model_inputs["teacher_topk_ids"][pid_idx]        # [P, T, K] detached

            kl = compute_topk_kl(
                student_topk_log_probs=s_topk_lp,
                student_topk_probs=s_topk_p,
                student_topk_ids=s_topk_ids,
                teacher_topk_log_probs=t_topk_lp,
                teacher_topk_ids=t_topk_ids,
                kl_direction=self.config.pid_kl_direction,
            )  # [P, T]
        else:
            # Single-token mode: read pre-computed teacher log-probs from model_inputs
            s_log_probs = student_log_probs[pid_idx]                  # [P, T] — with grad
            t_log_probs = model_inputs["teacher_log_probs"][pid_idx]  # [P, T] detached
            kl = compute_kl(
                log_probs=s_log_probs,
                ref_log_probs=t_log_probs,
                kl_penalty=self.config.pid_kl_penalty,
                kl_direction=self.config.pid_kl_direction,
            )  # [P, T]

        return average_loss(kl, pid_resp_mask, mode=self.config.loss_avg_mode)

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        if self.config.enable_pid:
            select_keys.extend(["pid_mask", "pid_group_mask"])
            if "teacher_topk_log_probs" in data.batch.keys():
                select_keys.extend(["teacher_topk_log_probs", "teacher_topk_ids"])
            if "teacher_log_probs" in data.batch.keys():
                select_keys.extend(["teacher_log_probs"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    torch.cuda.empty_cache()  # defragment pool from previous micro-batch's variable-length fwd/bwd
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # Determine if we need compact top-K (only for PID top-k mode with active PID)
                    need_topk = (
                        self.config.enable_pid
                        and self.config.pid_top_k > 0
                        and "pid_mask" in model_inputs
                        and model_inputs["pid_mask"].any()
                    )

                    if need_topk:
                        log_probs, topk_log_probs, topk_probs, topk_ids = self._forward_micro_batch(
                            model_inputs, temperature=temperature, pid_top_k=self.config.pid_top_k
                        )  # [B, T], [B, T, K], [B, T, K], [B, T, K]
                        student_topk = (topk_log_probs, topk_probs, topk_ids)
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        student_topk = None

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    # Standard KL loss (now with kl_direction support)
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                            kl_direction=self.config.kl_direction,
                        )

                        # Zero KL for PID groups (PID replaces ref KL entirely)
                        if self.config.enable_pid and "pid_group_mask" in model_inputs:
                            kld = kld * (~model_inputs["pid_group_mask"]).unsqueeze(-1).float()

                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item()})
                        append_to_dict(metrics, {"actor/kl_coef": self.config.kl_coef})
                    else:
                        loss = pg_loss

                    # PID loss (for PID-active samples)
                    if (
                        self.config.enable_pid
                        and "pid_mask" in model_inputs
                        and model_inputs["pid_mask"].any()
                    ):
                        pid_loss = self._compute_pid_loss(
                            model_inputs, model_inputs["pid_mask"],
                            log_probs, student_topk, response_mask,
                        )
                        loss = loss + pid_loss * self.config.pid_coef
                        append_to_dict(metrics, {"actor/pid_kl": pid_loss.detach().item()})
                        append_to_dict(metrics, {"actor/pid_loss": (pid_loss * self.config.pid_coef).detach().item()})
                        append_to_dict(metrics, {"actor/pid_ratio": model_inputs["pid_mask"].float().mean().item()})

                        # Ori-entropy loss: -E[log π(a|s)] on PID-active samples
                        # ≈ H(π_actor) — encourages actor not to collapse toward teacher
                        if self.config.use_ori_entropy_loss:
                            pid_idx = model_inputs["pid_mask"].nonzero(as_tuple=True)[0]
                            ori_entropy_loss = -VF.masked_mean(
                                log_probs[pid_idx], response_mask[pid_idx]
                            )  # scalar
                            loss = loss + self.config.ori_entropy_loss_coef * ori_entropy_loss
                            append_to_dict(metrics, {"actor/ori_entropy_loss": ori_entropy_loss.detach().item()})
                            append_to_dict(metrics, {"actor/ori_entropy_loss_coef": self.config.ori_entropy_loss_coef})

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
