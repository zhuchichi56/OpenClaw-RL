"""OPCD Full KL(student || teacher) distillation loss with top-K approximation.

Implements context distillation: the teacher sees experience-augmented prompts,
the student sees bare prompts. We minimize KL(student || teacher) over the
teacher's top-K vocabulary tokens plus a tail bin.

This is architecturally identical to topk_distillation_loss.py but adds
monitoring metrics (kl_mean, kl_max) for tracking convergence.

Supports both Megatron and FSDP backends:
  - Megatron: uses fused_vocab_parallel_cross_entropy + tensor-parallel group
  - FSDP: uses pure-PyTorch F.log_softmax (no tensor parallelism)

Usage:
    --loss-type custom_loss
    --custom-loss-function-path opcd_distillation_loss.opcd_distillation_loss_function
    --distill-topk 50
    --disable-compute-advantages-and-returns

Reference: OPCD (arXiv 2602.12275), SDFT (arXiv 2601.19897)
"""

from __future__ import annotations

from argparse import Namespace
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

# --- Backend detection: Megatron vs FSDP ---
# We import Megatron utilities eagerly, but decide at call time whether to
# actually use them (the TP group only exists when running under Megatron).
_MEGATRON_AVAILABLE = False
try:
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_responses
    from slime.utils.ppo_utils import compute_log_probs as _megatron_compute_log_probs

    _MEGATRON_AVAILABLE = True
except ImportError:
    pass


def _use_megatron() -> bool:
    """Return True only if Megatron imports succeeded AND TP group is live."""
    if not _MEGATRON_AVAILABLE:
        return False
    try:
        mpu.get_tensor_model_parallel_group()
        return True
    except (AssertionError, AttributeError):
        return False


# --- FSDP-compatible replacements ---

def _fsdp_get_responses(
    logits: torch.Tensor,
    total_lengths: list[int],
    response_lengths: list[int],
    unconcat_tokens: list[torch.Tensor],
    temperature: float = 1.0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (logits_chunk[R,V], tokens_chunk[R]) per sample -- FSDP version.

    Equivalent to Megatron's get_responses() with cp_size=1, qkv_format="thd".
    """
    logits_2d = logits.squeeze(0)  # [T, V]
    end = 0
    for total_len, resp_len, tokens in zip(total_lengths, response_lengths, unconcat_tokens):
        end += total_len
        start = end - resp_len
        logits_chunk = logits_2d[start - 1 : end - 1]  # autoregressive shift
        tokens_chunk = tokens[-resp_len:]
        if temperature != 1.0:
            logits_chunk = logits_chunk / temperature
        yield logits_chunk, tokens_chunk


def _fsdp_compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute log P(token | context) without Megatron fused kernel."""
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def opcd_distillation_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute OPCD full KL(student || teacher) with top-K + tail trick.

    Mathematics:
        KL(student || teacher) = sum_x P_s(x) * [log P_s(x) - log P_t(x)]

    Top-K approximation:
        Only compute over teacher's top-K tokens + a tail bin capturing the
        remaining probability mass: tail = log(1 - sum(exp(topk_logprobs))).

    Reads ``teacher_topk_log_probs`` ([T, K]) and ``teacher_topk_indices``
    ([T, K]) from the batch --- these are populated by the OPCD API server using
    experience-augmented prompts as the teacher context.
    """
    teacher_topk_logprobs = batch["teacher_topk_log_probs"]
    teacher_topk_indices = batch["teacher_topk_indices"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    K = args.distill_topk

    # Select backend-appropriate functions
    _megatron_active = _use_megatron()
    if _megatron_active:
        tp_group = mpu.get_tensor_model_parallel_group()
        max_seq_lens = batch.get("max_seq_lens", None)
        responses_iter = get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

        def compute_lp(logits_chunk, tokens_col):
            return _megatron_compute_log_probs(logits_chunk, tokens_col, tp_group)
    else:
        temperature = getattr(args, "rollout_temperature", 1.0)
        responses_iter = _fsdp_get_responses(
            logits,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            unconcat_tokens=batch["unconcat_tokens"],
            temperature=temperature,
        )

        def compute_lp(logits_chunk, tokens_col):
            return _fsdp_compute_log_probs(logits_chunk, tokens_col)

    all_student_topk_logps = []
    all_teacher_topk_logps = []
    for i, (logits_chunk, tokens_chunk) in enumerate(responses_iter):
        t_logps = teacher_topk_logprobs[i]
        t_indices = teacher_topk_indices[i]

        if not t_logps.is_cuda:
            t_logps = t_logps.to(device=logits_chunk.device)
        if not t_indices.is_cuda:
            t_indices = t_indices.to(device=logits_chunk.device)

        # Gather student log-probs at teacher's top-K positions
        if _megatron_active:
            # Megatron fused kernel needs per-column calls (clone for autograd)
            student_logps_k = []
            need_clone = torch.is_grad_enabled()
            for k in range(K):
                logit_input = logits_chunk.clone() if need_clone else logits_chunk
                lp_k = compute_lp(logit_input, t_indices[:, k])
                student_logps_k.append(lp_k.squeeze(-1))
            student_topk_logps = torch.stack(student_logps_k, dim=-1)
        else:
            # FSDP: single log_softmax + gather all K at once (memory efficient)
            log_probs = F.log_softmax(logits_chunk, dim=-1)  # [R, V]
            student_topk_logps = log_probs.gather(-1, t_indices[:, :K])  # [R, K]

        all_student_topk_logps.append(student_topk_logps)
        all_teacher_topk_logps.append(t_logps)

    # Concatenate all samples
    student_topk = torch.cat(all_student_topk_logps, dim=0)  # [total_tokens, K]
    teacher_topk = torch.cat(all_teacher_topk_logps, dim=0)  # [total_tokens, K]

    # Tail trick: log(1 - sum_of_topk_probs) for the remaining vocab mass
    student_log_s = torch.logsumexp(student_topk, dim=-1, keepdim=True)
    student_log_s = torch.clamp(student_log_s, max=-1e-7)
    student_tail = torch.log(-torch.expm1(student_log_s))

    teacher_log_s = torch.logsumexp(teacher_topk, dim=-1, keepdim=True)
    teacher_log_s = torch.clamp(teacher_log_s, max=-1e-7)
    teacher_tail = torch.log(-torch.expm1(teacher_log_s))

    # Build K+1 simplex: [topk_logprobs, tail_logprob]
    student_with_tail = torch.cat([student_topk, student_tail], dim=-1)
    teacher_with_tail = torch.cat([teacher_topk, teacher_tail], dim=-1)

    # KL(student || teacher): F.kl_div expects (input=log_Q, target=log_P)
    # where it computes sum P * (log P - log Q) when log_target=True
    # Here input=teacher (log Q), target=student (log P) -> KL(student || teacher)
    per_token_kl = F.kl_div(
        teacher_with_tail,
        student_with_tail,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    kl_loss = sum_of_sample_mean(per_token_kl)

    loss = kl_loss
    entropy_loss = torch.tensor(0.0, device=logits.device)
    if args.entropy_coef != 0.0:
        student_probs = torch.exp(student_with_tail)
        entropy = -(student_probs * student_with_tail).sum(dim=-1)
        entropy_loss = sum_of_sample_mean(entropy)
        loss = loss - args.entropy_coef * entropy_loss

    # Prevent unused-parameter error when no tokens
    if per_token_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    # Monitoring metrics
    reported_loss = {
        "loss": loss.clone().detach(),
        "kl_loss": kl_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
    }

    # Extra OPCD monitoring: KL statistics
    if per_token_kl.numel() > 0:
        reported_loss["kl_mean"] = per_token_kl.mean().detach()
        reported_loss["kl_max"] = per_token_kl.max().detach()
    else:
        reported_loss["kl_mean"] = torch.tensor(0.0, device=logits.device)
        reported_loss["kl_max"] = torch.tensor(0.0, device=logits.device)

    return loss, reported_loss
