"""
OpenClaw Combined-OPCD API Server
==================================

Combines RL (GRPO) with OPCD (experience-augmented distillation).

Inherits from OEL server (which provides experience accumulation +
experience-augmented teacher log-probs) and adds RL dispatch logic
from the Combined server.

Dispatch table:
    +-----------+----------+-----------------------------------+
    | accepted? | eval +-1 | result                            |
    +-----------+----------+-----------------------------------+
    | yes       | yes      | Combined sample (teacher lp + RL) |
    | yes       | no       | OPCD-only sample (reward=0.0)     |
    | no        | yes      | RL-only sample (reward=eval)      |
    | no        | no       | nothing                           |
    +-----------+----------+-----------------------------------+

Since OEL always accepts (experience is always injected), the "no" rows
are effectively rare / unused.  But we keep them for robustness.
"""

import asyncio
import logging
from typing import Any

import torch

# Import the OEL server as our base class.
# PYTHONPATH must include the openclaw-oel directory.
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openclaw-oel"))
from openclaw_oel_api_server import OpenClawOELAPIServer  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Standalone Slime integration functions (identical to OPD's generate/reward)
# -------------------------------------------------------------------------
def generate(sample: Sample, args=None, **kwargs) -> Sample:
    """Tokenize prompt + generate via SGLang, filling log-probs on Sample."""
    # Reuse OEL's generate (which is identical to OPD's)
    from openclaw_oel_api_server import generate as _oel_generate
    return _oel_generate(sample, args, **kwargs)


def reward_func(sample: Sample, **kwargs) -> dict:
    """Passthrough — reward was already set during sample submission."""
    return sample.reward


class OpenClawCombineOPCDAPIServer(OpenClawOELAPIServer):
    """OEL + RL combined training server.

    Inherits the full OEL machinery:
      - Experience accumulation (extract at session end)
      - Experience-augmented teacher log-prob computation
      - Multi-experience pool, experience truncation, extraction prompts

    Adds RL signal dispatch from the Combined server:
      - When eval_score in {+1, -1}: RL reward is included
      - When both teacher and RL available: combined sample
    """

    @staticmethod
    def _is_valid_rl_score(score) -> bool:
        return score in (1, -1, 1.0, -1.0)

    # ------------------------------------------------------------------
    # OPCD / combined sample: real teacher log-probs, configurable reward.
    # Overrides OEL's _submit_turn_sample to accept a reward parameter.
    # ------------------------------------------------------------------
    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        teacher_result: dict[str, Any],
        reward: float = 0.0,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_log_probs = teacher_result.get("teacher_log_probs") or []
        if len(teacher_log_probs) > len(response_ids):
            teacher_log_probs = teacher_log_probs[: len(response_ids)]
        elif len(teacher_log_probs) < len(response_ids):
            teacher_log_probs = teacher_log_probs + [0.0] * (
                len(response_ids) - len(teacher_log_probs)
            )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.teacher_log_probs = torch.tensor(teacher_log_probs, dtype=torch.float32)

        if self._use_topk_distillation:
            K = self._teacher_topk_request_size
            topk_lp = teacher_result.get("teacher_topk_log_probs") or []
            topk_idx = teacher_result.get("teacher_topk_indices") or []
            if len(topk_lp) > len(response_ids):
                topk_lp = topk_lp[: len(response_ids)]
                topk_idx = topk_idx[: len(response_ids)]
            elif len(topk_lp) < len(response_ids):
                pad_len = len(response_ids) - len(topk_lp)
                topk_lp = [[0.0] * K] * pad_len + topk_lp
                topk_idx = [list(range(K))] * pad_len + topk_idx
            sample.teacher_topk_log_probs = torch.tensor(topk_lp, dtype=torch.float32)
            sample.teacher_topk_indices = torch.tensor(topk_idx, dtype=torch.long)

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": reward}

        tag = "OPCD+RL" if reward != 0.0 else "OPCD"
        logger.info(
            "[OpenClaw-Combine-OPCD] submitted %s sample session=%s index=%d "
            "reward=%.1f prompt_len=%d response_len=%d",
            tag, session_id, sample.index, reward,
            len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # RL-only sample: no real teacher signal, reward = eval_score (+-1).
    # ------------------------------------------------------------------
    async def _submit_rl_turn_sample(
        self, turn_data: dict, session_id: str, eval_score: float,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]

        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (
                len(response_ids) - len(response_logprobs)
            )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = response_logprobs
        # teacher_log_probs = student_log_probs => teacher advantage = 0
        # Only GRPO reward signal contributes.
        sample.teacher_log_probs = torch.tensor(response_logprobs, dtype=torch.float32)

        if self._use_topk_distillation:
            K = self._teacher_topk_request_size
            sample.teacher_topk_log_probs = torch.zeros(len(response_ids), K, dtype=torch.float32)
            sample.teacher_topk_indices = torch.zeros(len(response_ids), K, dtype=torch.long)

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": float(eval_score)}

        logger.info(
            "[OpenClaw-Combine-OPCD] submitted RL sample session=%s index=%d "
            "score=%.1f prompt_len=%d response_len=%d",
            session_id, sample.index, float(eval_score),
            len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # Dispatch: ONE sample per turn, merging both signals when possible.
    # Overrides OEL's _maybe_submit_ready_samples.
    # ------------------------------------------------------------------
    def _maybe_submit_ready_samples(
        self, session_id: str, force_drop_without_next_state: bool = False,
    ):
        oel_tasks = self._oel_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = oel_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    if self._eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                    logger.info(
                        "[OpenClaw-Combine-OPCD] dropped session=%s turn=%d (no teacher task)",
                        session_id, turn_num,
                    )
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            oel_tasks.pop(turn_num, None)
            try:
                result = task.result()
            except Exception as e:
                logger.warning(
                    "[OpenClaw-Combine-OPCD] teacher task failed session=%s turn=%d: %s",
                    session_id, turn_num, e,
                )
                if self._eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            # Record eval score
            eval_score = result.get("eval_score")
            if self._eval_mode and eval_score is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(eval_score)

            # Skip training sample submission in non-training modes
            if self._mode in (self.MODE_EXTRACT, self.MODE_DEPLOY):
                continue

            opcd_accepted = result.get("accepted")
            has_valid_rl = self._is_valid_rl_score(eval_score)

            if opcd_accepted and has_valid_rl:
                # Both signals: combined sample
                self._safe_create_task(
                    self._submit_turn_sample(
                        td, session_id, result, reward=float(eval_score),
                    )
                )
            elif opcd_accepted:
                # OPCD only: teacher distillation, no RL reward
                self._safe_create_task(
                    self._submit_turn_sample(td, session_id, result, reward=0.0)
                )
            elif has_valid_rl:
                # RL only: no teacher signal
                self._safe_create_task(
                    self._submit_rl_turn_sample(td, session_id, float(eval_score))
                )
            # else: both failed, drop
