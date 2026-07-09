import concurrent.futures
import logging
from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

# Rank 0's PDM scoring during validation is CPU-bound (metric-cache pickle loads +
# PDM simulation) and shares the node's CPUs with all DDP ranks' DataLoader workers,
# which can fully saturate the node. If rank 0 ever stalls on a single validation
# batch (CPU contention, a slow NFS read, a ray hiccup, etc.) for longer than the
# NCCL collective timeout (1800s by default), the whole job is aborted by the NCCL
# watchdog because the other 7 ranks are already waiting at the same collective.
# Bounding rank 0's wall-clock time per batch guarantees it always reaches the
# collective well inside that window, regardless of what caused the slowdown.
_VALIDATION_SCORE_TIMEOUT_S = 600


def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)

    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz = False):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz
        self._train_real_total = 0
        self._train_sim_total = 0

    # {STEP}
    # Do a forward pass on drivor agent with the features. Then compute loss against 
    # targets. Log all sub-losses in loss_dict and return overall loss
    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            epoch_only_keys = {"best_score"}
            for key, value in loss_dict.items():
                if key in epoch_only_keys:
                    self.log(f"{logging_prefix}/{key}", value, on_step=False, on_epoch=True,
                             prog_bar=True, sync_dist=True)
                else:
                    self.log(f"{logging_prefix}/{key}", value, on_step=True, on_epoch=False,
                             prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

    # {TRAINING STEP} -> {STEP}
    # PyTorch Lightning directs trainer.fit() into training_step function in module.
    # Mostly a wrapper to call _step() on batch. Do some logging on number of real counts
    # and not real counts in features.
    # Tensorboard train loss logging in _step() function
    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        One epoch iterates through all real training samples and all (subsampled) sim samples exactly once, interleaved randomly.
        """
        features, _ = batch
        if "real" in features:
            real_mask = features["real"]
            self._train_real_total += int(real_mask.sum().item())
            self._train_sim_total += int((~real_mask).sum().item())
        else:
            self._train_real_total += next(iter(features.values())).shape[0]
        return self._step(batch, "train")

    def on_train_epoch_end(self) -> None:
        # Aggregate per-rank counts so the summary reflects the actual data sampled
        # across all GPUs, not just rank 0's shard.
        counts = torch.tensor([self._train_real_total, self._train_sim_total], device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        real_total, sim_total = int(counts[0].item()), int(counts[1].item())
        if self.global_rank == 0:
            total = real_total + sim_total
            real_pct = 100.0 * real_total / total if total > 0 else 0.0
            sim_pct = 100.0 * sim_total / total if total > 0 else 0.0
            sampled_ratio = sim_total / real_total if real_total > 0 else float("nan")
            print(
                f"\n[Epoch {self.current_epoch}] Data mix summary (all ranks):\n"
                f"  real : {real_total:>8d}  ({real_pct:.1f}%)\n"
                f"  sim  : {sim_total:>8d}  ({sim_pct:.1f}%)\n"
                f"  total: {real_total + sim_total:>8d}\n"
                f"  sampled sim/real ratio: {sampled_ratio:.3f}"
            )
        self._train_real_total = 0
        self._train_sim_total = 0

    @staticmethod
    def _score_validation_batch(agent, targets, all_chosen_trajectories, all_proposed_trajectories):
        """
        Pure computation, no self.log() calls: this runs on a worker thread with a
        wall-clock timeout, so it must not touch LightningModule state that isn't
        thread-safe.
        """
        final_score, _fake_best_score, proposal_scores, l2, trajectoy_scores = agent.compute_score(targets, all_chosen_trajectories)
        _, best_score, all_proposal_scores, _, _ = agent.compute_score(targets, all_proposed_trajectories)
        mean_score = proposal_scores.mean()
        return {
            "final_score": final_score,
            "proposal_scores": proposal_scores,
            "l2": l2,
            "trajectoy_scores": trajectoy_scores,
            "best_score": best_score,
            "all_proposal_scores": all_proposal_scores,
            "mean_score": mean_score,
        }

    # Tensorboard val loss logging
    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:,None]
            all_proposed_trajectories = predictions["proposals"]

            # PDM scoring is CPU-heavy (loads metric-cache pickles per proposal).
            # Running it on all 8 DDP ranks simultaneously exhausts node RAM.
            # Only rank 0 runs PDM; other ranks contribute 0.0 for the all-reduce.
            # val/score is logged sync_dist=True on all ranks so val/score_epoch
            # appears in callback_metrics for ModelCheckpoint. The all-reduced value
            # is real_score/8 (zeros from the 7 other ranks), but the scaling is
            # consistent across epochs so checkpoint ordering is preserved.
            # All other detailed PDM metrics are rank-0-only (sync_dist=False).
            logging_prefix = "val"
            if self.global_rank == 0:
                # Bound rank 0's wall-clock time: a fresh single-worker executor per
                # call means a stuck batch (e.g. CPU starvation, NFS stall) never
                # blocks subsequent steps -- it just gets abandoned in the background
                # and this step contributes 0.0, same as the other 7 ranks would.
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    self._score_validation_batch, self.agent, targets, all_chosen_trajectories, all_proposed_trajectories
                )
                executor.shutdown(wait=False)
                try:
                    result = future.result(timeout=_VALIDATION_SCORE_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        f"[Epoch {self.current_epoch}] PDM scoring exceeded "
                        f"{_VALIDATION_SCORE_TIMEOUT_S}s on this validation batch "
                        "(likely CPU/IO contention); contributing 0.0 for this step "
                        "instead of risking the DDP collective timing out."
                    )
                    final_score = torch.tensor(0.0, device=self.device)
                else:
                    final_score = result["final_score"]
                    proposal_scores = result["proposal_scores"]
                    mean_score = result["mean_score"]
                    best_score = result["best_score"]
                    all_proposal_scores = result["all_proposal_scores"]
                    l2 = result["l2"]
                    trajectoy_scores = result["trajectoy_scores"]

                    if "pdm_score" in predictions:
                        pdm_score = predictions["pdm_score"]
                        best_pred_score_values = pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                        score_error = torch.abs(best_pred_score_values - proposal_scores).mean()
                        self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

                        best_pred_score_index = torch.argmax(pdm_score, dim=1)
                        best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                        score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                        best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                        best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                        lost_score = torch.mean(best_possible_scores - best_actual_scores)
                        self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                        self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

                        top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                        top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                        self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

                    self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    collision=trajectoy_scores[:,0].mean()
                    self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    drivable_area_compliance=trajectoy_scores[:,1].mean()
                    self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    ego_progress=trajectoy_scores[:,2].mean()
                    self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    time_to_collision_within_bound=trajectoy_scores[:,3].mean()
                    self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                    comfort=trajectoy_scores[:,4].mean()
                    self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            else:
                final_score = torch.tensor(0.0, device=self.device)

            # All ranks participate in this all-reduce so val/score_epoch is visible
            # to ModelCheckpoint in callback_metrics.
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        current_state = self.state_dict()

        # Fill keys present in the current model but missing from checkpoint
        # (e.g. new modules added for fine-tuning — train from scratch).
        for key in current_state:
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = current_state[key]

        # Drop checkpoint keys that are absent from the current model or have a
        # shape mismatch (e.g. backbone was swapped between runs). Those params
        # are already covered by the fill-in pass above.
        stale_keys = [
            key for key, val in checkpoint["state_dict"].items()
            if key not in current_state or val.shape != current_state[key].shape
        ]
        for key in stale_keys:
            if key in current_state:
                checkpoint["state_dict"][key] = current_state[key]
            else:
                del checkpoint["state_dict"][key]

        # Reset optimizer/scheduler state when the parameter groups don't match.
        checkpoint["optimizer_states"] = []
        checkpoint["lr_schedulers"] = []

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[index]
                }
            else:
                result[token] = {'trajectory': proposal}
        return result
