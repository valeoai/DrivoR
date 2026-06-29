from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

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
        if self.global_rank == 0:
            total = self._train_real_total + self._train_sim_total
            real_pct = 100.0 * self._train_real_total / total if total > 0 else 0.0
            sim_pct = 100.0 * self._train_sim_total / total if total > 0 else 0.0
            print(
                f"\n[Epoch {self.current_epoch} | rank {self.global_rank}] Data mix summary:\n"
                f"  real : {self._train_real_total:>6d}  ({real_pct:.1f}%)\n"
                f"  sim  : {self._train_sim_total:>6d}  ({sim_pct:.1f}%)\n"
                f"  total: {total:>6d}"
            )
        self._train_real_total = 0
        self._train_sim_total = 0

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
                final_score, fake_best_score, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(targets, all_chosen_trajectories)
                _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
                mean_score=proposal_scores.mean()

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
