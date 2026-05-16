# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DrivoR-specific AgentLightningModule for navhard_two_stage GPU evaluation.

from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for DrivoR agent."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, tokens = batch
        prediction = self.agent.forward(features)
        loss, loss_dict = self.agent.compute_loss(features, targets, prediction, tokens)
        for k, v in loss_dict.items():
            self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, _, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
        result = {}
        for i, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            result[token] = {
                "trajectory": Trajectory(pose),
                "pdm_score": predictions["pdm_score"].cpu().numpy()[i],
                "proposals": predictions["proposals"].cpu().numpy()[i],
            }
        return result
