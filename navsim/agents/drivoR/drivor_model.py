from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.image_encoder.dinov2_hf import HFDINOv2Encoder
from .layers.utils.mlp import MLP
from .v2r_adapter import V2RAdapter
from .matrix_adapter import MatrixAdapter
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            if config_image_backbone.get("use_hf_dinov2", False):
                self.image_backbone = HFDINOv2Encoder(config_image_backbone)
            else:
                self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            if self._config.get("use_adapter", False):
                output_size = (config["image_size"][1], config["image_size"][0])
                if self._config.get("use_matrix_adapter", False):
                    self.adapter = MatrixAdapter(output_size=output_size)
                else:
                    self.adapter = V2RAdapter(output_size=output_size)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        self.b2d=config.b2d

        if self._config.get("freeze_perception", False):
            if hasattr(self, "image_backbone"):
                for param in self.image_backbone.parameters():
                    param.requires_grad = False
            if hasattr(self, "lidar_backbone"):
                for param in self.lidar_backbone.parameters():
                    param.requires_grad = False
            # scene_embeds / lidar_scene_embeds are standalone parameters that
            # feed into the backbone. HFDINOv2Encoder ignores scene_tokens, so
            # these would be trainable-but-unused and trigger a DDP error.
            if hasattr(self, "scene_embeds"):
                self.scene_embeds.requires_grad = False
            if hasattr(self, "lidar_scene_embeds"):
                self.lidar_scene_embeds.requires_grad = False
            # Adapter backprop flows through the frozen ViT; recompute activations
            # to avoid storing all block outputs in VRAM on A5000s.
            if hasattr(self, "image_backbone") and hasattr(self.image_backbone, "model"):
                _m = self.image_backbone.model
                _vit = _m.lora_vit if hasattr(_m, "lora_vit") else _m
                _vit.grad_checkpointing = True
                print("[freeze_perception] gradient checkpointing enabled on image backbone")

        if self._config.get("freeze_all_except_adapter", False):
            for param in self.parameters():
                param.requires_grad = False
            if hasattr(self, "adapter"):
                for param in self.adapter.parameters():
                    param.requires_grad = True

            # Gradient checkpointing: recompute activations on backward instead of
            # storing them. Required when backpropping through a frozen ViT to reach
            # the adapter — without this, all block activations are kept alive and
            # exhaust VRAM on A5000 (23.5 GiB) even with a small batch size.
            if hasattr(self, "image_backbone") and hasattr(self.image_backbone, "model"):
                _m = self.image_backbone.model
                _vit = _m.lora_vit if hasattr(_m, "lora_vit") else _m
                _vit.grad_checkpointing = True
                print("[freeze_all_except_adapter] gradient checkpointing enabled on image backbone")

        # Fully unfrozen mode: backbone is trainable, so activations from all ViT
        # blocks must be stored for backward — exhausts A5000 VRAM at batch_size=10.
        # Enable gradient checkpointing to recompute activations instead of storing them.
        if (not self._config.get("freeze_perception", False)
                and not self._config.get("freeze_all_except_adapter", False)):
            if hasattr(self, "image_backbone") and hasattr(self.image_backbone, "model"):
                _m = self.image_backbone.model
                _vit = _m.lora_vit if hasattr(_m, "lora_vit") else _m
                _vit.grad_checkpointing = True
                print("[unfrozen] gradient checkpointing enabled on image backbone")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)

            # ── V2R adapter: translate sim images into real-image space ──────────
            # `real` is a (B,) integer tensor: 1 = real sample, 0 = sim sample.
            # Real samples have no trainable parameters in their forward path
            # (adapter is skipped), so we run their backbone pass under no_grad
            # to avoid paying gradient-checkpointing cost for 75% of the batch.
            if self._config.get("use_adapter", False):
                real = features.get(
                    "real", torch.ones(batch_size, dtype=torch.long, device=img.device)
                )
                sim_mask = (real == 0)
                B_img, N_img, C_img, H_img, W_img = img.shape

                # When backbone is frozen, real images produce no gradient — skip
                # gradient checkpointing overhead by running them under no_grad.
                # When backbone is trainable, real images must run with grad enabled.
                _freeze_all = self._config.get("freeze_all_except_adapter", False)

                if not sim_mask.any():
                    # All real — adapter not involved.
                    ctx = torch.no_grad() if _freeze_all else torch.enable_grad()
                    with ctx:
                        image_scene_tokens = self.image_backbone(img, scene_tokens)
                else:
                    sim_idx = sim_mask.nonzero(as_tuple=False).view(-1)

                    # Apply adapter to sim images only.
                    sim_flat = img[sim_idx].reshape(len(sim_idx) * N_img, C_img, H_img, W_img)
                    sim_img = self.adapter(sim_flat.contiguous()).view(len(sim_idx), N_img, C_img, H_img, W_img)

                    # Sim path: needs gradient for adapter backprop.
                    sim_tokens = self.image_backbone(sim_img, scene_tokens[sim_idx])

                    real_mask = ~sim_mask
                    if real_mask.any():
                        real_idx = real_mask.nonzero(as_tuple=False).view(-1)
                        ctx = torch.no_grad() if _freeze_all else torch.enable_grad()
                        with ctx:
                            real_tokens = self.image_backbone(img[real_idx], scene_tokens[real_idx])

                        # Reconstruct full batch in original order.
                        image_scene_tokens = torch.zeros(
                            batch_size, *sim_tokens.shape[1:],
                            device=img.device, dtype=sim_tokens.dtype
                        )
                        image_scene_tokens = image_scene_tokens.index_put((real_idx,), real_tokens)
                        image_scene_tokens = image_scene_tokens.index_put((sim_idx,), sim_tokens)
                    else:
                        # All sim.
                        image_scene_tokens = sim_tokens
            else:
                image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        if hasattr(self, "adapter"):
            output["_adapter_anchor"] = 0.0 * sum(
                p.sum() for p in self.adapter.parameters()
            )

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        _eps = 1e-7
        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().clamp(min=_eps).log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().clamp(min=_eps).log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().clamp(min=_eps).log() +
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()
        + self._config.comfort * pred_logit['comfort'].sigmoid()).clamp(min=_eps).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output



