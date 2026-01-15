import os
import sys
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# Update these paths for the release environment
REPO_ROOT = Path("/path/to/your/drivor_repo")  # e.g. /home/user/workspace/drivor
DATASET_ROOT = Path("/path/to/datasets")       # e.g. /datasets_local/navsim_workspace

# Experiment Settings
CHECKPOINT_PATH = REPO_ROOT / "your ckpt"
EXPERIMENT_NAME = "cosine_similarity_analysis"
AGENT_NAME = "drivoR"
SPLIT = "navtrain"

# Compute Settings
NUM_WORKERS = 4
BATCH_SIZE = 32
OUTPUT_DIR = Path("results/similarity_analysis")

# ==========================================
# 2. DDP & ENVIRONMENT SETUP
# ==========================================
def setup_environment(rank, local_rank):
    """Sets up paths and suppresses logs on non-master nodes."""
    # Add repo to path
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    # Set dataset environment variables
    os.environ.update({
        "NUPLAN_MAPS_ROOT": str(DATASET_ROOT / "dataset/maps"),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "OPENSCENE_DATA_ROOT": str(DATASET_ROOT / "dataset/openscene-v1.1"),
        "NAVSIM_EXP_ROOT": str(DATASET_ROOT / "exp"),
        "NAVSIM_DEVKIT_ROOT": str(REPO_ROOT),
    })

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Environment configured. Saving results to: {OUTPUT_DIR}")

def init_distributed():
    """Initializes the process group for Distributed Data Parallel."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Fallback for local debugging
        print("Warning: DDP variables not found. Running in single-process mode.")
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank

# Initialize DDP
rank, world_size, local_rank = init_distributed()
setup_environment(rank, local_rank)

# Now safe to import navsim modules
import navsim.common.dataclasses as dc
import navsim.common.dataloader as dl
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.training.dataset import Dataset

# ==========================================
# 3. LOADER & AGENT SETUP
# ==========================================
def load_components(repo_root, checkpoint_path, agent_name, split):
    config_dir = repo_root / "navsim/planning/script/config/training"
    agent_config_dir = repo_root / "navsim/planning/script/config/common/agent"

    # Define Overrides
    overrides = [
        f"train_test_split={split}",
        f"experiment_name={EXPERIMENT_NAME}",
        f"dataloader.params.num_workers={NUM_WORKERS}",
        f"dataloader.params.batch_size={BATCH_SIZE}"
    ]
    
    agent_overrides = [
        f"checkpoint_path={str(checkpoint_path)}",
        "config.shared_refiner=false", "lr_args=null", "scheduler_args=null",
        "progress_bar=false", "config.refiner_ls_values=0.0",
        "config.image_backbone.focus_front_cam=false", "config.one_token_per_traj=true",
        "config.refiner_num_heads=1", "config.tf_d_model=256", "config.tf_d_ffn=1024",
        "config.area_pred=false", "config.agent_pred=false", "config.ref_num=4",
        "loss.prev_weight=0.0", "batch_size=null",
        # Specific Scorer params
        "config.noc=10.2", "config.dac=12.5", "config.ddc=6.0",
        "config.ttc=14", "config.ep=15", "config.comfort=2.1",
    ]

    # Initialize Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Load Agent
    with initialize_config_dir(version_base=None, config_dir=str(agent_config_dir)):
        agent_cfg = compose(config_name=agent_name, overrides=agent_overrides)
    agent = instantiate(agent_cfg)
    agent.initialize()
    agent.b2d = agent.ray = False

    # Load Data
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = cfg.val_logs

    loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config()
    )

    dataset = Dataset(
        scene_loader=loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, **cfg.dataloader.params, drop_last=True)
    
    return agent, dataloader

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    agent, val_loader = load_components(REPO_ROOT, CHECKPOINT_PATH, AGENT_NAME, SPLIT)
    agent.cuda(local_rank).eval()

    # Storage for similarities
    cams = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]
    sum_sims = {cam: torch.zeros((16, 16), device="cuda", dtype=torch.float32) for cam in cams}
    count_local = torch.tensor([0.0], device="cuda")

    # --- Compute Loop ---
    with torch.no_grad():
        iterator = tqdm(val_loader, disable=(rank != 0), desc="Computing Similarity")
        for batch in iterator:
            features, _, _ = batch
            features = {k: v.cuda(local_rank, non_blocking=True) for k, v in features.items()}

            # Forward pass
            preds = agent.forward(features)
            
            # Extract Attention: (B_cam, Heads, N, N)
            attn_last = preds["image_backbone_attentions"][-1]
            B, H, N, _ = attn_last.shape
            
            # Setup Indices (Registers=16, Prefix=21)
            R, prefix = 16, 21
            patch_idx = torch.arange(prefix, N, device=attn_last.device)
            reg_idx = torch.arange(0, R, device=attn_last.device)

            # Average Heads and Slice Registers -> Patches
            A = attn_last.mean(1) # (B, N, N)
            A = A / (A.sum(-1, keepdim=True) + 1e-9)
            R2P = A[:, reg_idx][:, :, patch_idx] # (B, R, P)

            # Compute Cosine Similarity per Camera
            # Assumes batch B=4 corresponds strictly to [f0, b0, l0, r0] order in features
            for b, cam in enumerate(cams):
                rf = R2P[b].flatten(1) # (R, P)
                rf = rf / (rf.norm(dim=-1, keepdim=True) + 1e-9)
                sim = rf @ rf.T # (R, R)
                sum_sims[cam] += sim
            
            count_local += 1

            # Cleanup
            del preds, features
            # torch.cuda.empty_cache() # Optional: keep commented for speed unless OOM

    # --- Distributed Reduction ---
    if rank == 0:
        print("Aggregating results across GPUs...")
        
    count_global = count_local.clone()
    dist.all_reduce(count_global, op=dist.ReduceOp.SUM)

    mean_sims = {}
    for cam, sim_local in sum_sims.items():
        sim_global = sim_local.clone()
        dist.all_reduce(sim_global, op=dist.ReduceOp.SUM)
        mean_sims[cam] = (sim_global / count_global).cpu()

    # --- Saving & Plotting (Rank 0 Only) ---
    if rank == 0:
        # Save raw tensor
        torch.save(mean_sims, OUTPUT_DIR / f"mean_reg_reg_cosine_{SPLIT}.pt")
        
        # Plot
        for cam, sim in mean_sims.items():
            plt.figure(figsize=(5, 4))
            plt.imshow(sim, cmap='viridis', vmin=0, vmax=1)
            plt.title(f"{cam.upper()} — Mean Reg↔Reg Cosine")
            plt.colorbar(label="Cosine Similarity")
            plt.xlabel("Register Index j")
            plt.ylabel("Register Index i")
            plt.tight_layout()
            
            save_path = OUTPUT_DIR / f"reg_reg_cosine_{cam}_{SPLIT}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            
            # Log Metric
            off_diag_mean = sim.triu(1).mean().item()
            print(f"Saved {save_path.name} | Mean Off-Diagonal Sim: {off_diag_mean:.4f}")

        print(f"\nProcessing Complete. Total batches: {count_global.item():.0f}")

    dist.barrier()
    dist.destroy_process_group()