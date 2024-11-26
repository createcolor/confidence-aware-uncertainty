import torch
import sys
from pathlib import Path

import torch.utils
import torch.utils.data

from utils.dataset import LIDCDataset, RIGADataset
from utils.models.UNet import UNet
from utils.SGLD import SGLD
from utils.metrics.SVLS_2D import CELossWithSVLS_2D, CELossWithOH_2D

def get_loss(loss_fn: str, n_classes: int, device: torch.device='cpu') -> torch.nn.Module:
    if loss_fn == "CELossWithSVLS_2D":
        loss_fn = CELossWithSVLS_2D(classes=n_classes, device=device)
    elif loss_fn == "CELossWithOH_2D":
        loss_fn = CELossWithOH_2D(classes=n_classes)
    else:
        loss_fn = getattr(sys.modules["torch.nn"], loss_fn, None)
        if loss_fn is None:
            raise ValueError(f"Loss function {loss_fn} is not supported.")
        loss_fn = loss_fn()

    return loss_fn

def get_dataset(dataset: str, dir: Path, gt: str, params: dict) -> torch.utils.data.Dataset:
    if dataset == "LIDC":
        data = LIDCDataset(data_dir=dir, gt_mode=gt, **params)
    elif dataset == "RIGA":
        data = RIGADataset(data_dir=dir, ground_truth=gt, **params)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    
    return data

def get_architecture(architecture: str) -> torch.nn.Module:
    architectures = {'unet': UNet}
    if architecture not in architectures.keys():
        raise ValueError(f"Model architecture {architecture} is not supported.")
    return architectures[architecture]

def get_optimizer(optimizer: dict):
    optimizer_type = optimizer["type"]
    optimizer_params = optimizer["params"]
    if optimizer_type == "SGLD":
        optimizer = SGLD
    else:
        optimizer = getattr(sys.modules["torch.optim"], optimizer_type, None)
        if optimizer is None:
            raise ValueError(f"Optimizer {optimizer_type} is not supported.")
        
    return optimizer, optimizer_params

def get_scheduler(scheduler: str, last_epoch: int, params: dict):
    if scheduler == "StepLR":
        scheduler_type = torch.optim.lr_scheduler.StepLR
        scheduler_params = {
            "step_size": params["step_size"],
            "gamma": params["gamma"],
            "last_epoch": last_epoch
        }
    else:
        raise ValueError(f"Scheduler type {scheduler} is not supported.")

    return scheduler_type, scheduler_params

def flatten_labels(dataset: str, path: Path | str, expert_id: int) -> torch.Tensor:
    if dataset == "riga":
        params = {
            "data_dir": Path(path),
            "ground_truth": f"expert{expert_id}",
            "sets": ["BinRushed"],
            "augment": False,
        }
        ds = RIGADataset(**params)

    elif dataset == "lidc":
        params = {
            "data_dir": Path(path),
            "gt_mode": f"expert{expert_id}",
        }
        ds = LIDCDataset(**params)

    labels = []

    for i in range(ds.__len__()):
        _, label = ds.__getitem__(i)
        labels.append(label)

    labels = torch.stack(labels)
    labels = torch.flatten(labels, start_dim=-2).long()
    return labels