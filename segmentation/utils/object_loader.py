import sys
from pathlib import Path
import torch
import torch.utils
import torch.utils.data

from utils.dataset import LIDCDataset, RIGADataset
from utils.models.UNet import UNet
from utils.SGLD import SGLD
from utils.metrics.SVLS_2D import CELossWithSVLS_2D, CELossWithOH_2D

def get_loss(loss_fn: str, n_classes: int, device: torch.device='cpu') -> torch.nn.Module:
    """
    Returns a loss function object given its name. If the desired loss is not one of 
    the pre-written classes, this function looks for a potential match in the torch.nn module.

    Args:
        loss_fn (str): name of a loss function.
        n_classes (int): number of predicted classes.
        device (torch.device or str): device to use for computation.

    Returns:
        torch.nn.Module: a loss function as a PyTorch class object.
    """
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

def get_dataset(dataset: str, ds_dir: Path, gt: str, params: dict) -> torch.utils.data.DataLoader:
    """
    Returns a dataset class object given its name and parameters. 

    Args:
        dataset (str): name of a dataset.
        ds_dir (Path): path to the directory of the data.
        gt (str): how to generate ground truth; available values are dataset-specific.
        params (dict): dictionary containing relevant dataset parameters.

    Returns:
        torch.utils.data.Dataset: a dataset as a PyTorch class object.
    """
    if dataset == "LIDC":
        data = LIDCDataset(data_dir=ds_dir, gt_mode=gt, **params)
    elif dataset == "RIGA":
        data = RIGADataset(data_dir=ds_dir, ground_truth=gt, **params)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    
    return data

def get_architecture(architecture: str) -> type[torch.nn.Module]:
    """
    Returns an architecture class given its name. 

    Args:
        architecture (str): name of an architecture.

    Returns:
        type[torch.nn.Module]: an architecture as a PyTorch class.
    """
    architectures = {'unet': UNet}
    if architecture not in architectures:
        raise ValueError(f"Model architecture {architecture} is not supported.")
    return architectures[architecture]

def get_optimizer(optimizer: dict) -> tuple[type[torch.optim.Optimizer], dict]:
    """
    Returns an optimizer class and parameters given its configuration dictionary.
    If the desired optimizer is not one of the pre-written classes, 
    this function looks for a potential match in the torch.optim module. 

    Args:
        optimizer (dict): configuration dictionary of an optimizer 
                          containing its name and parameters.

    Returns:
        type[torch.optim.Optimizer]: an optimizer as a PyTorch class.
        dict: a dictionary of parameters to supply to the optimizer.
    """
    optimizer_type = optimizer["type"]
    optimizer_params = optimizer["params"]
    if optimizer_type == "SGLD":
        optimizer = SGLD
    else:
        optimizer = getattr(sys.modules["torch.optim"], optimizer_type, None)
        if optimizer is None:
            raise ValueError(f"Optimizer {optimizer_type} is not supported.")
        
    return optimizer, optimizer_params

def get_scheduler(scheduler: str, last_epoch: int, 
                  params: dict) -> tuple[type[torch.optim.lr_scheduler._LRScheduler], dict]:
    """
    Returns a scheduler class and parameters given its 
    name, last epoch and a dictionary of parameters. 

    Args:
        scheduler (str): name of a scheduler.
        last_epoch (int): number of epochs the model was previously trained for;
                          used to restart training.
        params (dict): dictionary of a scheduler parameters.

    Returns:
        type[torch.optim.lr_scheduler._LRScheduler]: a scheduler as a PyTorch class.
        dict: a dictionary of parameters to supply to the scheduler.
    """
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
    """
    Loads labels from a dataset and flattens last two dimensions.

    Args:
        dataset (str): name of a dataset.
        path (Path or str): path to the directory of the data.
        expert_id (int): assessments of which expert to load.

    Returns:
        torch.Tensor: a tensor of labels shaped (num. samples, height * width).
    """
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

    for _, label in ds:
        labels.append(label)

    labels = torch.stack(labels)
    labels = torch.flatten(labels, start_dim=-2).long()
    return labels