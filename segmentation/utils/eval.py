from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def evaluate_model(model_dir: Path, architecture: torch.nn.Module,
                   nets: int | list, epochs: int,
                   metric: torch.nn.Module, data_loader: DataLoader,
                   device: torch.device | str = 'cpu') -> tuple[list, dict]:
    """
    Evaluates an ensemble of models via the given metric.

    Args:
        model_dir (Path): path to the directory where model checkpoints are stored.
        architecture (torch.nn.Module): model architecture as a PyTorch class object.
        nets (int or list): number of networks from the ensemble to inference (int)
                            or their indices (list).
        epochs (int): how long the model was trained.
        metric (torch.nn.Module): metric to evaluate the model as a PyTorch class object.
        data_loader (torch.utils.data.DataLoader): dataloader of the test data.
        device (torch.device or str): device to use for computation.

    Returns:
        list: list of model predictions as PyTorch tensors.
        dict: dictionary containing pairs of an image identifier and Dice scores
              of the models on that image.
    """
    predictions = []
    scores = {}

    if isinstance(nets, int):
        nets = [i for i in range(nets)]

    for idx in nets:
        model_path = model_dir / (model_dir.name + f"_{idx}_{epochs}ep")
        model = architecture.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        model_predictions = []
        scores[idx] = {}

        with torch.no_grad():
            for (X_test, Y_test, key) in tqdm(data_loader):
                Y_pred = model(X_test.to(device)).cpu()
                model_predictions.append(Y_pred.squeeze())
                scores[idx][key[0]] = np.array(metric(Y_pred, Y_test))

        predictions.append(torch.stack(model_predictions))
        model_score = np.mean(list(scores[idx].values()), axis=0)
        print(f"Model {model_path.name} Dice score(s): {model_score}")

    return predictions, scores


def evaluate_abnn(model_dir: Path, architecture: torch.nn.Module,
                  nets: list | int, epochs: int,
                  metric: torch.nn.Module, data_loader: DataLoader,
                  samples: int = 10, device: torch.device | str = 'cpu') -> tuple[list, dict]:
    """
    Evaluates an ABNN ensemble of models via the given metric.
    Predictions from each model are sampled multiple times, and the samples are averaged.

    Args:
        model_dir (Path): path to the directory where model checkpoints are stored.
        architecture (torch.nn.Module): model architecture as a PyTorch class object.
        nets (int or list): number of networks from the ensemble to inference (int)
                            or their indices (list).
        epochs (int): how long the model was trained.
        metric (torch.nn.Module): metric to evaluate the model as a PyTorch class object.
        data_loader (torch.utils.data.DataLoader): dataloader of the test data.
        samples (int): how many times to sample predictions from each model.
        device (torch.device or str): device to use for computation.

    Returns:
        list: list of model predictions as PyTorch tensors.
        dict: dictionary containing pairs of an image identifier and Dice scores
              of the models on that image.
    """
    predictions = []
    scores = {}

    if isinstance(nets, int):
        nets = [i for i in range(nets)]

    for idx in nets:
        model_path = model_dir / (model_dir.name + f"_{idx}_{epochs}ep")
        model = architecture.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        model_predictions = []
        scores[idx] = {}

        with torch.no_grad():
            for (X_test, Y_test, key) in tqdm(data_loader):
                X_test = X_test.to(device)
                Y_pred = torch.stack([model(X_test).cpu() for _ in range(samples)])
                model_predictions.append(Y_pred.squeeze())
                scores[idx][key[0]] = np.array(metric(Y_pred.mean(dim=0), Y_test))

        predictions.append(torch.stack(model_predictions))
        model_score = np.mean(list(scores[idx].values()))
        print(f"Model {model_path.name} Dice score: {model_score:.4f}")

    return predictions, scores
