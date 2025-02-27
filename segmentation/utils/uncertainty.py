from pathlib import Path
import torch


def load_mcmc(path: Path | str, step_size: int, total_epochs: int) -> torch.Tensor:
    """
    Loads predictions of an MCMC model taken with checkpoints from a directory.

    Args:
        path (Path | str): path to the directory containing the predictions.
        step_size (int): size of the step with which the checkpoints were taken.
        total_epochs (int): number of epochs the model was trained for.

    Returns:
        torch.Tensor: a tensor of shape (num. nets, num. checkpoints, num. samples,
                                         num. classes, height, width)
    """
    epoch_preds = []
    for step in range(step_size, total_epochs + 1, step_size):
        epoch_ens = torch.load(Path(path) / f"{step}_epochs" / "predictions.pt")
        epoch_preds.append(torch.softmax(epoch_ens, dim=-3))

    epoch_preds = torch.movedim(torch.stack(epoch_preds), 0, 1)
    return epoch_preds


def calculate_uncertainty(preds: torch.Tensor, method: str | None = "var") -> torch.Tensor:
    """
    Calculates uncertainty of predictions.

    If the chosen method is "var", then the uncertainty is the variance of predictions along
    the first dimension:

        U = 1/K * sum [(pred - pred.mean) ** 2]

    If the chosen method is "bern_var", then the uncertainty for each net is the variance of
    its predictions assuming Bernoulli distribution, and the total uncertainty is the mean
    uncertainty per net:

        U = 1/K * sum [pred * (1 - pred)]

    If the chosen method is None, returns a zero uncertainty map. This behavior is needed for
    the tv_uncertainty function.

    Args:
        preds (torch.Tensor): a tensor of predictions where the first dimension will be used for
                              uncertainty calculation.
        method (str | None): either "var", "bern_var" or None.

    Returns:
        torch.Tensor: tensor containing the uncertainty map of predictions.
    """
    if method == 'var':
        return preds.var(dim=0, correction=0)
    elif method == 'bern_var':
        return (preds * (1. - preds)).mean(dim=0)
    elif method is None:
        return torch.zeros_like(preds.mean(dim=0))
    else:
        raise ValueError(f"Method {method} is not supported.")


def tv_uncertainty(preds: torch.Tensor | None,
                   expert_preds: torch.Tensor | None = None,
                   method: str | None = 'var',
                   expert_method: str | None = None) -> torch.Tensor:
    """
    Calculates uncertainty from two sources as given by law of total variance.
    The epistemic component is calculated from preds and the aleatoric component is
    computred from expert_preds. See calculate_uncertainty function for reference.

    It is possible to compute only epistemic or only aleatoric uncertainty by supplying
    method=None for the other component. At least one method has to be not None.

    Args:
        preds (torch.Tensor | None): predictions to calculate epistemic uncertainty from.
        expert_preds (torch.Tensor | None): predictions to calculate aleatoric uncertainty from.
        method (str | None): method by which the epistmic uncertainty is to be calculated.
        expert_method (str | None): method by which the aleatoric uncertainty is to be calculated.

    Returns:
        torch.Tensor: tensor containing the uncertainty map of predictions.
    """
    if method in ('none', 'None'):
        method = None
    if expert_method in ('none', 'None'):
        expert_method = None

    assert (preds is not None) or (expert_preds is not None), \
        "Either preds or expert_preds has to be not None!"
    assert (method is None) or (preds is not None), \
        "You provided method but no predictions."
    assert (expert_method is None) or (expert_preds is not None), \
        "You provided expert_method but no expert_predictions."

    if preds is not None:
        unc_ens = calculate_uncertainty(preds, method)
    else:
        # Create a zero map for epistemic uncertainty if no preds provided.
        unc_ens = calculate_uncertainty(expert_preds, None)

    if (expert_preds is not None) and (expert_method is not None):
        unc_exp = calculate_uncertainty(expert_preds, expert_method)
    else:
        # Create a zero map for aleatoric uncertainty if no expert_preds provided.
        unc_exp = calculate_uncertainty(preds, None)

    return unc_ens + unc_exp


def multiclass_uncertainty(n_classes: int, preds: torch.Tensor | None,
                           expert_preds: torch.Tensor | None = None,
                           method: str | None = 'var',
                           expert_method: str | None = None) -> torch.Tensor:
    """
    Calculate agregate uncetainty from multiple classes. For this, multiple instances of
    tv_uncertainty are called.

    Args:
        n_classes (int): total number of classes.
        preds (torch.Tensor): predictions to calculate epistemic uncertainty from.
        expert_preds (torch.Tensor | None): predictions to calculate aleatoric uncertainty from.
        method (str | None): method by which the epistmic uncertainty is to be calculated.
        expert_method (str | None): method by which the aleatoric uncertainty is to be calculated.

    Returns:
        torch.Tensor: tensor containing the uncertainty map of predictions
            combined from multiple classes.
    """
    if n_classes == 1:
        class_preds = None if preds is None else preds.float()
        class_exps = None if expert_preds is None else expert_preds.float()
        return tv_uncertainty(class_preds, class_exps, method=method, expert_method=expert_method)

    uncs = []
    for i in range(n_classes):
        class_preds = None if preds is None else preds[..., i, :].float()
        class_exps = None if expert_preds is None else expert_preds[..., i, :].float()
        uncs.append(tv_uncertainty(class_preds, class_exps, method=method,
                                   expert_method=expert_method))

    return torch.sum(torch.stack(uncs), dim=0)
