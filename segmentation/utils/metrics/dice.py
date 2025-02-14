import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceScore(nn.Module):
    """
    Class for evaluating Dice score between single-class predictions and targets.

    Args:
        use_sigmoid (bool): whether to apply sigmoid to predictions or not.
        binarize (bool): whether to binarize predictions via a 0.5 threshold.
        smooth (float): smoothing parameter in division.
    """
    def __init__(self, use_sigmoid: bool = True, binarize: bool = True, smooth: float = 1.) -> None:
        super(DiceScore, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.binarize = binarize
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Returns a Dice score between predictions and targets:

        Dice = 2 * |Intersection| / (|Predictions| + |Targets|).

        Args:
            inputs (torch.Tensor): preditions.
            targets (torch.Tensor): targets.

        Returns:
            float: Dice score.
        """
        if self.use_sigmoid:
            inputs = torch.sigmoid(inputs)
        if self.binarize:
            inputs = inputs >= 0.5

        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return dice


class MulticlassDice(nn.Module):
    """
    Class for evaluating Dice score between multi-class predictions and targets.
    For that, a DiceScore instance is called on each class.

    Args:
        n_classes (int): number of classes.
        reduction (str or None): if "mean", returns average Dice score over all classes;
                                if None, returns a list of Dice scores.
        onehot_labels (bool): whether to onehot-encode labels by class.
        use_softmax (bool): whether to apply softmax to predictions.
        binarize (bool): whether to onehot-encode predictions by class.
        smooth (float): smoothing parameter in division.
    """
    def __init__(self, n_classes: int | None = None, reduction: str | None = "mean",
                 onehot_labels: bool = True, use_softmax: bool = True,
                 binarize: bool = True, smooth: float = 1.):
        super(MulticlassDice, self).__init__()
        self.classes = n_classes
        self.reduction = reduction
        self.onehot = onehot_labels
        self.softmax = use_softmax
        self.binarize = binarize
        self.criteria = DiceScore(use_sigmoid=False, binarize=False, smooth=smooth)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float | torch.Tensor:
        """
        Calculates Dice score between predictions and targets for multiple classes.

        Args:
            inputs (torch.Tensor): preditions.
            targets (torch.Tensor): targets.

        Returns:
            float or torch.Tensor: average Dice score or a list of Dice scores per class.
        """
        n_classes = inputs.shape[-1] if self.classes is None else self.classes

        if self.softmax:
            inputs = F.softmax(inputs, dim=1)

        if self.binarize:
            inputs = F.one_hot(torch.argmax(inputs, dim=1), num_classes=n_classes)  # pylint: disable=not-callable
            inputs = inputs.movedim(-1, -3)

        if self.onehot:
            targets = F.one_hot(targets, num_classes=n_classes)  # pylint: disable=not-callable
            targets = targets.movedim(-1, -3)

        dice = []
        for c in range(n_classes):
            dice.append(self.criteria(inputs[..., c, :, :], targets[..., c, :, :]))

        if self.reduction == "mean":
            return torch.mean(torch.tensor(dice))
        elif self.reduction is None:
            return dice


def stacked_Dice(preds: torch.Tensor, labels: torch.Tensor,
                 binarize: bool = True, smooth: float = 1.) -> torch.Tensor:
    """
    Calculates Dice score for multiple items, treating each element in the first tensor
    dim as a separate item.

    Args:
        inputs (torch.Tensor): preditions.
        targets (torch.Tensor): targets.

    Returns:
        torch.Tensor: one-dimensional tensor containing Dice scores for multiple items.
    """
    if binarize:
        preds = preds > 0.5

    intersection = (preds * labels).sum(dim=1)
    return (2. * intersection + smooth) / (preds.sum(dim=1) + labels.sum(dim=1) + smooth)
