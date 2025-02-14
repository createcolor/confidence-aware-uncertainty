import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.uncertainty import load_mcmc, multiclass_uncertainty  # pylint: disable=import-error
from utils.metrics.dice import stacked_Dice  # pylint: disable=import-error
from utils.object_loader import flatten_labels  # pylint: disable=import-error


def onehot_argmax(t: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Find the maximum probability prediction, then one-hot-encode the tensor.

    Args:
        t (torch.Tensor): predictions of shape (..., num_classes, height * width).
        num_classes (int): number of classes.

    Returns:
        torch.Tensor: one-hot-encoded tensor of the same shape as t.
    """
    t = torch.argmax(t.detach(), dim=-2)
    t = F.one_hot(t, num_classes=num_classes)  # pylint: disable=not-callable
    return t.movedim(-1, -2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=Path,
                        default="configs/rejection_curve.json",
                        help="Path to the config file.")
    parser.add_argument('-f', '--final_activation', type=bool, default=False,
                        help="Flag to use sigmoid/softmax for expert predictions.")
    parser.add_argument('-s', '--save_path', type=Path, required=True,
                        help="Path to save uncertainties.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # labels = torch.load(open(config["labels_path"], 'rb')).to(device)
    labels = flatten_labels(config["dataset"], config["data_path"], config["expert_id"]).to(device)

    models = config["models_ids"]
    expert_models = config["expert_ids"]

    if isinstance(models, int):
        models = [i for i in range(models)]
    if isinstance(expert_models, int):
        expert_models = [i for i in range(expert_models)]

    print("Loading predictions...")
    models_predictions = F.softmax(torch.load(config["models_predictions"]), dim=2).to(device)
    models_predictions = torch.reshape(models_predictions, (models_predictions.shape[:-2]) + (-1,))

    n_classes = models_predictions.shape[-2]
    labels = F.one_hot(labels, num_classes=n_classes).movedim(-1, -2)  # pylint: disable=not-callable
    models_predictions = models_predictions[models]

    print("Calculating uncertainties...")
    expert_path = config.get("expert_predictions", None)
    if expert_path is not None:
        expert_path = Path(expert_path)
        method = config.get("expert_method", None)

        if method == "mcmc":
            expert_predictions = load_mcmc(expert_path, config["step_size"], config["total_epochs"])
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            net_uncs = []
            for model in expert_models:
                uncs = multiclass_uncertainty(n_classes,
                                              preds=None, expert_preds=expert_predictions[model],
                                              method=None, expert_method="var")
                net_uncs.append(uncs)
            slide_uncs = torch.stack(net_uncs)

        elif method == "abnn":
            expert_predictions = F.softmax(torch.load(expert_path), dim=3).to(device)
            expert_predictions = expert_predictions[expert_models]
            expert_predictions = torch.movedim(torch.flatten(expert_predictions,
                                                             start_dim=-2), 2, 0)

            slide_uncs = multiclass_uncertainty(n_classes,
                                                preds=None, expert_preds=expert_predictions,
                                                method=None, expert_method="var")

        else:
            print("Using final activation." if args.final_activation
                  else "Not using final activation.")
            expert_predictions = torch.load(config["expert_predictions"]).to(device)
            expert_predictions = expert_predictions[expert_models]
            if args.final_activation:
                expert_predictions = F.softmax(expert_predictions, dim=2)
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            slide_uncs = multiclass_uncertainty(n_classes, preds=models_predictions,
                                                expert_preds=expert_predictions,
                                                method=config.get("models_method", None),
                                                expert_method=method)
            slide_uncs = torch.stack([slide_uncs] * 10)
    else:
        slide_uncs = multiclass_uncertainty(n_classes=n_classes,
                                            preds=models_predictions, expert_preds=None,
                                            method=config.get("models_method", None),
                                            expert_method=None)
        slide_uncs = torch.stack([slide_uncs] * 10)

    print(slide_uncs.shape)

    models_predictions = onehot_argmax(models_predictions, num_classes=n_classes)

    sort_indices = np.argsort(slide_uncs.cpu().numpy(), axis=2)

    throwaway_rates = np.concatenate((np.linspace(0.0001, 0.1, 1000), np.linspace(0.101, 1.0, 900)))
    avg_dice = []

    print("Calculating pixel-wise rejection curves:")
    with torch.no_grad():
        for cls in range(1, n_classes):
            dice_per_class = []

            for idx in range(len(models)):
                predictions_sorted = np.take_along_axis(models_predictions[idx, :, cls],
                                                        sort_indices[idx], axis=1)
                labels_sorted = np.take_along_axis(labels[..., cls, :], sort_indices[idx], axis=1)
                model_dice = []

                zero_score = stacked_Dice(predictions_sorted, labels_sorted, binarize=False)
                print(f"Class {cls}, net {idx}:", zero_score.mean())
                model_dice.append(zero_score.mean().cpu().item())

                for rate in tqdm(throwaway_rates):
                    throwaway_num = round(rate * predictions_sorted.shape[-1])
                    score = stacked_Dice(predictions_sorted[:, :-throwaway_num],
                                         labels_sorted[:, :-throwaway_num], binarize=False)
                    model_dice.append(score.mean().cpu().item())
                dice_per_class.append(np.array(model_dice))
            avg_dice.append(dice_per_class)

    if args.save_path is not None:
        print("Saving:")
        if not args.save_path.exists():
            os.makedirs(args.save_path)

        throwaway_rates = np.insert(throwaway_rates, 0, 0.)
        thresholds = [str(thr) for thr in throwaway_rates]
        for cls in range(1, n_classes):

            class_dice = np.stack(avg_dice[cls - 1]).mean(axis=0)
            print(f"Class {cls}:", class_dice)
            plt.plot(throwaway_rates, class_dice)

            class_dice = [str(score) for score in class_dice]
            rejection_curve = {rate: score for (rate, score) in zip(thresholds, class_dice)}
            with open(args.save_path / f"rejection_curve_{cls}.json", 'w', encoding='utf-8') as f:
                json.dump(rejection_curve, f, ensure_ascii=False, indent=4)

        plt.grid()
        plt.savefig(args.save_path / "rejection_curve.png", format='png', bbox_inches='tight')
        plt.close()
