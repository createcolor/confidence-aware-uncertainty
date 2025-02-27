import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.eval import evaluate_model  # pylint: disable=import-error
from utils.object_loader import get_architecture, get_dataset  # pylint: disable=import-error
from utils.metrics.dice import DiceScore, MulticlassDice  # pylint: disable=import-error


def parse_args():
    parser = argparse.ArgumentParser('Test an MCMC segmentator.')

    parser.add_argument('-c', '--config', type=Path,
                        default=Path("configs/test_mcmc.json"),
                        help="Path to test config.")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    device = torch.device(config.get("device", 'cpu') if torch.cuda.is_available() else 'cpu')
    n_channels, n_classes = config["n_channels"], config["n_classes"]

    # Load data;
    print("Loading data...")
    ds_dict = config["dataset"]
    ds = get_dataset(ds_dict["type"], ds_dir=ds_dict["data_path"],
                     gt=ds_dict["gt_mode"], params=ds_dict["params"])
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # Create architecture instance;
    model_dir = Path(config["models_path"]) / config["model"]
    architecture = get_architecture(architecture=config["architecture"])
    architecture = architecture(n_channels=config["n_channels"], n_classes=config["n_classes"])

    metric = DiceScore() if n_classes == 1 else MulticlassDice(n_classes=n_classes, reduction=None)

    for step in range(config["checkpoint"], config["epochs"] + 1, config["checkpoint"]):
        predictions, scores = evaluate_model(model_dir=model_dir,
                                             architecture=architecture,
                                             nets=config["nets"],
                                             metric=metric,
                                             epochs=step,
                                             data_loader=dl,
                                             device=device)

        save_path = Path(config["save_path"]) / f"{step}_epochs"

        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.exists():
                os.makedirs(save_path)

            predictions = torch.stack(predictions)
            print("Model predictions shape:", predictions.shape)
            with open(save_path / "predictions.pt", 'wb') as f:
                torch.save(predictions, f)

            str_scores = {str(model): {key: str(value) for (key, value) in values.items()}
                          for (model, values) in scores.items()}
            with open(save_path / "Dice.json", 'w', encoding='utf-8') as f:
                json.dump(str_scores, f, ensure_ascii=False, indent=4)

        scores = np.array([np.mean(list(model_scores.values()))
                           for model_scores in scores.values()])
        print(f"Average Dice score: {scores.mean():.4f} \u00B1 {scores.std():.4f}")
