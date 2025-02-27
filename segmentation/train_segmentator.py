import os
import json
import argparse
from pathlib import Path
import torch
import torch.utils
from torch.utils.data import DataLoader, random_split
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.train import train  # pylint: disable=import-error
from utils.object_loader import (get_architecture, get_dataset,  # pylint: disable=import-error
                                 get_loss, get_optimizer, get_scheduler)
from utils.metrics.dice import DiceScore, MulticlassDice  # pylint: disable=import-error


def parse_args():
    parser = argparse.ArgumentParser('Train a segmentator.')

    parser.add_argument('-c', '--config', type=Path,
                        help="Path to train config .json file.",
                        default="segmentation/configs/train_ensemble.json")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = parse_args()

    # Load config;
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Set device;
    device = torch.device(config.get("device_type", 'cpu') if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")

    loss_fn = get_loss(config["loss_func"], config["n_classes"], device)

    # Load data;
    print("Loading data...")
    ds_dict = config["dataset"]
    data = get_dataset(ds_dict["type"], ds_dir=ds_dict["data_path"],
                       gt=ds_dict["gt_mode"], params=ds_dict["params"])

    # Create save directories;
    save_dir = Path(config["net_dir_path"], config["net_dir_name"])
    logdir = save_dir / "plots"
    if not save_dir.exists():
        os.makedirs(save_dir)
    if not logdir.exists():
        os.makedirs(logdir)

    # Save a copy of config;
    with open(str(save_dir / f"{config.get('net_dir_name')}_config.json"),
              'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # This prepares lists for the ensemble;
    nets_number = config["nets_number"]
    models, optimizers, schedulers = [], [], []
    train_dls, val_dls, seeds = [], [], []
    writers = []

    # Load generator seeds for random split;
    # This is to make sure that initial training and fine-tuning
    # are done on the same data split;
    if not config.get("learn_from_scratch", True):
        load_dir = Path(config["learn_from_model"])
        with open(load_dir / "seeds", 'r', encoding='utf-8') as f:
            load_seeds = json.load(f)["seeds"]

    print("Initializing models:")
    n_channels = config["n_channels"]
    n_classes = config["n_classes"]

    # Select a model architecture;
    architecture = get_architecture(architecture=config["architecture"])

    # Select an optimizer;
    optimizer_class, optimizer_params = get_optimizer(config["optimizer"])

    # Gather scheduler parameters
    scheduler_type = None
    scheduler_params = {}
    scheduler_config = config.get("lr_scheduler", None)
    if scheduler_config is not None:
        last_epoch = config.get("pretrained_epochs", -1) if config.get("use_previous_lr",
                                                                       False) else -1
        scheduler_type, scheduler_params = get_scheduler(scheduler_config["type"], last_epoch,
                                                         scheduler_config["parameters"])

    for idx in tqdm(range(nets_number)):
        # Create model instance
        model = architecture(n_channels=n_channels, n_classes=n_classes).to(device)

        # If resuming training, load model parameters;
        if not config.get("learn_from_scratch", True):
            load_model = load_dir / (load_dir.name + f"_{idx}_{config['pretrained_epochs']}ep")
            model.load_state_dict(torch.load(load_model, map_location=device))

        # Create optimizer instance
        optimizer = optimizer_class(params=model.parameters(), **optimizer_params)

        # Create scheduler instance
        if scheduler_type is not None:
            scheduler = scheduler_type(optimizer=optimizer, **scheduler_params)
        else:
            scheduler = None

        # Seed the generator to create an appropriate train/val split;
        gen = torch.Generator()
        if not config.get("learn_from_scratch", True):
            gen.manual_seed(load_seeds[idx])
        else:
            gen.seed()
        seeds.append(gen.initial_seed())

        # Splir the data;
        train_data, val_data = random_split(data, [config["train_part"],
                                                   1 - config["train_part"]], gen)
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

        # Create a tensorboard summary writer
        writer = SummaryWriter(logdir=logdir / str(idx))

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        train_dls.append(train_loader)
        val_dls.append(val_loader)
        writers.append(writer)

    # Save a copy of generator seeds for future use;
    with open(save_dir / "seeds", 'w', encoding='utf-8') as f:
        json.dump({"seeds": seeds}, f)

    # Set training length and checkpoints;
    epochs_total = config["epochs_total"]
    checkpoint_step = config.get("checkpoint_step", epochs_total)
    validation_step = config.get("validation_step", 1)

    onehot_labels = False if config["dataset"]["gt_mode"] == "mOH" else True
    metric = DiceScore() if n_classes == 1 else MulticlassDice(n_classes=n_classes,
                                                               onehot_labels=onehot_labels)

    # Initiate training for a length of checkpoint_step, then save a switch to another network;
    # Do until each network is trained for a total number of epochs;
    for checkpoint in range(0, epochs_total - checkpoint_step + 1, checkpoint_step):
        for idx in range(nets_number):
            save_path = save_dir / (config["net_dir_name"] +
                                    f"_{idx}_{checkpoint + checkpoint_step}ep")
            print(f"Training model {idx} " +
                  f"from epoch {checkpoint} to {checkpoint + checkpoint_step}.")
            train(model=models[idx],
                  optimizer=optimizers[idx],
                  scheduler=schedulers[idx],
                  loss_fn=loss_fn,
                  metric=metric,
                  epochs=checkpoint_step,
                  pretrained_epochs=checkpoint,
                  data_tr=train_dls[idx],
                  data_val=val_dls[idx],
                  device=device,
                  writer=writers[idx],
                  save_path=save_path,
                  validation_step=validation_step)

    for writer in writers:
        writer.close()
