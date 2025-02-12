import torch
import torch.optim as optim
from pathlib import Path
import argparse
import json
import sys
from nn.train import train
from nn.get_training_info import get_train_val_markups, get_loaders, get_gt_std_nums_per_bin

import EfficientNet, MobileNet, DenseNet
from nn_utils import mkdirs, load_markup, get_net_architecture
from tqdm import tqdm 
import cv2
import shutil
from nn.SGLD_optimizer import SGLD
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        'Usage example: python nn/train_nn.py -n MobileNet_10_10_80_lm -c nn/net_configs/MobileNet_10_10_80/MobileNet_1050ep_try5.json -m markup/train_dataset_10_10_80.json')
    
    parser.add_argument('-r', '--dataset_dir', default=Path(
        "/mnt/nuberg/datasets/medical_data/alvs_dataset_all"), type=Path, help='Path to images')
    parser.add_argument('-n', '--net_name', required=True, type=str, help='Net name')
    parser.add_argument('-m', '--markup', default=Path("markup/train_dataset_60_20_20.json"), \
        type=Path, help='Path to train dataset markup')
    parser.add_argument('-em', '--experts_markup', default=Path("markup/experts_info_per_alvs.json"), \
        type=Path, help='Path to experts markup')
    parser.add_argument('-c', '--net_config', required=True, type=Path, \
        help='The nets config. For exmaple, nn/net_configs/tmp/tmp.json')

    args = parser.parse_args()

    with open(args.net_config, "r") as config_file:
        config = json.load(config_file)

    args.experts_markup = load_markup(args.experts_markup)

    if config["learn_from_scratch"]:
        assert config["pretrained epochs number"] == 0
    assert config["epochs_step"] <= config["epochs_total"]

    return args, config, config["architecture"], config["epochs_total"], config["nets_number"], \
        config["loss_func"], config["epochs_step"]


def get_net_optimizer_lrshed(net, net_id, trained_ep_num, epoch_num2train, epochs_step, 
    config, args, net_dir, device):
    '''
    Loads or initializes the net, the optimizer and the learning rate sheduler
    '''
    if epoch_num2train == epochs_step: # at the start of the training loop
        if not config["learn_from_scratch"]:
            assert "take_previous_lr" in config
            assert config['net_name2learn_from'] is not None, \
                "The name of the net to learn from is not provided"
            net_dir2learn_from = config['net_name2learn_from'][:config['net_name2learn_from'].rfind("_")]
            net_path_saved = f"{config['path2nets_dir']}/{net_dir2learn_from}/{config['net_name2learn_from']}_{net_id}"
            
            net.load_state_dict(torch.load(net_path_saved, map_location=device), strict=False)

            path2copy_config = net_dir / f'config_{args.net_name}.json'
            shutil.copyfile(f"{config['path2nets_dir']}/{net_dir2learn_from}/config_{net_dir2learn_from}.json", path2copy_config)
            with open(path2copy_config, "r") as config_file:
                lr_config = json.load(config_file)
            if "take_previous_lr" in config and not config["take_previous_lr"]:
                lr = config["learning rate"]
                print(f"Starting from lr={lr}.")
            else:
                lr = lr_config["last learning rates"][str(trained_ep_num)]
        else:
            reagents_num = 13 if "meta" in config["architecture"] else 0
            net = get_net_architecture(net_arch_name)(reagents_num=reagents_num)
            net.to(device)
            lr = config["learning rate"]

    else:
        net_path_saved = net_dir / (args.net_name + f"_{trained_ep_num}ep_{net_id}")
        net.load_state_dict(torch.load(net_path_saved, map_location=device))
        
        with open(net_dir / f'config_{args.net_name}.json', "r") as config_file:
            lr_config = json.load(config_file)
        lr = lr_config["last learning rates"][str(trained_ep_num)]

    optimizer = SGLD(params=net.parameters(), lr=config["learning rate"])
    
    if config["use_lr_scheduler"]:
        if "StepLR" in config["lr_scheduler"]:
            epochs_num_when_change = config["lr_scheduler"]["StepLR"]["epochs_num_when_change"]

            epoch_id = 0
            assert 0 in epochs_num_when_change
            for epoch_id in range(len(epochs_num_when_change) - 1):
                if trained_ep_num >= epochs_num_when_change[epoch_id] and \
                   trained_ep_num < epochs_num_when_change[epoch_id + 1]:
                    break
            gamma = config["lr_scheduler"]["StepLR"]["gammas"][epoch_id]
            print(f"LR_shed: gamma is equal to {gamma} after {trained_ep_num} epochs.")
            
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=config["lr_scheduler"]["StepLR"]["step_size"], gamma=gamma)
            
        elif "ExponentialLR" in config["lr_scheduler"]:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                gamma=config["lr_scheduler"]["ExponentialLR"]["gamma"])
    
    return net, optimizer, lr_scheduler


def overwriting_check(nets_number, args, epochs, config, epochs_step):
    tmp_config = config.copy()

    nets_pathes = []
    for _ in range(epochs_step, max(epochs, epochs // epochs_step) + 1, epochs_step):
        if "pretrained epochs number" in tmp_config:
            trained_ep_num = tmp_config["pretrained epochs number"]
        else:
            trained_ep_num = 0
        tmp_config["pretrained epochs number"] = trained_ep_num + epochs_step
        
        for net_id in range(nets_number):
            nets_pathes.append(args.net_name + f"_{trained_ep_num + epochs_step}ep_{net_id}")
    
    nets2be_overwritten = []
    for net_path in nets_pathes:
        if (net_dir / net_path).exists():
            nets2be_overwritten.append(net_path)
    
    if len(nets2be_overwritten) > 0:
        want2overwrite = input(f"You are going to OVERWRITE the following existing nets: \n{nets2be_overwritten}" + \
            "\nDo you want to continue (Y/N): ")
        if want2overwrite != 'Y':
            print("Please choose other nets names.")
            exit()


def save_test_config(args, config):
    test_config_keys = ["architecture", "path2dataset"]
    test_config = {}
    for key in test_config_keys:
        test_config[key] = config[key]
    test_config["path2nets_dir"] = f'{config["path2nets_dir"]}/{config["net_dir_name"]}'

    with open(net_dir / f'test_config_{args.net_name}.json', "w") as outfile:
        json.dump(test_config, outfile, indent=4)

if __name__ == "__main__":
    args, config, net_arch_name, epochs, nets_number, \
        loss_func_name, epochs_step = parse_args()

    net_dir = Path(config["path2nets_dir"]) / config["net_dir_name"]
    log_dir = net_dir / "plots" 
    mkdirs(net_dir, log_dir)
    print(f"Nets are going to be saved into folder: {net_dir}")
    meta_learning_mode = "meta" in config["architecture"] 
    reagents_num = 13 if meta_learning_mode else 0

    net = get_net_architecture(net_arch_name)(reagents_num=reagents_num)
    device = torch.device(config["device_type"] if torch.cuda.is_available() else "cpu")
    net.to(device)

    loss_func = getattr(sys.modules["torch.nn"], loss_func_name)
    if "label_smoothing" in config and config["label_smoothing"] is not None:
        loss_func = loss_func(label_smoothing=config["label_smoothing"])

    markup_trains, markup_vals = get_train_val_markups(
        nets_number, config, args.net_name, args.markup, net_dir, train_part=0.75)

    overwriting_check(nets_number, args, epochs, config, epochs_step)

    print("Loading images: ")
    images = {}
    with open(str(args.markup), 'r', encoding='utf-8') as f:
        markup = json.load(f)
    for img_name in tqdm(markup):
        img_path = Path(args.dataset_dir) / img_name
        images[img_name] = cv2.imread(str(img_path)) 

    thresholds = {str(net_id): {} for net_id in range(nets_number)}
    for epoch_num2train in range(epochs_step, max(epochs, epochs // epochs_step) + 1, epochs_step):
        trained_ep_num = config["pretrained epochs number"] if "pretrained epochs number" in config \
            else 0

        config["pretrained epochs number"] = trained_ep_num + epochs_step
        
        for net_id in range(nets_number):  
            net, optimizer, lr_scheduler = get_net_optimizer_lrshed(net, net_id, trained_ep_num, 
                epoch_num2train, epochs_step, config, args, net_dir, device)
            net_name2save = args.net_name + f"_{trained_ep_num + epochs_step}ep_{net_id}"
            print(f"Starts training: {net_name2save}")

            alv_name_std_num_dict = get_gt_std_nums_per_bin(markup_trains[net_id]) \
                if config["use_custom_loss"] else None

            trainloader, valloader = get_loaders(args.dataset_dir, markup_trains[net_id], \
                markup_vals[net_id], images, config, args.experts_markup, \
                metalearning_mode=meta_learning_mode, batch_size=config["batch_size"])
            
            if config["finetune"]:
                for param in net.parameters():
                    param.requires_grad = False

                for param in net.pred_classifier.parameters():
                    param.requires_grad = True

                if meta_learning_mode:
                    net.classifier_meta.requires_grad = True
                else:
                    net.classifier.requires_grad = True

            net, thresholds[str(net_id)] = train(device, net, trainloader, loss_func(), optimizer, lr_scheduler, epochs_step, 
                log_dir / str(net_id), valloader, trained_ep_num=trained_ep_num, choose_thr=config["choose_thr"],
                use_label_smoothing=config["use_label_smoothing"], experts_mode=config["experts_mode"], 
                metalearning_mode=config["metalearning_mode"], use_custom_loss=config["use_custom_loss"], 
                alv_name_std_num_dict=alv_name_std_num_dict, experts_markup=args.experts_markup, \
                step2check_val_acc=config["step2check_val_acc"], thresholds=thresholds[str(net_id)])
            
            torch.save(net.state_dict(), net_dir / net_name2save)
        
        trained_ep_num += epochs_step

        if not config["learn_from_scratch"]:
            with open(net_dir / f'config_{args.net_name}.json', "r") as config_file:
                lr_config = json.load(config_file)
            lr_config["last learning rates"][trained_ep_num] = lr_scheduler.get_last_lr()[0]
            with open(net_dir / f'config_{args.net_name}.json', "w") as f:
                json.dump(lr_config, f, indent=4)
        else:
            if "last learning rates" not in config:
                config["last learning rates"] = {}
            config["last learning rates"][trained_ep_num] = lr_scheduler.get_last_lr()[0]
            with open(net_dir / f'config_{args.net_name}.json', "w") as f:
                json.dump(config, f, indent=4)

    if config["choose_thr"]:
        with open(net_dir / f'thresholds_{args.net_name}.json', "w+") as outfile:
            json.dump(thresholds, outfile, indent=4)
