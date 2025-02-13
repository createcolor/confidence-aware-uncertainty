from nn.dataset import DatasetGenerator, ToTensor, NormalizeImg
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import shutil
from pathlib import Path

import numpy as np 
from random import choice
import json
from collections import OrderedDict

transforms_dict = {
    "efficientnet": [ToTensor()],
    "mobilenet": [ToTensor()],
    "densenet": [ToTensor()],
    "resnet": [NormalizeImg(), ToTensor()],
    "default": [ToTensor()]
}

def get_train_val_markups(nets_number, config, net_name, 
                          markup_train_path, net_dir, train_part=0.5):
    markup_trains, markup_vals = {}, {}
    for net_id in range(nets_number):
        if config["learn_from_scratch"]:
            markup_trains[net_id], markup_vals[net_id] = \
                train_val_split(markup_train_path, net_id,
                                train_part=train_part)
            save_train_val_markup(
                markup_trains[net_id], markup_vals[net_id], net_id, net_dir, net_name)
        else:
            markup_train_path = net_dir / \
                (net_name + f"_{net_id}_markup_train.json")
            markup_val_path = net_dir / \
                (net_name + f"_{net_id}_markup_val.json")

            if not config["learn_from_scratch"]:
                net_dir2learn_from = \
                    Path(config['net_name2learn_from'][:config['net_name2learn_from'].rfind("_")])
                assert config["net_name2learn_from"] is not None, \
                    "Please provide the name of the net to learn from"
                if not markup_train_path.exists():
                    src_path = Path(config['path2nets_dir']) / net_dir2learn_from / \
                               (net_dir2learn_from.name + f"_{net_id}_markup_train.json")
                    shutil.copyfile(src_path, markup_train_path)
                if not markup_val_path.exists():
                    src_path = Path(config['path2nets_dir']) / net_dir2learn_from / \
                               (net_dir2learn_from.name + f"_{net_id}_markup_val.json")
                    shutil.copyfile(src_path, markup_val_path)

            with open(markup_train_path, "r") as f:
                markup_trains[net_id] = json.load(f)
            with open(markup_val_path, "r") as f:
                markup_vals[net_id] = json.load(f)

    return markup_trains, markup_vals


def get_loaders(path_to_dataset, markup_train, markup_val, 
                images_dict, config, experts_markup=None,
                metalearning_mode=False, batch_size=10):
    if metalearning_mode:
        reagents_list = [
            "A", "B", "O(I)", "A(II)", "B(III)", "D", "C", "c", "E", "e", "Cw", "K", "k"]
    else:
        reagents_list = None

    images_train = [images_dict[img_name] for img_name in markup_train.keys()]
    traindataset = DatasetGenerator(path_to_dataset, markup_train, images_train,
                                    transform=transforms.Compose([ToTensor()]), config=config,
                                    augment=True, random_crop=True, 
                                    experts_markup=experts_markup, reagents_list=reagents_list)
    trainloader = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=8)

    images_val = [images_dict[img_name] for img_name in markup_val.keys()]
    valdataset = DatasetGenerator(path_to_dataset, markup_val, images_val,
                                  transform=transforms.Compose([ToTensor()]), config=config,
                                  augment=False, random_crop=False, 
                                  experts_markup=experts_markup, reagents_list=reagents_list)
    
    valloader = DataLoader(valdataset, batch_size=1,
                           shuffle=False, num_workers=8)

    return trainloader, valloader

def train_val_split_by_rows(markup_path):
    '''
    Prepares training and validation split for cross-validation.
    '''
    with open(markup_path, "r") as f:
        markup = json.load(f, object_pairs_hook=OrderedDict)

    first_alv_name_split = list(markup.keys())[0].split('_')
    img_name_prev = '_'.join(first_alv_name_split[:-2])

    markup_train = {}
    markup_val = {}
    markup_img = {}

    for alv_name in markup.keys():
        alv_name_split = alv_name.split('_')
        img_name_cur = '_'.join(alv_name_split[:-2])

        if img_name_prev != img_name_cur:
            img_name_prev = img_name_cur
            val_key = choice(list(markup_img.keys()))

            for k in markup_img.keys():
                if k == val_key:
                    markup_val.update(markup_img[k])
                else:
                    markup_train.update(markup_img[k])

            markup_img = {}

        row = alv_name_split[-2]

        if row not in markup_img.keys():
            markup_img[row] = {}
        
        markup_img[row][alv_name] = markup[alv_name]
    
    return markup_train, markup_val


def train_val_split(markup_path, net_id, train_part=0.5):
    
    train_markup, val_markup = {}, {}

    with open(markup_path, 'r') as f:
        markup = json.load(f)

    ## making dict with reagents and number of 0/1
    reagents_dict = {}
    keys = list(markup.keys())
    np.random.seed(net_id)
    np.random.shuffle(keys)

    for alv in markup.values():
        reagent = alv['reagent']
        gt = alv['gt_result']

        if reagent not in reagents_dict:
            # (num of 0, num of 1, num of 0 in train now, num of 1 in train now)
            reagents_dict[reagent] = [0, 0, 0, 0]
        
        reagents_dict[reagent][gt] += 1
        
    for alv_name in keys:
        alv = markup[alv_name]
        reagent = alv['reagent']
        gt = alv['gt_result']

        if int(reagents_dict[reagent][gt] * train_part) > reagents_dict[reagent][gt + 2]:
            reagents_dict[reagent][gt + 2] += 1
            train_markup[alv_name] = alv
        else:
            val_markup[alv_name] = alv

    return train_markup, val_markup


def save_train_val_markup(markup_train, markup_val, net_id, net_dir, net_name):
    '''
    Save markups for cross-validation to .json-s
    '''
    new_markup_train_path = net_dir  / \
        (net_name + f"_{net_id}_markup_train.json")
    new_markup_val_path = net_dir  / \
        (net_name + f"_{net_id}_markup_val.json")

    with open(new_markup_train_path, "w") as f:
        json.dump(markup_train, f, indent=4)
    with open(new_markup_val_path, "w") as f:
        json.dump(markup_val, f, indent=4)