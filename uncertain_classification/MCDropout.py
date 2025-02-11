import sys

import numpy as np

import torch
import json
from nn.dataset import DatasetGenerator, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from nn.MobileNet import mobilenet_v3_large, mobilenet_v3_small_meta
from tqdm import tqdm
from pathlib import Path
import argparse
from nn.nn_utils import get_datasetloader

def parse_args():
    parser = argparse.ArgumentParser(
        'Inference for MC Dropout')
    
    parser.add_argument('-c', '--config', default=Path("./nn/net_configs/test_mc_dropout.json"), type=Path)
    args = parser.parse_args()

    return args

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_samples,
                                device,
                                path2save):

    markup = {}

    
    for i in tqdm(range(forward_passes)):
        model.eval()
        enable_dropout(model)
        for i, data in enumerate(data_loader):
            inputs = data['image'].to(device).float()
            name = data['name'][0]
            labels = data['agg_type'] 
            labels = labels.to(device)
            metalearning_mode = 1
            meta_reagent = data['meta_reagent_type'].to(device) if metalearning_mode else None
            
            outputs = model(inputs, meta_reagent)
            
            answer = outputs.item()
            if (name in markup):
                markup[name].append(answer)
            else:
                markup[name] = [answer]

    with open(path2save, "w") as f:
        json.dump(markup, f, indent=4)


def make_scores_alvs_markup(alvs_answers_markup):
    new_markup = {}

    for alv in alvs_answers_markup:

        ans = alvs_answers_markup[alv]
        unc = round(np.std(ans), 3)

        if unc in new_markup:
            new_markup[unc].append(alv)
        else:
            new_markup[unc] = [alv]

    return new_markup

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    path2dataset = config["path2dataset"]
    markup_path = config["test_markup"]
    train_config_path = config["train_config"]
    path2models = config["nets_dir"]
    path2save_ans = Path(config["answers_path"])
    path2save_scores = Path(config["alv_scores_path"])
    basic_net_name = config["basic_net_name"]
    num_of_inferences = config["number_of_inferences"]
    test_config_path = config["test_config"]
    number_of_nets = config["number_of_nets"]

    if not path2save_ans.exists():
        path2save_ans.mkdir()

    if not path2save_scores.exists():
        path2save_scores.mkdir()

    with open(test_config_path, "r") as f:
        config_test = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(train_config_path, "r") as f:
            config_train = json.load(f)
    dataset_params = config_train["dataset_info"]

    with open(markup_path, "r") as f:
        test_markup = json.load(f)

    transforms_list = [ToTensor()] 
    testloader = get_datasetloader(config_test, test_markup)

    list_p = [0.2]

    models_names = [basic_net_name + "_" + str(i) for i in range(number_of_nets)]
    for model_name in models_names:
        path2model = path2models/Path(model_name)

        for p in list_p:
            path_ans = path2save_ans / Path(model_name + "_answers_" + str(p)  + ".json")
            path_unc = path2save_scores / Path(model_name + ".json")

            model = mobilenet_v3_small_meta(reagents_num=13).to(device)
            model.load_state_dict(torch.load(path2model, map_location=device))

            model.pred_classifier[2] = torch.nn.Dropout(p=p, inplace=True)

            get_monte_carlo_predictions(testloader, num_of_inferences, model, len(test_markup), device, path_ans)

            with open(path_ans, "r") as f:
                markup = json.load(f) 

            new_markup = make_scores_alvs_markup(markup)

            with open(path_unc, "w") as f:
                json.dump(new_markup, f, indent=4)


