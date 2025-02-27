import json
from pathlib import Path
import argparse
from typing import Dict
import numpy as np
import cv2
from sklearn import metrics
import torch
from tqdm import tqdm

from nn.nn_utils import (get_datasetloader, load_markup,  # pylint: disable=import-error
                         get_net_architecture, check_path_existence)


def get_testloader(test_config):
    path2test_markup = test_config["path2test_markup"]
    path2dataset = test_config["path2dataset"]
    check_path_existence(path2dataset, Path(
        test_config["path2nets_dir"]) / test_config["nets_names"], path2test_markup)
    test_markup = load_markup(path2test_markup)

    print("Loading images: ")
    images = []
    for img_name in tqdm(test_markup):
        img_path = Path(path2dataset) / img_name
        images.append(cv2.imread(str(img_path)))

    testloader = get_datasetloader(
        test_config, test_markup, images)

    return testloader


def load_net(net_id, device, test_config):
    net_name = f"{test_config['nets_names']}_{test_config['epochs']}ep_{net_id}"
    reagents_num = 13 if "meta" in test_config["architecture"] else 0
    net = get_net_architecture(test_config["architecture"])(
        reagents_num=reagents_num)
    net.load_state_dict(torch.load(Path(test_config["path2nets_dir"], map_location=device) /
                                   f"{test_config['nets_names']}/{net_name}", map_location=device))
    net.to(device)
    net.eval()
    return net


def run_net(device, net, testloader, metalearning_mode: bool = False):
    '''
    Get the net's outputs.
    '''

    net_outputs = {}
    net.to(device)
    net.eval()

    with torch.no_grad():
        for data in testloader:
            img, name = data['image'].to(device).float(), data['name'][0]

            meta_reagent = data['meta_reagent_type'].to(
                device) if metalearning_mode else None

            output = net(img, meta_reagent)
            output = output.cpu().detach().numpy()[..., 0]
            net_outputs[name] = output[0]

    return net_outputs


def run_net_with_true_lebels(device, net, testloader, thr=0.5, metalearning_mode: bool = False):
    '''
    Get the net's outputs with true labels.
    '''

    y_true = []
    y_pred = []
    net.to(device)
    net.eval()

    with torch.no_grad():
        for data in testloader:
            img, _, labels = data['image'].to(device).float(), data['name'][0], data['agg_type']

            meta_reagent = data['meta_reagent_type'].to(
                device) if metalearning_mode else None

            if metalearning_mode:
                output = net(img, meta_reagent)
            else:
                output = net(img)

            output = output.cpu().detach().numpy()[..., 0]

            cur_pred = np.where(output >= thr, 1, 0)
            y_pred += list(cur_pred)

            labels = labels.cpu().detach().numpy().astype(int)
            y_true += list(labels)

    return y_pred, y_true


# remove
def find_best_thr(device, net, valloader, val_markup, print_thr=True, metalearning_mode=False):

    net_outputs_dict = run_net(device, net, valloader, metalearning_mode)

    gt_labels, predicted_outputs = [], []
    for alv_name, output in net_outputs_dict.items():
        predicted_outputs.append(output)
        gt_labels.append(val_markup[alv_name]["gt_result"])

    fpr, tpr, thresholds = metrics.roc_curve(gt_labels, predicted_outputs)

    P = gt_labels.count(1)
    N = gt_labels.count(0)
    TP = tpr * P
    TN = (1 - fpr) * N
    accuracy = (TP + TN) / (P + N)

    indexes = np.argwhere(accuracy == np.amax(accuracy)).flatten().tolist()
    index = indexes[len(indexes) // 2]
    threshold_opt = thresholds[index].round(4)
    acc_opt = round(accuracy[index], 3)
    if print_thr:
        print(f"\nBest Threshold: {threshold_opt}")

    return threshold_opt, acc_opt


def prepare2run_net(net_id, test_config: Dict):
    '''
    Runs the nets and saves their outputs (if path2save_nets_outputs provided)
    and lists of misclassified alveoluses names (if path2save_misclas_alvs is provided).
    '''
    testloader = get_testloader(test_config)

    device = torch.device(
            test_config["device_type"] if torch.cuda.is_available() else "cpu")

    net = load_net(net_id, device, test_config)

    return device, net, testloader


def test_nets(test_config: Dict, testloader: torch.utils.data.DataLoader = None):
    '''
    Runs the nets and saves their outputs (if path2save_nets_outputs provided)
    and lists of misclassified alveoluses names (if path2save_misclas_alvs is provided).
    '''
    net_ids = test_config["net_ids"]
    if testloader is None:
        testloader = get_testloader(test_config)

    device = torch.device(
            test_config["device_type"] if torch.cuda.is_available() else "cpu")

    nets_outputs = {}
    for net_id in net_ids:
        print(f"Running net #{net_id}:")
        net = load_net(net_id, device, test_config)

        net_outputs = run_net(device, net, testloader,
                              metalearning_mode="meta" in test_config["architecture"])

        nets_outputs[net_id] = {alv_name: str(
            output) for alv_name, output in net_outputs.items()}

    return nets_outputs


def get_thresholds(test_config):
    net_ids = test_config["net_ids"]

    nets_thresholds = {}

    device = torch.device(
            test_config["device_type"] if torch.cuda.is_available() else "cpu")

    for net_id in net_ids:
        net = load_net(net_id, device, test_config)

        if test_config["choose_thr"]:
            markup_val_path = (Path(test_config["path2nets_dir"]) /
                               f"{test_config['nets_names']}/"
                               f"{test_config['nets_names']}_{net_id}_markup_val.json")
            val_markup = load_markup(markup_val_path)

            valloader = get_datasetloader(test_config, val_markup)

            thr, _ = find_best_thr(
                device, net, valloader, val_markup,
                metalearning_mode="meta" in test_config["architecture"])
        else:
            assert "threshold" in test_config
            thr = test_config["threshold"]

        nets_thresholds[net_id] = str(thr)

    return nets_thresholds


def get_ensemble_mistakes(path2nets_outputs: Path, path2misclas_alvs: Path):
    '''
    For an ensemble calculates misclassified alveoluses (an alv is misclassified
    if it is misclassfied by not less than a half of the ensemble nets).

    Saves the calculated mistakes to the path2misclas_alvs.
    '''
    nets_outputs = load_markup(path2nets_outputs)
    misclas_alvs = load_markup(path2misclas_alvs)

    alvs_outputs_dict = {}
    for net_id, outputs in nets_outputs.items():
        for alv_name in outputs.keys():
            if alv_name not in alvs_outputs_dict:
                alvs_outputs_dict[alv_name] = []
            if alv_name in misclas_alvs[net_id]:
                alvs_outputs_dict[alv_name].append(0)
            else:
                alvs_outputs_dict[alv_name].append(1)

    ensemble_misclas_alvs = []
    for alv_name, results in alvs_outputs_dict.items():
        if results.count(0) >= len(results) // 2:
            ensemble_misclas_alvs.append(alv_name)

    misclas_alvs["ensemble"] = ensemble_misclas_alvs
    with open(path2misclas_alvs, "w", encoding="utf-8") as f:
        json.dump(misclas_alvs, f, indent=4)


def save_statistics(outputs_path, markup, missclass_path, stats_path, thr_path=None):
    # WHY GIVE A DEFAULT VALUE JUST TO RAISE A VALUEERROR WHAT???
    if thr_path is not None:
        with open(thr_path, "r", encoding='utf-8') as f:
            thr_dict = json.load(f)
    else:
        # raise ValueError(f"Path {thr_path} is None.")
        thr_dict = dict()

    with open(outputs_path, "r", encoding='utf-8') as f:
        outputs_dict = json.load(f)

    misclassified_alvs_dict = {}
    staistics = {}

    for net_i in outputs_dict.keys():
        misclassified_alvs_dict[net_i] = []
        thr = float(thr_dict.get(net_i, 0.5)) if thr_path is not None else 0.5

        for img_i in outputs_dict[net_i]:
            gt = markup[img_i]['gt_result']
            output = float(outputs_dict[net_i][img_i])
            output = 1 if output >= thr else 0

            if output != gt:
                misclassified_alvs_dict[net_i].append(img_i)

        staistics[f"Accuracy {net_i}"] = 1 - (len(misclassified_alvs_dict[net_i])
                                              / len(outputs_dict[net_i]))

    accs = [x for x in staistics.values()]
    staistics["Accuracy mean"] = np.mean(accs)
    staistics["Accuracy median"] = np.median(accs)
    staistics["Accuracy min"] = np.min(accs)
    staistics["Accuracy max"] = np.max(accs)

    with open(stats_path, "w", encoding='utf-8') as f:
        json.dump(staistics, f, indent=4)

    with open(missclass_path, "w", encoding='utf-8') as f:
        json.dump(misclassified_alvs_dict, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        "This file is used to test neural networks.")
    parser.add_argument('-ctest', '--test_config',
                        default=Path('nn/net_configs/test_config_classifier.json'))
    parser.add_argument('-o', '--output_dir', default=Path('nn/outputs'))

    parsed_args = parser.parse_args()

    test_config = load_markup(parsed_args.test_config)
    parsed_args.test_config = test_config
    parsed_args.output_dir = Path(parsed_args.output_dir)

    return parsed_args


if __name__ == "__main__":
    args = parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir()

    with open(args.test_config["path2test_markup"], "r", encoding="utf-8") as file:
        test_markup_loaded = json.load(file)

    print(f"Testing {args.test_config['nets_names']}_{args.test_config['epochs']}ep "
          f"on {args.test_config['device_type']}")

    path2outputs = args.output_dir / \
        f"{args.test_config['nets_names']}_{args.test_config['epochs']}ep_outputs.json"
    path2thresholds = args.output_dir / \
        f"{args.test_config['nets_names']}_thresholds.json"
    path2save_misclass = args.output_dir / \
        f"{args.test_config['nets_names']}_misclass.json"
    path2save_statistics = args.output_dir / \
        f"{args.test_config['nets_names']}_stats.json"

    nets_outputs_tested = test_nets(args.test_config)

    if path2outputs is not None:
        with open(path2outputs, "w", encoding="utf-8") as file:
            json.dump(nets_outputs_tested, file, indent=4)

    nets_thresholds_test = get_thresholds(args.test_config)

    if path2thresholds is not None:
        with open(path2thresholds, "w", encoding="utf-8") as file:
            json.dump(nets_thresholds_test, file, indent=4)

    save_statistics(path2outputs,
                    test_markup_loaded,
                    path2save_misclass,
                    path2save_statistics)
