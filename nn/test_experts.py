import json
from pathlib import Path
import numpy as np

from nn.test_classifier import test_nets, run_net

from typing import Dict, List

import json
import cv2
import argparse
import scipy

from nn.nn_utils import load_markup

def run_net4experts_prediction(device, net, valloader, metalearning_mode, 
                               experts_mode, experts_markup, epsilon=0.0625):

    net_outputs = run_net(device, net, valloader, metalearning_mode=metalearning_mode)

    wrong_preds = 0 
    for alv_name, pred_result in net_outputs.items(): 
        if experts_mode == "class":
            if np.argmax(pred_result) != experts_markup[alv_name]["expert_class"]:
                wrong_preds += 1
        else:
            pred_result = float(pred_result) / 2
            
            if experts_mode == "std":
                gt_result = np.std(experts_markup[alv_name]["expert_votes_positions"]) / 2
            elif experts_mode == "dist2integer":
                gt_result = 0.5 - abs(0.5 - np.mean(experts_markup[alv_name]["expert_votes_positions"]) / 4)
            elif experts_mode == "average_vote":
                gt_result = np.mean(experts_markup[alv_name]["expert_votes_positions"]) / 4
            else:
                raise ValueError("Unknown experts_mode")
            
            abs_error = abs(float(pred_result) - gt_result)
            if abs_error > epsilon:
                wrong_preds += 1

    accuracy = round(1 - wrong_preds / len(net_outputs), 3)

    return accuracy

def calculate_statistics(path2predictions: Path, path2experts_markup: Path, \
    path2dataset: Path, stats_path: Path, prediction_type: str="std", path2save_mistakes: Path=None,
    epsilon=0.0625):
    '''
    Estimates the quality of experts prediction.  

    @param prediction_type: "std", "dist2integer". Defines the way to calculate gt answers. 
    '''
    predicitons_per_nets = load_markup(path2predictions)
    gt_results = load_markup(path2experts_markup)

    staistics = {}
    
    for net_id, predicitons in predicitons_per_nets.items():
        wrong_preds = 0 
        abs_errors = []
        predicitons_list, gt_list = [], []
        for alv_name, pred_result in predicitons.items(): 
            pred_result = float(pred_result) / 2
            predicitons_list.append(pred_result)
            if prediction_type == "std":
                gt_result = np.std(gt_results[alv_name]["expert_votes_positions"]) / 2
            elif prediction_type == "dist2integer":
                gt_result = 0.5 - abs(0.5 - np.mean(gt_results[alv_name]["expert_votes_positions"]) / 4)

            gt_list.append(gt_result)

            abs_error = abs(float(pred_result) - gt_result)
            abs_errors.append(abs_error)

            if abs_error > epsilon:
                wrong_preds += 1
                if path2save_mistakes is not None:
                    alv_img = cv2.imread(f"{path2dataset}/{alv_name}")
                    cv2.imwrite(f"{path2save_mistakes}/{alv_name}", alv_img)

        staistics[net_id] = {}
        staistics[net_id]["MAE"] = round(sum(abs_errors) / len(abs_errors), 3)
        staistics[net_id]["Accuracy"] = round(1 - wrong_preds / len(predicitons), 3)
        staistics[net_id]["Pearson_cor"] = round(scipy.stats.pearsonr(predicitons_list, gt_list)[0], 3)

    staistics["Agregation"] = {}
    staistics["Agregation"]["Mean"] = {}
    staistics["Agregation"]["Mean"]["MAE"] = np.mean([x["MAE"] for x in staistics.values() if "MAE" in x])
    staistics["Agregation"]["Mean"]["Accuracy"] = np.mean([x["Accuracy"] for x in staistics.values() if "Accuracy" in x])
    staistics["Agregation"]["Mean"]["Pearson_cor"] = np.mean([x["Pearson_cor"] for x in staistics.values() if "Pearson_cor" in x])

    staistics["Agregation"]["Median"] = {}
    staistics["Agregation"]["Median"]["MAE"] = np.median([x["MAE"] for x in staistics.values() if "MAE" in x])
    staistics["Agregation"]["Median"]["Accuracy"] = np.median([x["Accuracy"] for x in staistics.values() if "Accuracy" in x])
    staistics["Agregation"]["Median"]["Pearson_cor"] = np.median([x["Pearson_cor"] for x in staistics.values() if "Pearson_cor" in x])

    staistics["Agregation"]["Min"] = {}
    staistics["Agregation"]["Min"]["MAE"] = np.min([x["MAE"] for x in staistics.values() if "MAE" in x])
    staistics["Agregation"]["Min"]["Accuracy"] = np.min([x["Accuracy"] for x in staistics.values() if "Accuracy" in x])
    staistics["Agregation"]["Min"]["Pearson_cor"] = np.min([x["Pearson_cor"] for x in staistics.values() if "Pearson_cor" in x])

    staistics["Agregation"]["Max"] = {}
    staistics["Agregation"]["Max"]["MAE"] = np.max([x["MAE"] for x in staistics.values() if "MAE" in x])
    staistics["Agregation"]["Max"]["Accuracy"] = np.max([x["Accuracy"] for x in staistics.values() if "Accuracy" in x])
    staistics["Agregation"]["Max"]["Pearson_cor"] = np.max([x["Pearson_cor"] for x in staistics.values() if "Pearson_cor" in x])


    staistics["EPS"] = epsilon

    with open(stats_path, "w") as f:
        json.dump(staistics, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        "This file is used to test neural networks.")
    parser.add_argument('-ctest', '--test_config',
                        default=Path('nn/net_configs/test_config_experts.json'))
    parser.add_argument('-m', '--path2experts_markup', type=Path, default=Path("markup/experts_info_per_alvs.json"))
    parser.add_argument('-o', '--output_dir', type=Path, default=Path('nn/outputs_experts'))

    args = parser.parse_args()

    test_config = load_markup(args.test_config)
    args.test_config = test_config

    return args

if __name__ == "__main__":
    args = parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir()

    with open(args.test_config["path2test_markup"], "r", encoding="utf-8") as f:
        test_markup = json.load(f)

    print(
        f"Testing {args.test_config['nets_names']}_{args.test_config['epochs']}ep on {args.test_config['device_type']}")

    path2outputs = args.output_dir / \
        f"{args.test_config['nets_names']}_{args.test_config['epochs']}ep_outputs.json"
    path2save_misclass = args.output_dir / \
        f"{args.test_config['nets_names']}_misclass.json"
    path2save_statistics = args.output_dir / \
        f"{args.test_config['nets_names']}_stats.json"

    nets_outputs = test_nets(args.test_config)
    
    if path2outputs is not None:
        with open(path2outputs, "w", encoding="utf-8") as f:
            json.dump(nets_outputs, f, indent=4)

    calculate_statistics(path2outputs, args.path2experts_markup, \
        args.test_config["path2dataset"], path2save_statistics, \
        prediction_type="dist2integer", \
        epsilon=0.0625)

    