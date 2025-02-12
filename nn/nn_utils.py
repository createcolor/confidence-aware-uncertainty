import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List

from nn.dataset import DatasetGenerator, ToTensor
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import argparse
from tqdm import tqdm
import nn.EfficientNet
import nn.MobileNet 
import nn.DenseNet
import scipy
import sys


def check_path_existence(*args):
    for arg in args:
        assert Path(arg).exists(), f"{arg} does not exist"

def mkdirs(*dir_paths):
    for dir_path in dir_paths:
        if not (dir_path).exists():
            (dir_path).mkdir()

def load_markup(path2markup: Path):
    '''
    Loads the .json file
    '''
    with open(path2markup, "r", encoding='utf-8') as f:
        markup = json.load(f)
    return markup

def get_datasetloader(config: Dict, data_markup: Dict, images: Dict=None, run_mode: bool=True, \
    experts_markup=None, batch_size: int=None):
    '''
    Downloads the dataloader for training and running nets.
    
    @param run_mode: if is True then the dataloader will be used for running nets 
    otherwise --- for training them.
    @param dataset_params: info from train or test config.
    '''
    reagents_list = ["A", "B", "O(I)", "A(II)", "B(III)", "D", "C", "c", "E", "e", "Cw", "K", "k"] \
        if "meta" in config["architecture"] else None

    if run_mode:
        shuffle = False
        batch_size = batch_size if batch_size is not None else 1
        augment = False
        random_crop = False
    else:
        shuffle = True
        batch_size = batch_size if batch_size is not None else 8
        augment = True
        random_crop = True

    if images is None:
        images = []
        for img_name in tqdm(data_markup):
            img_path = Path(config["path2dataset"]) / img_name
            images.append(cv2.imread(str(img_path)))

    dataset = DatasetGenerator(config["path2dataset"], data_markup, images, \
        transform=transforms.Compose([ToTensor()]), config=config, \
        augment=augment, random_crop=random_crop, experts_markup=experts_markup, \
        reagents_list=reagents_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    
    return dataloader


def get_net_architecture(net_arch_name):
    if "efficientnet" in net_arch_name:
        net_architecture = getattr(sys.modules["nn.EfficientNet"], net_arch_name)
    elif "mobilenet" in net_arch_name:
        net_architecture = getattr(sys.modules["nn.MobileNet"], net_arch_name)
    elif "densenet" in net_arch_name:
        net_architecture = getattr(sys.modules["nn.DenseNet"], net_arch_name)
    else:
        net_architecture = getattr(sys.modules["nn.nets"], net_arch_name)
    
    return net_architecture


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


def save_ensemble_mistakes_imgs(path2ensemble_mistakes: Path, path2gt_markup: Path,
    path2dataset: Path, path2save_imgs: Path):
    '''
    Gets the images of alveoluses where the ensemble made mistakes according to the markup
    from path2ensemble_mistakes. Adds the gt answer to these images, the gt answer is taken
    from path2gt_markup (for example, an experts votes markup). Saves the obtained images 
    to the path2save_imgs.
    '''
    mistakes = load_markup(path2ensemble_mistakes)
    gt_markup = load_markup(path2gt_markup)

    for alv_name in mistakes["ensemble"]:           
        alv_img = cv2.imread(f"{path2dataset}/{alv_name}")
        alv_img = cv2.putText(alv_img, str(gt_markup[alv_name]["gt_result"]), 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10, cv2.LINE_AA)
        cv2.imwrite(f"{path2save_imgs}/{alv_name}", alv_img)


def estimate_experts_votes_prediction(path2predictions: Path, path2experts_markup: Path, \
    path2dataset: Path, prediction_type: str="std", path2save_mistakes: Path=None,
    epsilon=0.0625):
    '''
    Estimates the quality of experts prediction.  

    @param prediction_type: "std", "dist2integer". Defines the way to calculate gt answers. 
    '''
    predicitons_per_nets = load_markup(path2predictions)
    gt_results = load_markup(path2experts_markup)

    accuracies, maes, pearson_cors = [], [], []
    
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
        mae = sum(abs_errors) / len(abs_errors)
        maes.append(mae)  
        accuracy = round(1 - wrong_preds / len(predicitons), 3)
        accuracies.append(accuracy)

        pearson_cor = scipy.stats.pearsonr(predicitons_list, gt_list)[0]
        pearson_cors.append(pearson_cor)
        print(f"Net #{net_id} has mae = {round(mae, 3)}, accuracy = {accuracy} and pearson_correlation = {round(pearson_cor, 3)}")
    
    return accuracies, mae, pearson_cors


def experts_predictions_plot_heatmap(path2predictions: Path, path2gt_experts_markup: Path, net_ids=range(20),
    log_scale_mode=True, reagent="all", step=0.1, prediction_type="dist2integer", max_value=0.5):
    '''
    Plots the number of predicted experts values per bins against the number of gt ones.    
    '''
    def select_given_reagent(reagent, nets_predictions, experts_markup, net_ids):
        if reagent != "all":
            experts_markup = {alv_name: alv_info for alv_name, alv_info in experts_markup.items() \
                            if "reagent" in alv_info and alv_info["reagent"] == reagent}
            for net_id in net_ids:
                nets_predictions[str(net_id)] = {alv_name: alv_info for alv_name, alv_info \
                    in nets_predictions[str(net_id)].items() if alv_name in experts_markup}
        return nets_predictions
    
    def aggregate_predictions(nets_predictions, net_ids):
        predicted_dict = {}
        for net_id in net_ids:
            predicted_cur = nets_predictions[str(net_id)]
            for alv_name, net_output in predicted_cur.items():
                net_output = float(net_output) / 2 
                if alv_name not in predicted_dict:
                    predicted_dict[alv_name] = []
                predicted_dict[alv_name].append(net_output)
        predicted_dict = {alv_name: np.mean(outputs) for alv_name, outputs in predicted_dict.items()}
        return predicted_dict
    
    def calculate_heatmap_values(experts_markup, predicted_dict, step, max_value):
        stability_const = 1 / step
        heatmap_data = [[0 for _ in range(int(max_value * stability_const) // int (step * stability_const))] \
                        for _ in range(int(max_value * stability_const) // int (step * stability_const))]

        for alv_name, net_output in predicted_dict.items(): 
            if prediction_type == "std":
                gt_result = np.std(experts_markup[alv_name]["expert_votes_positions"]) / 2
            elif prediction_type == "dist2integer":
                gt_result = 0.5 - abs(0.5 - np.mean(experts_markup[alv_name]["expert_votes_positions"]) / 4)
            
            heatmap_data[min(int(net_output / step), len(heatmap_data) - 1)]\
                        [min(int(gt_result / step), len(heatmap_data[0]) - 1)] += 1
            
        return heatmap_data
    
    nets_predictions = load_markup(path2predictions)
    experts_markup = load_markup(path2gt_experts_markup)
 
    nets_predictions = select_given_reagent(reagent, nets_predictions, experts_markup, net_ids)
    predicted_dict = aggregate_predictions(nets_predictions, net_ids)

    heatmap_data = calculate_heatmap_values(experts_markup, predicted_dict, step, max_value)
        
    fig, ax = plt.subplots()

    if log_scale_mode:
        for i in range(len(heatmap_data)):
            for j in range(len(heatmap_data[0])):
                heatmap_data[i][j] = np.log(heatmap_data[i][j])
    else:
        for (i, j), z in np.ndenumerate(heatmap_data):
            ax.text(j, i, max(0, z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        
    caxes = ax.matshow(heatmap_data, cmap='inferno')

    ax.set_xticks(np.arange(len(heatmap_data) + 1) - 0.5, labels=np.round(np.arange(0, max_value + step, step), 1))
    ax.set_yticks(np.arange(len(heatmap_data) + 1) - 0.5, labels=np.round(np.arange(0, max_value + step, step), 1))

    ax.set_title(f"Experts {prediction_type}: predicted VS gt (log scale of alvs numbers), reagent={reagent}")
    fig.colorbar(caxes)
    plt.xlabel(f"Gt {prediction_type} values")
    plt.ylabel(f"Predicted {prediction_type} values")

    plt.savefig(f"results/predicted_{prediction_type}_heatmap.png")
