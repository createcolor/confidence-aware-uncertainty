import argparse
from pathlib import Path
import json
import numpy as np
from uncertain_classification.calculate_ensemble_uncertainties import \
                    convert_alvs_unc2unc_alvs, convert_unc_alvs2alvs_unc
from bisect import bisect
from uncertain_classification.DDU import DDU


def load_features(path2features, nets_outputs, net_id):
    with open(path2features, "r") as f:
        features_per_alvs_dict = json.load(f)
    
    train_features_per_class = {}
    features = None
    for alv_name,  alv_features in features_per_alvs_dict.items():
        if features is None:
            features = alv_features
        else:
            features = np.concatenate((features, alv_features), axis=0)

        class_label = int(nets_outputs[net_id][alv_name])
        if class_label not in train_features_per_class:
            train_features_per_class[class_label] = alv_features
        else:
            train_features_per_class[class_label] = np.concatenate(
                (train_features_per_class[class_label], alv_features), axis=0)

    return features, train_features_per_class


def HUQ(args):
    def prepare_uncertainties(path2uncs):
        with open(path2uncs, "r") as f:
            uncs = json.load(f)
        
        uncs_list = []
        for unc, alvs_list in uncs.items():
            uncs_list += [float(unc) for _ in range(len(alvs_list))]

        return sorted(uncs_list)
    
    def get_rank(uncs_list, cur_unc):
        cur_unc = float(cur_unc)
        rank = bisect(uncs_list, cur_unc)
        
        for unc in uncs_list[:rank - 1]:
            if unc == cur_unc:
                rank -= 0.5
        return int(rank)
    
    def get_huq_uncs(aleatorics, epistemics, aleatoric_test, epistemic_test_alv_uncs, 
                     path2save_huq_uncs, alpha=0.5):
        huq_alvs_unc_dict = {}
        for unc, alv_names in aleatoric_test.items(): 
            rank_aleatoric = get_rank(aleatorics, unc) / (len(epistemic_test_alv_uncs) + 1)

            for alv_name in alv_names:
                rank_epistemic = get_rank(epistemics, epistemic_test_alv_uncs[alv_name]) / (len(epistemic_test_alv_uncs) + 1)
                huq_alvs_unc_dict[alv_name] = alpha * rank_aleatoric + (1-alpha) * rank_epistemic

        huq_unc_alvs_dict = convert_alvs_unc2unc_alvs(huq_alvs_unc_dict)

        with open(path2save_huq_uncs, "w") as f:
            json.dump(huq_unc_alvs_dict, f, indent=4)

    net_ids = args.test_config["net_ids"]
    Path(args.path2save_dir).mkdir(parents=True, exist_ok=True)

    for net_id in net_ids:
        path2aleatoric_test = f"{args.path2ddu_outputs}/per-net_entropies_test_{net_id}.json"
        path2epistemic_test = f"{args.path2ddu_outputs}/per-net_epistemic_{args.pca_components_num}PCAcomps_test_{net_id}.json"
        path2aleatoric_val = f"{args.path2ddu_outputs}/per-net_entropies_val_{net_id}.json"
        path2epistemic_val = f"{args.path2ddu_outputs}/per-net_epistemic_{args.pca_components_num}PCAcomps_val_{net_id}.json"
        
        with open(path2aleatoric_test, "r") as f:
            aleatoric_test = json.load(f)
        with open(path2epistemic_test, "r") as f:
            epistemic_test = json.load(f)

        epistemic_test_alv_uncs = convert_unc_alvs2alvs_unc(epistemic_test)

        aleatorics = prepare_uncertainties(path2aleatoric_val)
        epistemics = prepare_uncertainties(path2epistemic_val)

        get_huq_uncs(aleatorics, epistemics, aleatoric_test, epistemic_test_alv_uncs, 
                    path2save_huq_uncs=f"{args.path2save_dir}/huq_alpha={round(args.alpha, 2)}_{net_id}.json", alpha=args.alpha)
    

def parse_args():
    parser = argparse.ArgumentParser("This file allows to calculate uncertainty using HUQ method.")
    parser.add_argument('-ctest', '--path2test_config', default=Path('nn/net_configs/test_config_classifier.json'),
        help='configuration file')
    
    parser.add_argument('-pca', '--pca_components_num', default=10, type=int,
        help='the number of PCA components')
    
    parser.add_argument('-s', '--path2save_dir', type=Path, default=Path('results/huq'),
        help='Path to the dir where uncertainties and other output files should be saved.')
    
    parser.add_argument('-ddu', '--path2ddu_outputs', type=Path, default=Path('results/ddu'),
        help='Path to the directory with DDU outputs.')
    
    parser.add_argument('-otrain', '--path2outputs_train', type=Path, 
        default=Path('nn/outputs/MobileNet_small_10_10_80_meta_8_400ep_outputs_train_val.json'),
        help="Path to the net's outputs on train dataset.")
    
    parser.add_argument('-a', '--alpha', type=float, 
        default=0.5, help="Paramether of weighted sum of two uncertainties.")
    
    args = parser.parse_args()

    with open(args.path2test_config, "r") as config_file:
        args.test_config = json.load(config_file)

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Running DDU on validaion:")
    DDU(args.test_config, args.path2ddu_outputs, args.path2outputs_train, 
        test_on_val_data=True, calc_epistemic=True, calc_aleatoric=True, 
        n_components=args.pca_components_num, test_set='val')
   
    print("Running HUQ:")
    HUQ(args)

    

   