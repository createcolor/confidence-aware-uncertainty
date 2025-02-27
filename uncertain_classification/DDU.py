import argparse
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from nn.nn_utils import get_datasetloader
from nn.test_classifier import prepare2run_net
from uncertain_classification.calculate_ensemble_uncertainties import convert_alvs_unc2unc_alvs


def get_net_features(net, device, testloader, features_per_alvs_dict={}, metalearning_mode=False):
    '''
    Convertes testloader elements to the net's feature space ---
    features are taken after the last convolutional layer.
    '''
    all_features = None
    with torch.no_grad():
        for data in testloader:
            img, alv_name = data['image'].to(device).float(), data['name'][0]

            meta_reagent = data['meta_reagent_type'].to(device) if metalearning_mode else None

            if metalearning_mode:
                features = net(img, meta_reagent, return_features=True, get_net_outputs=False).cpu()
            else:
                features = net(img, return_features=True, get_net_outputs=False).cpu()

            if all_features is None:
                all_features = features
            else:
                all_features = np.concatenate((all_features, features), axis=0)

            features_per_alvs_dict[alv_name] = features.numpy()[0]

    return all_features, features_per_alvs_dict


def calc_GMM_parameters(class_features, class_trainloader, whole_train_size: int):
    '''
    Calculates the Gaussian distribution parameters for a specific class (agglutination
    present or absent): mean mu and covariation matrix sigma. Also calculates pi ---
    the probability of the considered class which is actually the proportion of this class
    elements in the training set.
    (GMM stands for the Gaussian Mixture Model)
    '''
    class_objects_num = len(class_trainloader)

    mu = sum(class_features) / class_objects_num
    if class_objects_num > 1:  # check if there is more than a single object in the class
        sigma = (class_features - mu).T @ (class_features - mu) / (class_objects_num - 1)
    else:
        sigma = np.zeros((len(class_features), len(class_features))) + 999999
    pi = class_objects_num / whole_train_size
    return mu, sigma, pi


def calculate_density(test_img, net, meta_reagent, class_params_dict,
                      features_mean_before=None, features_std_before=None, pca=None):
    '''
    For the given test_img calculates the epistemic uncertainty as the Gausion misture density
    '''
    if meta_reagent is not None:
        test_img_features, output = net(test_img, meta_reagent, return_features=True,
                                        get_net_outputs=True)
    else:
        test_img_features, output = net(test_img, return_features=True,
                                        get_net_outputs=True)

    test_img_features = test_img_features.cpu().numpy()
    test_img_features = (test_img_features - features_mean_before) / features_std_before
    if pca is not None:
        test_img_features = pca.transform(test_img_features)

    output = float(output.cpu().numpy()[0][0])

    test_img_features = test_img_features[0]
    n = len(test_img_features)

    density = 0
    for class_label in [0, 1]:
        if class_label in class_params_dict:
            mu, sigma, pi = class_params_dict[class_label]

            power = -0.5 * (test_img_features - mu) @ \
                np.linalg.inv(sigma) @ (test_img_features - mu).T

            density += pi * (np.exp(power)) / np.sqrt((2 / np.pi)**n * np.linalg.det(sigma))
            if density == float("inf"):
                density = 9999999999999
        else:
            density += 0.0

    return density


def load_features(path2features, nets_outputs, net_id):
    with open(path2features, "r", encoding='utf-8') as f:
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


def get_pca(test_config, trainloaders, testloader, n_components, net, device, nets_outputs, net_id,
            path2features=None, calc_features=True):
    '''
    Fits the pca transformer on the whole dataset (without validation).
    '''
    pca = PCA(n_components=n_components)

    if calc_features:
        features, features_per_alvs_dict = get_net_features(net, device, testloader,
                                                            metalearning_mode="meta" in
                                                            test_config["architecture"])

        train_features_per_class = {}
        for class_label in [0, 1]:
            train_features_per_class[class_label], features_per_alvs_dict = get_net_features(
                net, device, trainloaders[class_label], features_per_alvs_dict,
                metalearning_mode="meta" in test_config["architecture"])

            features = np.concatenate((features, train_features_per_class[class_label]), axis=0)

    else:
        features, train_features_per_class = load_features(path2features, nets_outputs, net_id)

    features_mean_before = np.mean(features, axis=0)
    features_std_before = np.std(features, axis=0)

    features = (features - features_mean_before) / features_std_before
    pca.fit(features)

    transformed_features = pca.transform(features)

    return pca, transformed_features, features_mean_before, features_std_before


def get_trainloaders(test_config, net_id, nets_outputs):
    trainloaders = {}
    whole_train_size = 0
    markup_train_path = (f"{test_config['path2nets_dir']}/{test_config['nets_names']}/"
                         f"{test_config['nets_names']}_{net_id}_markup_train.json")

    with open(markup_train_path, "r", encoding='utf-8') as config_file:
        markup_train = json.load(config_file)

    for class_label in [0, 1]:
        markup_train_cur_class = {alv_name: alv_info for alv_name, alv_info in markup_train.items()
                                  if nets_outputs[str(net_id)][alv_name] == class_label}

        trainloaders[class_label] = get_datasetloader(test_config, markup_train_cur_class,
                                                      run_mode=True)

        whole_train_size += len(trainloaders[class_label])

    return trainloaders, whole_train_size


def normalize_densities(densities):
    '''
    currently not used, but can help for high dimensions
    edge values --> quantile
    '''
    upper_bound = np.quantile(list(densities.values()), 0.85)
    lower_bound = np.quantile(list(densities.values()), 0.15)
    for k, v in densities.items():
        if v < lower_bound:
            densities[k] = lower_bound
        elif v > upper_bound:
            densities[k] = upper_bound

    return densities


def calculate_entropy(output):
    entropy = 0.0
    output = float(output)
    if output < 1.0 and output > 0.0:
        entropy = -(output * np.log(output) + (1.0 - output) * np.log(1.0 - output))

    return entropy


def calc_epistemic_uncs_per_nets(path2outputs_train, net_id, test_config, n_components, densities,
                                 epistemic_per_net, path2save_dir, test_set):
    with open(path2outputs_train, "r", encoding='utf-8') as f:
        nets_outputs = json.load(f)

    for alv_name, output in nets_outputs[str(net_id)].items():
        nets_outputs[str(net_id)][alv_name] = round(float(output))

    device, net, testloader = prepare2run_net(net_id, test_config)

    trainloaders, whole_train_size = get_trainloaders(test_config, net_id, nets_outputs)

    pca, transformed_features, features_mean_before, features_std_before = \
        get_pca(test_config, trainloaders, testloader, n_components, net,
                device, nets_outputs, net_id)

    class_params_dict = {}
    train_features_per_class = {
        0: transformed_features[len(testloader): len(testloader) + len(trainloaders[0])],
        1: transformed_features[len(testloader) + len(trainloaders[0]):]
    }
    for class_label in [0, 1]:
        class_features = train_features_per_class[class_label]

        mu, sigma, pi = calc_GMM_parameters(class_features,
                                            trainloaders[class_label], whole_train_size)
        class_params_dict[class_label] = mu, sigma, pi

    metalearning_mode = "meta" in test_config["architecture"]

    with torch.no_grad():
        for data in tqdm(testloader):
            sample, img_name = data['image'].to(device).float(), data['name'][0]
            meta_reagent = data['meta_reagent_type'].to(device) if metalearning_mode else None

            density = calculate_density(sample, net, meta_reagent, class_params_dict,
                                        features_mean_before, features_std_before, pca=pca)

            densities[net_id][img_name] = density
            epistemic_per_net[img_name] = 1.0 - density

    epistemic_per_net = convert_alvs_unc2unc_alvs(epistemic_per_net)
    with open(f"{path2save_dir}/per-net_epistemic_{n_components}PCAcomps_{test_set}_{net_id}.json",
              "w", encoding='utf-8') as f:
        json.dump(epistemic_per_net, f, indent=4)


def calc_entropies_per_nets(test_on_val_data, path2outputs_train, path2outputs_test,
                            test_config, net_id, entropies, entropies_per_net,
                            path2save_dir, test_set):
    if test_on_val_data:
        with open(path2outputs_train, "r", encoding='utf-8') as f:
            nets_outputs = json.load(f)
    else:
        with open(path2outputs_test, "r", encoding='utf-8') as f:
            nets_outputs = json.load(f)

    with open(test_config["path2test_markup"], "r", encoding='utf-8') as f:
        test_markup = json.load(f)

    nets_outputs_cur_net = nets_outputs[str(net_id)]

    nets_outputs_cur_net = {alv_name: output for alv_name, output in nets_outputs_cur_net.items()
                            if alv_name in test_markup}
    assert len(nets_outputs_cur_net) > 0, "Please check nets_outputs match the test markup"

    for img_name, output in nets_outputs_cur_net.items():
        entropies[net_id][img_name] = calculate_entropy(output)
        entropies_per_net[img_name] = entropies[net_id][img_name]

    entropies_per_net = convert_alvs_unc2unc_alvs(entropies[net_id])
    with open(f"{path2save_dir}/per-net_entropies_{test_set}_{net_id}.json", "w",
              encoding='utf-8') as f:
        json.dump(entropies_per_net, f, indent=4)


def prepare_paths(pca_components_num, path2save_dir, test_set):
    path2save_dir = Path(path2save_dir)
    path2save_dir.mkdir(parents=True, exist_ok=True)

    path2save_densities = f"{path2save_dir}/densities_{pca_components_num}PCAcomps_{test_set}.json"
    path2save_epist_uncs = (f"{path2save_dir}/epistemic_uncs_{pca_components_num}"
                            f"PCAcomps_{test_set}.json")
    path2save_entropies = f"{path2save_dir}/entropies_unc_{test_set}.json"

    return path2save_densities, path2save_epist_uncs, path2save_entropies


def aggregate_and_save_epistemic_uncs(densities, path2save_epist_uncs):
    ensemble_uncs = {}
    for net_outputs in densities.values():
        for alv_name, density in net_outputs.items():
            if alv_name not in ensemble_uncs:
                ensemble_uncs[alv_name] = []

            unc = 1.0 - density
            ensemble_uncs[alv_name].append(float(unc))

    ensemble_mean_alvs_names_dict = {}

    for alv_name, outputs in ensemble_uncs.items():
        ensemble_mean = np.mean(outputs)
        if ensemble_mean not in ensemble_mean_alvs_names_dict:
            ensemble_mean_alvs_names_dict[ensemble_mean] = []
        ensemble_mean_alvs_names_dict[ensemble_mean].append(alv_name)

    with open(path2save_epist_uncs, "w", encoding='utf-8') as f:
        json.dump(ensemble_mean_alvs_names_dict, f, indent=4)


def aggregate_and_save_entropies(entropies, path2save_entropies):
    ensemble_uncs = {}
    for net_outputs in entropies.values():
        for alv_name, entropy in net_outputs.items():
            if alv_name not in ensemble_uncs:
                ensemble_uncs[alv_name] = []

            ensemble_uncs[alv_name].append(float(entropy))

    ensemble_mean_alvs_names_dict = {}

    for alv_name, uncs in ensemble_uncs.items():
        ensemble_mean = np.mean(uncs)
        if ensemble_mean not in ensemble_mean_alvs_names_dict:
            ensemble_mean_alvs_names_dict[ensemble_mean] = []
        ensemble_mean_alvs_names_dict[ensemble_mean].append(alv_name)

    with open(path2save_entropies, "w", encoding='utf-8') as f:
        json.dump(ensemble_mean_alvs_names_dict, f, indent=4)


def DDU(test_config, path2save_dir, path2outputs_train, path2outputs_test=None,
        test_on_val_data=False, calc_epistemic=True, calc_aleatoric=True,
        n_components=2, test_set="test"):

    path2save_densities, path2save_epist_uncs, path2save_entropies = \
        prepare_paths(n_components, path2save_dir, test_set)

    densities, entropies = {}, {}
    for net_id in test_config["net_ids"]:
        print(f"Testing net #{net_id}")
        if test_on_val_data:
            test_config["path2test_markup"] = (f"{test_config['path2nets_dir']}/"
                                               f"{test_config['nets_names']}/"
                                               f"{test_config['nets_names']}_{net_id}_"
                                               f"markup_val.json")

        densities[net_id], entropies[net_id] = {}, {}
        epistemic_per_net, entropies_per_net = {}, {}

        if calc_epistemic:
            calc_epistemic_uncs_per_nets(path2outputs_train, net_id, test_config, n_components,
                                         densities, epistemic_per_net, path2save_dir, test_set)

        if calc_aleatoric:
            calc_entropies_per_nets(test_on_val_data, path2outputs_train, path2outputs_test,
                                    test_config, net_id, entropies, entropies_per_net,
                                    path2save_dir, test_set)

    if calc_aleatoric:
        aggregate_and_save_entropies(entropies, path2save_entropies)

    if calc_epistemic:
        with open(path2save_densities, "w", encoding='utf-8') as f:
            json.dump(densities, f, indent=4)

        aggregate_and_save_epistemic_uncs(densities, path2save_epist_uncs)


def parse_args():
    parser = argparse.ArgumentParser("This file allows to calculate uncertainty using DDU method.")
    parser.add_argument('-ctest', '--path2test_config',
                        default=Path('nn/net_configs/test_configs/test_config_classifier.json'),
                        help='configuration file')

    parser.add_argument('-pca', '--pca_components_num', default=10, type=int,
                        help='the number of PCA components')

    parser.add_argument('-tm', '--test_on_val_data', default=False, type=bool,
                        help=('Defines whether the set used for testing is the test or validation. '
                              'Validation is needed to find HUQ parameters'))

    parser.add_argument('-s', '--path2save_dir', type=Path, default=Path('results/ddu'),
                        help=('Path to the dir where uncertainties and '
                              'other output files should be saved.'))

    parser.add_argument('-otrain', '--path2outputs_train', type=Path,
                        default=Path('nn/outputs/MobileNet_small_10_10_80_'
                                     'meta_8_400ep_outputs_train_val.json'),
                        help="Path to the net's outputs on train dataset.")

    parser.add_argument('-otest', '--path2outputs_test', type=Path,
                        default=Path('nn/outputs/test/outputs.json'),
                        help="Path to the net's outputs on test dataset.")

    parsed_args = parser.parse_args()

    with open(parsed_args.path2test_config, "r", encoding='utf-8') as config_file:
        parsed_args.test_config = json.load(config_file)

    return parsed_args


if __name__ == "__main__":
    args = parse_args()

    test_set = "val" if args.test_on_val_data else "test"

    DDU(args.test_config, args.path2save_dir, args.path2outputs_train,
        args.path2outputs_test, test_on_val_data=args.test_on_val_data, calc_epistemic=True,
        calc_aleatoric=True, n_components=args.pca_components_num, test_set=test_set)
