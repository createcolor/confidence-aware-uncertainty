import os
import json
from pathlib import Path
import argparse
from typing import Dict, List
from itertools import chain, combinations
import numpy as np
import matplotlib.pyplot as plt


def get_rc_info_averaging_mode(unc_alv_names_dicts: List[Dict], test_markup: Dict, n_bins: int = 20,
                               reagent: str = "all", misclassified_alvs=None,
                               ensemble_classifier_mode=False, path2save_markup: Path = None,
                               net_ids=range(10)):
    '''
    Given uncertainties and list of classificator mistakes and test markups, obtains info
    to plot rejection curves. If net_ids contains more than one id, rejection curves
    for the corresponding classifying networks are averaged along y axis.

    @param unc_alv_names_dicts: [{uncertainty: [alv_names]}, {...}, ...];

    @param test_markup: {alv_name: alv_info} --- test dataset;

    @param net_ids: ids of classifying neural networks;

    @param n_bins: x axis (throwaway rate) bins for averaging curves. For example,
    if n_bins is equal to 5, y axis values for curves (accuracies) are calculated
    at the following x-axis points: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0;

    @param reagent: only test wells with these reagents are considered;

    @param misclassified_alvs: list of classifier's mistakes;

    @param ensemble_classifier_mode: if is True, then curves are not averaged
    along y-axis because an ensemble is considered a single classifier.

    @param path2save_markup: where to save rejection curves info accuracies
    for the corresponding throwaway rates.
    '''
    x_list = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ensemble_accuracies_dict = {x: [] for x in x_list}

    for net_id in net_ids:
        if len(unc_alv_names_dicts) > 1:
            unc_alv_names_dict = unc_alv_names_dicts[net_id]
        else:
            unc_alv_names_dict = unc_alv_names_dicts[0]

        if not ensemble_classifier_mode:
            misclas_alvs = misclassified_alvs[str(net_id)]
        else:
            misclas_alvs = misclassified_alvs

        accuracies, throwaway_rates = get_accs_throwaway_rates(
            unc_alv_names_dict, test_markup, misclas_alvs, reagent=reagent)

        for x in x_list:
            if x in throwaway_rates:
                ensemble_accuracies_dict[x].append(accuracies[throwaway_rates.index(x)])
            else:
                throwaway_rates_less_x = list(filter(lambda rate: rate < x, throwaway_rates))
                left_closest_rate = min(throwaway_rates_less_x, key=lambda rate: x - rate)
                right_closest_rate = throwaway_rates[throwaway_rates.index(left_closest_rate) + 1]
                alpha = (x - left_closest_rate) / (right_closest_rate - left_closest_rate)

                accuracy1 = accuracies[throwaway_rates.index(left_closest_rate)]
                accuracy2 = accuracies[throwaway_rates.index(right_closest_rate)]
                accuracy = accuracy1 + alpha * (accuracy2 - accuracy1)

                ensemble_accuracies_dict[x].append(accuracy)

    ensemble_accuracies = {x: (np.mean(ensemble_accuracies_dict[x]),
                           np.std(ensemble_accuracies_dict[x])) for x in x_list}

    if path2save_markup is not None:
        with open(path2save_markup, "w", encoding="utf-8") as ensemble_file:
            json.dump(ensemble_accuracies, ensemble_file, indent=4)

    return ensemble_accuracies


def calculate_accuracy(misclas_alvs_names: List[str], test_alvs_names: List[str]):
    '''
    Calculates accuracy given two lists of alveoluses' names:
    one where the classification model makes mistakes (misclas_alvs_names)
    and one with all the alveoluses to be tested (test_alvs_names).
    '''
    assert len(test_alvs_names) != 0
    mistakes_num = len(list(set(misclas_alvs_names) & set(test_alvs_names)))
    return 1.0 - mistakes_num / len(test_alvs_names)


def get_accs_throwaway_rates(uncert_alv_name_dict: Dict, input_test_markup: Dict,
                             misclas_alvs_names: List, reagent: str = "all"):
    '''
    Calculates data for rejective curves --- accuracies depending on throwaway rates.
    Throwaway rate is the part of the test set being thrown away --- the higher the uncertainty,
    the earlier the alveolus is thrown away. Accuracies are calculated
        on the rest part of the test set.

    @param uncert_alv_name_dict: the dictionary where keys are uncrtainty values,
        values are the lists of the alveoluses having corresponding uncertainties;

    @param test_markup: {alv_name: alv_info} --- test dataset;

    @param misclassified_alvs: list of classifier's mistakes;

    @param reagent: only test wells with these reagents are considered.
    '''
    test_markup = input_test_markup.copy()

    if reagent != "all":
        # leave in test_markup uncert_alv_name_dict only alvs with the required reagent:
        test_markup = {alv_name: alv_info for alv_name, alv_info in test_markup.items()
                       if alv_info["reagent"] == reagent}

        reagent_uncert_alv_name_dict = {}
        for unc, alv_names_list in uncert_alv_name_dict.items():
            for alv_name in alv_names_list:
                if alv_name in test_markup.keys():

                    if unc not in reagent_uncert_alv_name_dict:
                        reagent_uncert_alv_name_dict[unc] = []

                    reagent_uncert_alv_name_dict[unc].append(alv_name)

        uncert_alv_name_dict = reagent_uncert_alv_name_dict

    assert len(test_markup) > 0
    initial_acc = calculate_accuracy(misclas_alvs_names, list(test_markup.keys()))
    accuracies, throwaway_rates = [initial_acc], [0.0]
    test_markup_left = test_markup.copy()
    init_alvs_number = len(test_markup)

    uncert_alv_name_dict_rounded = {}
    for rate in uncert_alv_name_dict:
        rounded_rate = round(float(rate), 7)
        if rounded_rate not in uncert_alv_name_dict_rounded:
            uncert_alv_name_dict_rounded[rounded_rate] = []
        uncert_alv_name_dict_rounded[rounded_rate] += uncert_alv_name_dict[rate]

    sorted_uncertainty_rates = sorted(list(map(float, uncert_alv_name_dict_rounded.keys())),
                                      reverse=True)

    throwaway_rate = 0.0

    for uncertainty_rate in sorted_uncertainty_rates:
        alvs2throwaway_names_candidates = uncert_alv_name_dict_rounded[uncertainty_rate]
        alvs2throwaway_names_list = [alv_name for alv_name in alvs2throwaway_names_candidates
                                     if alv_name in test_markup_left]
        if len(alvs2throwaway_names_list) != 0:
            for alv_name in alvs2throwaway_names_list:
                del test_markup_left[alv_name]

            if len(test_markup_left) > 0:
                throwaway_rate += len(alvs2throwaway_names_list) / init_alvs_number
                throwaway_rates.append(throwaway_rate)

                accuracy = calculate_accuracy(misclas_alvs_names, test_markup_left)
                accuracies.append(accuracy)

                if accuracy == 1.0:
                    break

    accuracies.append(1.0)
    throwaway_rates.append(1.0)

    return accuracies, throwaway_rates


def plot_rejection_curves(test_markup: Dict, misclas_alvs_names: List[str],
                          uncertainty_types: Dict, plot_path: Path = None,
                          title: str = "", reagent="all"):
    '''
    Plots accuracy-rejection curves (dependence of accuracies on the left test set part
    after the most uncertain alveoluses are thrown away).
    '''
    plt.clf()
    plt.grid()
    plt.xlabel('Throwaway rate')
    plt.ylabel('Accuracy')
    plt.title(title)

    accuracies_dict, throwaway_rates_dict = {}, {}
    for label, uncert_alv_name_dict in uncertainty_types.items():
        accuracies, throwaway_rates = get_accs_throwaway_rates(
            uncert_alv_name_dict, test_markup, misclas_alvs_names, reagent=reagent)
        plt.plot(throwaway_rates, accuracies, label=label)
        accuracies_dict[label], throwaway_rates_dict[label] = accuracies, throwaway_rates

    plt.legend()
    if plot_path is not None:
        plt.savefig(plot_path)

    return accuracies_dict, throwaway_rates_dict


def get_accs_rejection_rates(uncert_alv_name_dict: Dict, test_markup: Dict,
                             misclas_alvs_names: List[str], rejection_step: float = 0.1):
    '''
    Prepares rejection rates for plotting accuracy-rejection curves
    where rejection rates are put along x-axis.
    Rejection rate is the threshold for uncertainty:
    alvs are thrown away from the test set if their uncertainty is higher than the threshold.
    '''
    def normalize_uncertainties(uncert_alv_name_dict: Dict):
        '''
        Is used for accuracy-rejection curves, where rejetion rates (not throwaway rates)
        are put along x-axis. This approach is not used currently.
        '''
        uncertainties = list(uncert_alv_name_dict.keys())
        max_uncert = float(max(uncertainties))
        min_uncert = float(min(uncertainties))

        uncert_alv_name_dict = {(float(k) - min_uncert) / max_uncert: v
                                for k, v in uncert_alv_name_dict.items()}
        return uncert_alv_name_dict

    uncert_alv_name_dict = normalize_uncertainties(uncert_alv_name_dict)
    test_markup_left = test_markup.copy()
    rejection_rates = np.arange(1.0, -rejection_step, -rejection_step)

    y = []
    for rejection_rate in rejection_rates:
        for uncert, alv_names in uncert_alv_name_dict.items():
            if uncert > rejection_rate:
                for alv_name in alv_names:
                    if alv_name in test_markup_left:
                        del test_markup_left[alv_name]

        if len(test_markup_left) > 0:
            y.append(calculate_accuracy(misclas_alvs_names, test_markup_left))

    return y, rejection_rates


def calculate_aucs(accuracies_dict: Dict, throwaway_rates_dict: Dict, min_accuracy: float = None):
    '''
    Calculates areas between the accuracy-rejection curve and the bottom horizontal line
    given by min_accuracy.

    @param accuracies_dict: {unc estimation method name: accuracies_list}
    @param throwaway_rates_dict: {unc estimation method name: throwaway_rates_list (x axis values)}
    '''
    def normalize_accuracies(accuracies: List[float], min_acc: float):
        '''
        Fits accuracies to [0, 1] range (for convinient AUC calculation).
        '''
        return [(accuracy - min_acc) / (1.0 - min_acc) for accuracy in accuracies]

    aucs = {}
    if min_accuracy is None:
        min_accuracy = min(list(chain(accuracies_dict.values())))

    for method_name, accuracies in accuracies_dict.items():
        accuracies = normalize_accuracies(accuracies, min_accuracy)
        aucs[method_name] = round(np.trapz(y=accuracies, x=throwaway_rates_dict[method_name]), 6)
        print(f"Area under {method_name} curve: {aucs[method_name]}")

    return aucs


# for advanced research #
def get_accuracies_per_bins(unc_alvs_dict, misclas_alvs_names, bins_num=50):
    positive_uncertainties = sorted([unc for unc in unc_alvs_dict.keys()
                                     if float(unc) > 0], reverse=True)
    pos_uncs_alvs = []
    alvs_pos_uncs_dict = {}
    for unc in positive_uncertainties:
        alv_names_list = unc_alvs_dict[unc]
        for alv_name in alv_names_list:
            pos_uncs_alvs.append(alv_name)
            alvs_pos_uncs_dict[alv_name] = unc

    alvs_num_per_bin = len(pos_uncs_alvs) / bins_num
    cur_bin_alvs = []
    accuracies = []
    mean_uncs = []
    for alv in pos_uncs_alvs:
        if len(cur_bin_alvs) >= alvs_num_per_bin:
            accuracies.append(calculate_accuracy(misclas_alvs_names, cur_bin_alvs))
            mean_uncs.append(np.mean(np.asarray([alvs_pos_uncs_dict[alv_name]
                                                 for alv_name in cur_bin_alvs])))
            cur_bin_alvs = []
        cur_bin_alvs.append(alv)

    mean_uncs.append(0.0)

    zero_unc_alvs = [alv_name for unc, alvs_list in unc_alvs_dict.items() for alv_name in alvs_list
                     if float(unc) == 0]
    accuracies.append(calculate_accuracy(misclas_alvs_names, zero_unc_alvs))

    return accuracies, mean_uncs


def data4multiple_dataset_and_experts_combinations(path2misclas_alvs, unc_alv_names_dict,
                                                   test_markup, path2save_markup=None):
    dataset_parts = [(i + 1) / 10 for i in range(10)]

    for used_experts_num in range(1, 7):
        aggregated_markup = {}
        used_experts_ids = list(combinations(range(6), used_experts_num))
        for used_experts in used_experts_ids:
            used_experts = ''.join([str(exp) for exp in used_experts])

            for dataset_part in dataset_parts:
                print(f"Running dataset_part={dataset_part}_experts={used_experts}")
                with open(path2misclas_alvs, "r", encoding='utf-8') as misclassified_file:
                    misclassified_alvs = json.load(misclassified_file)
                uncs_markup = get_rc_info_averaging_mode(unc_alv_names_dict, test_markup,
                                                         misclassified_alvs=misclassified_alvs,
                                                         path2save_markup=path2save_markup)

                for x, unc in uncs_markup.items():
                    if x not in aggregated_markup:
                        aggregated_markup[x] = [[], []]
                    for i in range(2):
                        aggregated_markup[x][i].append(unc[i])

        aggregated_markup = {x: [np.mean(uncs[0]), np.mean(uncs[1])] for x, uncs
                             in aggregated_markup.items()}
        with open(path2save_markup, "w", encoding='utf-8') as f:
            json.dump(aggregated_markup, f, indent=4)
#####################################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tm', '--test_markup',
                        default=Path("markup/BloodyWell/test_dataset_10_10_80.json"),
                        type=Path, help='Path to test dataset markup')
    parser.add_argument('-ds', '--dir2save', default=Path("results/rejection_curves"),
                        type=Path, help='Path to the directory where to save markups')
    parser.add_argument('-i', '--config',
                        default=Path("uncertain_classification/rejection_curves_args.json"),
                        type=Path, help=('Path to the json file with uncertainty '
                                         'estimation methods info'))
    parser.add_argument('-nn', '--net_num', default=10, type=int,
                        help='The number of nets in the ensemble')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.dir2save.exists():
        os.makedirs(args.dir2save)

    with open(args.test_markup, "r", encoding='utf-8') as f:
        test_markup = json.load(f)

    with open(args.config, "r", encoding='utf-8') as f:
        config = json.load(f)

    with open(config["path2misclassified_alvs"], "r", encoding='utf-8') as f:
        misclassified_alvs = json.load(f)

    for method, info in config["UE_methods"].items():
        path2save_reg_curves_info = f"{args.dir2save}/{info['path2save_reg_curves_info']}"
        unc_alv_names_dicts = []

        if info['single_net_file_name_template'] is None:
            with open(info['path2uncertainties'], "r", encoding='utf-8') as f:
                unc_alv_names_dicts.append(json.load(f))
        else:
            for net_id in range(args.net_num):
                with open(f"{info['path2uncertainties']}/{info['single_net_file_name_template']}"
                          f"_{net_id}.json", "r", encoding='utf-8') as f:
                    unc_alv_names_dicts.append(json.load(f))

        get_rc_info_averaging_mode(unc_alv_names_dicts, test_markup, n_bins=1000,
                                   misclassified_alvs=misclassified_alvs,
                                   path2save_markup=path2save_reg_curves_info,
                                   net_ids=range(args.net_num))
