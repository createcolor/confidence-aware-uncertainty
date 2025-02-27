import json
from pathlib import Path
import argparse
from typing import Dict
import numpy as np


def calc_experts_uncertainty(path2markup: Path, path2save_markup: Path = None,
                             mode: str = "dist2integer"):
    '''
    Calculates experts uncertainties from experts votes markup.

    @param mode: sets the formula for the uncertainty, available modes are:
        "dist2integer": the distance from the normalized averaged experts vote
                        to the closest integer (0 or 1);
        "std": the standard deviation of normalized experts votes;
        "entropy": entropy of experts for binary classes with the averaged experts vote
            as probability.
    '''
    with open(path2markup, "r", encoding="utf-8") as f:
        experts_markup = json.load(f)

    unc_alv_name_dict = {}
    for alv_name, alv in experts_markup.items():
        if "expert_votes_positions" in alv:
            if mode == "std":
                unc = np.std(alv["expert_votes_positions"]) / 4

            elif mode == "dist2integer":
                unc = 0.5 - abs(0.5 - np.mean(alv["expert_votes_positions"]) / 4)

            elif mode == "entropy":
                average_vote = np.mean(alv["expert_votes_positions"]) / 4
                unc = 0 if average_vote == 0 or average_vote == 1 else \
                    (-average_vote * np.log2(average_vote) - (1 - average_vote) *
                     np.log2(1 - average_vote)) / 2

            elif mode == "TV":
                mean_exp = np.mean(alv["expert_votes_positions"]) / 4
                unc = mean_exp * (1 - mean_exp)

            else:
                raise ValueError(f"Mode {mode} is not supported.")

            if unc not in unc_alv_name_dict:
                unc_alv_name_dict[unc] = []
            unc_alv_name_dict[unc].append(alv_name)

    if path2save_markup is not None:
        with open(path2save_markup, "w", encoding="utf-8") as f:
            json.dump(unc_alv_name_dict, f, indent=4)

    return unc_alv_name_dict


def calc_ensemble_stds(ensemble_outputs: Path | dict, path2save_markup: Path = None):
    '''
    Downloads ensemble's outputs and builds the dictionary Dict[std (str): List of alvs names]

    @param path2ensemble_outputs: classifiers' outputs in the format:
        {"net_0": {"alv_name_0": 0.1, ...}, ...};
    @param path2save_alvs_std_dict: where to save the obtained markup.
    '''
    if isinstance(ensemble_outputs, dict):
        with open(ensemble_outputs, "r", encoding="utf-8") as f:
            ensemble_outputs_per_nets = json.load(f)
    else:
        ensemble_outputs_per_nets = ensemble_outputs.copy()

    ensemble_outputs = {}
    for net_outputs in ensemble_outputs_per_nets.values():
        for alv_name, output in net_outputs.items():
            if alv_name not in ensemble_outputs:
                ensemble_outputs[alv_name] = []
            ensemble_outputs[alv_name].append(float(output))

    ensemble_std_alvs_names_dict = {}
    alvs_std_dict = {}
    for alv_name, outputs in ensemble_outputs.items():
        ensemble_std = np.std(outputs)
        if ensemble_std not in ensemble_std_alvs_names_dict:
            ensemble_std_alvs_names_dict[ensemble_std] = []
        ensemble_std_alvs_names_dict[ensemble_std].append(alv_name)

        alvs_std_dict[alv_name] = ensemble_std

    if path2save_markup is not None:
        with open(path2save_markup, "w", encoding="utf-8") as f:
            json.dump(ensemble_std_alvs_names_dict, f, indent=4)

    return ensemble_std_alvs_names_dict


def calc_mcmc(path2nets_outputs: Path, unc_dir: Path):
    outputs_files_list = path2nets_outputs.glob("*.json")
    mcmc_ens = {}

    for outputs_file_path in outputs_files_list:
        with open(outputs_file_path, "r", encoding='utf-8') as f:
            cur_outputs = json.load(f)

            for net_i in cur_outputs.keys():
                if net_i not in mcmc_ens:
                    mcmc_ens[net_i] = {outputs_file_path.stem: cur_outputs[net_i]}
                else:
                    mcmc_ens[net_i][outputs_file_path.stem] = cur_outputs[net_i]

    for net_i in mcmc_ens.keys():
        unc_path = unc_dir / ("mcmc_" + net_i + ".json")

        calc_ensemble_stds(mcmc_ens[net_i], unc_path)


def get_experts_unc(path2experts_markup: Path, are_predicted: bool = False,
                    mode: str = "dist2integer", ensemble_mode=True) -> Dict:
    '''
    Either loads predicted experts uncertainties or calculates them
    from the ground truth experts votes.
    '''
    if are_predicted:
        with open(path2experts_markup, "r", encoding="utf-8") as f:
            experts_alvs_unc_dict = json.load(f)
        if ensemble_mode:
            experts_alvs_unc_dict_markup = experts_alvs_unc_dict
            experts_alvs_unc_dict = {}
            for experts_alvs_unc_dict_cur_net in experts_alvs_unc_dict_markup.values():
                for alv_name, unc in experts_alvs_unc_dict_cur_net.items():
                    if alv_name not in experts_alvs_unc_dict:
                        experts_alvs_unc_dict[alv_name] = []
                    experts_alvs_unc_dict[alv_name].append(float(unc))
            experts_alvs_unc_dict = {alv_name: np.mean(uncs) * (1 - np.mean(uncs))
                                     for alv_name, uncs in experts_alvs_unc_dict.items()}
        experts_unc_alvs_dict = convert_alvs_unc2unc_alvs(experts_alvs_unc_dict)
    else:
        experts_unc_alvs_dict = calc_experts_uncertainty(path2experts_markup, mode)

    return experts_unc_alvs_dict


def aggregating_experts_and_nets_uncs(path2ensemble_outputs: Path, are_experts_unc_predicted: bool,
                                      path2experts_markup: Path, path2save_markup: Path,
                                      experts_unc_mode: str = "dist2integer",
                                      aggregation_type: str = "sum", weighting_coef=0.5):
    '''
    Combines experts inconsistency (that models aleatoric uncertatinty) and neural ensemble std
    (~epistemic one) to obtain the total uncertainty.

    @param path2nets_outputs: path to the classification nets ensemble predictions;

    @param are_experts_unc_predicted: if False (experts votes are taken from the ground truth),
    then uncertainty is calculated, otherwise just loaded;

    @param path2experts_markup: if experts uncertainties are predicted it is the path to the nets
    answers, otherwise --- to the ground truth experts votes;

    @param path2save_markup: where to save the aggregated uncertainty;

    @param experts_unc_mode: "std", "entropy" (over classes, where probability is the averaged
    experts vote) or distance between the experts average vote and closest integer ("dist2integer");

    @param aggregation_type: how to aggregate experts uncertainty and ensemble's one:
    "sum" (weighted one), "max" (maximum);

    @param weighting_coef: if aggregation_type is "sum", it is the weight
        of the ensemble uncertainty;
    '''
    with open(path2ensemble_outputs, "r", encoding="utf-8") as f:
        ensemble_outputs_per_nets = json.load(f)

    ens_outputs = {}
    for net_outputs in ensemble_outputs_per_nets.values():
        for alv_name, output in net_outputs.items():
            if alv_name not in ens_outputs:
                ens_outputs[alv_name] = []
            ens_outputs[alv_name].append(float(output))

    ensemble_alvs_unc_dict = {alv_name: np.var(outputs_list) for alv_name, outputs_list
                              in ens_outputs.items()}

    experts_alvs_unc_dict = convert_unc_alvs2alvs_unc(get_experts_unc(
        path2experts_markup, are_experts_unc_predicted, experts_unc_mode))

    total_unc = {}
    for alv_name, ensemble_unc in ensemble_alvs_unc_dict.items():
        if aggregation_type == "max":
            unc = max(ensemble_unc, experts_alvs_unc_dict[alv_name])

        elif aggregation_type == "sum":
            unc = (weighting_coef * ensemble_unc +
                   (1 - weighting_coef) * experts_alvs_unc_dict[alv_name])

        elif aggregation_type == "calib":
            unc = ensemble_unc ** 4 + experts_alvs_unc_dict[alv_name] ** 4

        else:
            raise ValueError(f"Aggregation type {aggregation_type} is not supported.")

        if unc not in total_unc:
            total_unc[unc] = []
        total_unc[unc].append(alv_name)

    if path2save_markup is not None:
        with open(path2save_markup, "w", encoding="utf-8") as f:
            json.dump(total_unc, f, indent=4)

    return total_unc


def convert_alvs_unc2unc_alvs(alvs_unc_markup: Dict):
    '''
    Converts {alv_name: uncertainty} into {uncertainty: [alv_name_1, ...]}
    '''
    uncertainty_alvs_names_dict = {}
    for alv_name, uncertainty in alvs_unc_markup.items():
        uncertainty = float(uncertainty)
        if uncertainty not in uncertainty_alvs_names_dict:
            uncertainty_alvs_names_dict[uncertainty] = []
        uncertainty_alvs_names_dict[uncertainty].append(alv_name)

    return uncertainty_alvs_names_dict


def convert_unc_alvs2alvs_unc(unc_alvs_markup: Dict) -> Dict:
    '''
    Converts {uncertainty: [alv_name_1, ...]}   into {alv_name: uncertainty}.
    '''
    alvs_names_uncertainty_dict = {alv_name: float(uncertainty) for uncertainty, alv_names
                                   in unc_alvs_markup.items() for alv_name in alv_names}

    return alvs_names_uncertainty_dict


def combine_aleat_epist_unc(aleatoric_unc_alvs_dict, epistemic_unc_alvs_dict):
    '''
    Given precalculated aleatoric and epistemic uncertainties,
    sums them up to obtain the total uncertainty.
    '''
    total_alvs_names_dict = {}
    aleatoric_alvs_unc_dict = {}
    for unc, alvs_names in aleatoric_unc_alvs_dict.items():
        for alv_name in alvs_names:
            aleatoric_alvs_unc_dict[alv_name] = unc

    for ep_unc, alvs_names in epistemic_unc_alvs_dict.items():
        for alv_name in alvs_names:
            if alv_name in aleatoric_alvs_unc_dict:
                total_unc = ep_unc + aleatoric_alvs_unc_dict[alv_name]

                if alv_name not in total_alvs_names_dict:
                    total_alvs_names_dict[alv_name] = []
                total_alvs_names_dict[alv_name].append(total_unc)

    return total_alvs_names_dict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-em', '--experts_markup', type=Path,
                        default=Path("markup/BloodyWell/experts_info_per_alvs.json"),
                        help='Path to the experts votes markup (per alvs)')

    parser.add_argument('-ep', '--experts_pred', type=Path,
                        default=Path("/alpha/korchagin/projects/expressto/nn/outputs_experts/"
                                     "MobileNet_small_10_10_80_meta_8_experts_average_"
                                     "700ep_outputs.json"),
                        help='Path to the experts votes markup (per alvs)')

    parser.add_argument('-sd', '--path2save_markup_dir', default=Path("results"), type=Path,
                        help='Path to the directory where to save the obtained uncertainty markups')

    parser.add_argument('-ens', '--path2ensemble_outputs',
                        default=Path('/alpha/korchagin/projects/expressto/nn/outputs/'
                                     'MobileNet_small_10_10_80_meta_8_400ep_outputs.json'),
                        help=("Path to the classifying ensemble outputs: "
                              "{'net_0': {'alv_0': output_0, ...}, ...}"))

    parser.add_argument('-mcmc', '--path2mcmc_outputs',
                        default=Path('results/mcmc_outputs'),
                        help=("Path to the directory with classifying mcmc ensemble "
                              "outputs: {'net_0': {'alv_0': output_0, ...}, ...}"))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.path2save_markup_dir.exists():
        args.path2save_markup_dir.mkdir()

    mcmc_unc_dir = args.path2save_markup_dir / "mcmc_scores"
    if not mcmc_unc_dir.exists():
        mcmc_unc_dir.mkdir()

    calc_mcmc(args.path2mcmc_outputs, mcmc_unc_dir)

    for mode in ["TV"]:
        path2save_markup = args.path2save_markup_dir / f"unc_experts_{mode}.json"
        calc_experts_uncertainty(args.experts_markup, path2save_markup, mode)

        aggregation_type = "sum"
        path2save_markup = f"unc_pred_experts_{mode}_{aggregation_type}_CE_var.json"
        aggregating_experts_and_nets_uncs(args.path2ensemble_outputs,
                                          are_experts_unc_predicted=True,
                                          path2experts_markup=args.experts_pred,
                                          experts_unc_mode=mode,
                                          aggregation_type=aggregation_type,
                                          weighting_coef=0.5,
                                          path2save_markup=(args.path2save_markup_dir /
                                                            path2save_markup))

        path2save_markup = f"unc_gt_experts_{mode}_{aggregation_type}_CE_var.json"
        aggregating_experts_and_nets_uncs(args.path2ensemble_outputs,
                                          are_experts_unc_predicted=False,
                                          path2experts_markup=args.experts_markup,
                                          experts_unc_mode=mode,
                                          aggregation_type=aggregation_type,
                                          weighting_coef=0.5,
                                          path2save_markup=(args.path2save_markup_dir /
                                                            path2save_markup))

    calc_ensemble_stds(args.path2ensemble_outputs, args.path2save_markup_dir / "unc_CE.json")
