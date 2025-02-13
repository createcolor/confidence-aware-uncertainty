import argparse
import json
from pathlib import Path

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mk', '--markup', default=Path('markup/experts_info_per_alvs.json'))
    parser.add_argument('-n', '--name', default="average_expert_per_alvs.json")
    parser.add_argument('-t', '--train_markup', default=None)
    parser.add_argument('-kv', '--keep_experts_votes', type=bool, default=False)

    return parser.parse_args()

def average_experts(exps_markup, onehot_class=False, keep_experts_votes=False):
    avgexp_markup = {}

    for key in exps_markup.keys():
        if not exps_markup[key]["is_empty"] and isinstance(exps_markup[key]["gt_result"], int):
            avgexp_markup[key] = exps_markup[key]

            avg_expert = np.mean(exps_markup[key]["expert_votes_positions"])
            avgexp_markup[key]["average_expert"] = avg_expert / 4

            if onehot_class:
                onehot = np.zeros(5)
                onehot[round(avg_expert)] = 1
                avgexp_markup[key]["expert_class"] = list(onehot)
            else:
                avgexp_markup[key]["expert_class"] = round(avg_expert)

            if not keep_experts_votes:
                avgexp_markup[key].pop("expert_votes_numbers", None)
                avgexp_markup[key].pop("expert_votes_positions", None)

    return avgexp_markup

def separate_train_experts(exps_markup, train_markup):
    exps_train_markup = {}

    for key in train_markup.keys():
        exps_train_markup[key] = exps_markup[key]

    return exps_train_markup

if __name__ == '__main__':
    args = parse_args()

    with open(args.markup, "r", encoding="utf-8") as f:
        exps_markup = json.load(f)

    avgexp_markup = average_experts(exps_markup=exps_markup, keep_experts_votes=args.keep_experts_votes)
    if args.train_markup is not None:
        with open(args.train_markup, "r", encoding="utf-8") as f:
            train_markup = json.load(f)

        avgexp_markup = separate_train_experts(avgexp_markup, train_markup)

    with open(args.markup.parent / args.name, "w", encoding="utf-8") as f:
        json.dump(avgexp_markup, f, ensure_ascii=False, indent=4)
        