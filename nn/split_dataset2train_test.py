import argparse
import json
import numpy as np
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        'Split dataset with alveoluses images into train and test.')
    parser.add_argument('-trm', '--train_markup', default=Path(
        "markup/train_dataset_10_10_80.json"), \
            type=Path, help='Path to train markup')
    parser.add_argument('-tm', '--test_markup', default=Path(
        "markup/test_dataset_10_10_80.json"), \
            type=Path, help='Path to test markup')
    parser.add_argument('-pm', '--new_plates_markup', default=Path(
        "markup/new_markup_by_plates.json"), \
            type=Path, help='Path to test markup')
    parser.add_argument('-im', '--info_markup', default=Path(
        "markup/markup_by_plates.json"), \
            type=Path, help='Path to info markup')
    args = parser.parse_args()
    return args

np.random.seed(0)


def split_by_rows(info_markup, invalid_rows):
    train_markup, test_markup = {}, {}

    for plate_name in info_markup.keys():
        idxs = np.arange(6)
        np.random.shuffle(idxs)
        alvs = info_markup[plate_name]
        banned_rows = []
        
        for alv in alvs:
            row_idx = alv["row_idx"]
            if row_idx in banned_rows:
                alv["invalid"] = True 
                alv["split_type"] = None
            else:
                if ("is_empty" not in alv or alv["is_empty"] == False):
                    if alv["gt_result"] is None:
                        banned_rows.append(row_idx)
                        alv["split_type"] = None
                        continue

                    if plate_name in invalid_rows and row_idx in invalid_rows[plate_name]:
                        alv["invalid"] = True 
                        banned_rows.append(row_idx)
                        alv["split_type"] = None
                        continue
                    else:
                        alv["invalid"] = False
                
                    col_idx = alv["col_idx"]
                    alv_name = str(Path(plate_name).stem + f"_{row_idx}_{col_idx}.png")
                    new_alv = {alv_name: {
                        "gt_result": alv["gt_result"],
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "reagent": alv["reagent"],
                        "cx": alv["cx"],
                        "cy": alv["cy"],
                        "r": alv["r"],
                        "is_empty": False
                    }}
            
                    if row_idx in idxs[:5] and not alv["invalid"]:
                        split_type = "train"
                        new_alv[alv_name]["split_type"] = split_type
                        alv["split_type"] = split_type
                        train_markup.update(new_alv)

                    elif row_idx == idxs[5] and not alv["invalid"]:
                        split_type = "test"
                        new_alv[alv_name]["split_type"] = split_type
                        alv["split_type"] = split_type
                        test_markup.update(new_alv)
    return train_markup, test_markup


def generate_per_alvs_markup(path2save):
    '''
    if args.info_markup --- markup by plates
    '''
    alvs_markup = {}
    with open(args.info_markup, "r") as f:
        plates_markup = json.load(f)

    for plate_name, alvs_list in plates_markup.items():
        for alv in alvs_list:
            row_idx, col_idx = alv["row_idx"], alv["col_idx"]
            alv_name = str(Path(plate_name).stem + f"_{row_idx}_{col_idx}.png")

            alvs_markup[alv_name] = {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "split_type": None
            }

            if "reagent" in alv:
                alvs_markup[alv_name]["reagent"] = alv["reagent"]
                alvs_markup[alv_name]["cx"] = alv["cx"]
                alvs_markup[alv_name]["cy"] = alv["cy"]
                alvs_markup[alv_name]["r"] = alv["r"]
                alvs_markup[alv_name]["is_empty"] = False
                if "gt_result" in alv:
                    alvs_markup[alv_name]["gt_result"] = alv["gt_result"]
                else:
                    alvs_markup[alv_name]["gt_result"] = None
            else:
                alvs_markup[alv_name]["is_empty"] = True
                alvs_markup[alv_name]["gt_result"] = None
            
    
    with open(path2save, "w") as f:
        json.dump(alvs_markup, f, indent=4)

    return alvs_markup


def calc_class_props_per_reagent(info_markup):
    '''
    Calculates classes proportions per reagents

    @returns: reagents_class_props = {reagent: {"0": float, "1": float}}
    '''
    reagents_class_props = {}
    for alv in info_markup.values():
        if ("is_empty" not in alv or alv["is_empty"] == False) and \
            alv["gt_result"] is not None:
            reagent = alv["reagent"]
            if reagent not in reagents_class_props:
                # reagents_class_props[reagent] = {0: 0, 1: 0, "alvs_num": 0}
                reagents_class_props[reagent] = {0: 0, 1: 0}
            
            reagents_class_props[reagent][alv["gt_result"]] += 1

    return reagents_class_props


def split_by_reagents(info_markup, invalid_rows, train_val_prop, test_prop):
    '''
    alvs_num_per_reagent = {"reagent": {"train/val/test": {"0": int, "1": int}}}
    '''
    train_markup, test_markup = {}, {}
    train_val_prop, test_prop = train_val_prop / 100, test_prop / 100
    print(train_val_prop)

    reagents_props = calc_class_props_per_reagent(info_markup)
    print(reagents_props)
    alvs_num_per_reagent = {}

    for reagent, reag_alvs_num in reagents_props.items():
        alvs_num_per_reagent[reagent] = {"train": {}, "test": {}}
        for class_label in [0, 1]:
            alvs_num = reag_alvs_num[class_label]
            if alvs_num <= 15:
                min_alvs_num2train = max(1, alvs_num // 3)
            else: 
                min_alvs_num2train = 0
            
            alvs_num_per_reagent[reagent]["train"][class_label] = \
                max(min_alvs_num2train, int(alvs_num * train_val_prop))
            alvs_num_per_reagent[reagent]["test"][class_label] = alvs_num - \
                alvs_num_per_reagent[reagent]["train"][class_label]
    
    ###
    taken_alvs_num_per_reagent = {reagent: 
        {"train": {0: 0, 1: 0}, "test": {0: 0, 1: 0}} 
        for reagent in reagents_props.keys()}

    for alv_name, alv in info_markup.items():
        if "reagent" in alv and alv["gt_result"] is not None:
            reagent = alv["reagent"]
            row_idx = alv["row_idx"]
            gt_result = alv["gt_result"]
            plate_name = f"{alv_name[:alv_name.rfind('_') - 2]}.png"

            if plate_name in invalid_rows and row_idx in invalid_rows[plate_name]:
                alv["invalid"] = True
                continue
            else:
                alv["invalid"] = False
                    
                if taken_alvs_num_per_reagent[reagent]["train"][gt_result] < \
                        alvs_num_per_reagent[reagent]["train"][gt_result]:
                    alv["split_type"] = "train"
                    train_markup[alv_name] = alv
                    taken_alvs_num_per_reagent[reagent]["train"][gt_result] += 1

                else:
                    alv["split_type"] = "test"
                    test_markup[alv_name] = alv
                    taken_alvs_num_per_reagent[reagent]["test"][gt_result] += 1
    
    print(taken_alvs_num_per_reagent)
    return train_markup, test_markup


if __name__ == "__main__":
    args = parse_args()
    train_val_prop, test_prop = 20, 80

    invalid_rows = {"2022.04.14_001_map.png": [0, 1, 2, 3, 4, 5], 
                   "2022.04.14_004_map.png": [1], 
                   "2022.04.14_005_map.png": [0]}
    
    assert train_val_prop + test_prop == 100

    with open("markup/markup_by_alvs.json", "r") as f:
        info_markup = json.load(f)

    train_markup, test_markup = split_by_reagents(info_markup, invalid_rows,
                                                  train_val_prop, test_prop)

    # with open(args.train_markup, "w") as f:
    #     json.dump(train_markup, f, indent=4)
    # with open(args.test_markup, "w") as f:
    #     json.dump(test_markup, f, indent=4)
