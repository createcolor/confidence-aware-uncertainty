from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
import cv2

from scipy import ndimage

from scipy import ndimage

import imgaug as imgaug
import imgaug.augmenters as iaa
 
class DatasetGenerator(Dataset):
    
    def __init__(self, root_dir, markup, images, transform, config, augment=False, random_crop=False,
                 experts_markup=None, reagents_list=None):
        self.root_dir = Path(root_dir)
        self.markup = markup
        self.transform = transform
        self.augment = augment
        if self.augment:
            self.config_aug = config["augmentations"]
        self.config_info = config

        self.img_side_size = self.config_info.get("dataset_info", {}).get("img_side_size", 512)
        
        self.random_crop = random_crop
        self._initialize_augmentations(augment)

        self.experts_markup = experts_markup
        self.reagents_list = reagents_list

        self.markup_keys = list(self.markup.keys())
        self.markup_values = list(self.markup.values())

        self.images = images

    def _initialize_augmentations(self, augment):
        auglist = []

        if augment:
            if self.random_crop:
                self.alv_rad_percent = self.config_aug["crop"]["alv_radius_range"]
                self.h_shift = self.config_aug["crop"]["horizshift_range"]
                self.v_shift = self.config_aug["crop"]["vertshift_range"]

            self.grouped_keys = [[], []]
            for k, v in self.markup.items():
                self.grouped_keys[v["gt_result"]].append(k)
            
            if "flip" in self.config_aug:
                auglist.append(iaa.Fliplr(self.config_aug["flip"]))
                
            if "rotate" in self.config_aug:
                auglist.append(iaa.Rotate(tuple(self.config_aug["rotate"])))

            if "noise" in self.config_aug:
                noise = iaa.Sometimes(self.config_aug["noise"][0],
                    iaa.AdditiveGaussianNoise(scale=self.config_aug["noise"][1:2]))
                auglist.append(noise)

            if "cutmix" in self.config_aug:
                self.cutmix = self.config_aug["cutmix"]

            if "mixup" in self.config_aug:
                self.mixup = self.config_aug["mixup"]

                if not hasattr(self, 'cutmix'):
                    self.cutmix = {"probability": 0}

            if "jigsaw" in self.config_aug:
                jigsaw = iaa.Sometimes(self.config_aug["jigsaw"]["probability"], \
                    iaa.Jigsaw(nb_rows=self.config_aug["jigsaw"]["x"], nb_cols=self.config_aug["jigsaw"]["y"])) 
                auglist.append(jigsaw)
            
            if "pie" in self.config_aug:
                self.pie = self.config_aug["pie"]

            if "contrast" in self.config_aug:
                contrast = iaa.Sometimes(self.config_aug["contrast"]["probability"], \
                    iaa.LinearContrast((self.config_aug["contrast"]["min"], self.config_aug["contrast"]["max"])))
                auglist.append(contrast)

            if "sharpen" in self.config_aug:
                sharpen = iaa.Sometimes(self.config_aug["sharpen"]["probability"], \
                    iaa.Sharpen(alpha = tuple(self.config_aug["sharpen"]["alpha"]), \
                    lightness=tuple(self.config_aug["sharpen"]["light"])))
                auglist.append(sharpen)

            if "affine" in self.config_aug:
                affine = iaa.Sometimes(self.config_aug["affine"]["probability"],
                    iaa.Affine( scale={"x": tuple(self.config_aug["affine"]["xscale"]), 
                                       "y": tuple(self.config_aug["affine"]["yscale"])},
                    shear=tuple(self.config_aug["affine"]["shear"]), order=[0, 1],
                    cval=0, mode=imgaug.ALL))
                auglist.append(affine)

            if "perspective" in self.config_aug:
                perspective = iaa.Sometimes(self.config_aug["perspective"]["probability"],
                    iaa.PerspectiveTransform(scale=self.config_aug["perspective"]["scale"]))
                auglist.append(perspective)

            if "cutout" in self.config_aug:
                cutout = iaa.Sometimes(self.config_aug["cutout"]["probability"],
                                       iaa.Cutout(fill_mode="constant", cval=(0, 255),
                                                  nb_iterations=tuple(self.config_aug["cutout"]["count"]),
                                                  size=self.config_aug["cutout"]["size"], squared=False))
                auglist.append(cutout)
        
        self.aug = iaa.Sequential(auglist, random_order=True)
        
    def __len__(self):
        return len(self.markup.keys())

    def _get_sector_mask(self, image, angle, offset):
        x_size = image.shape[1]
        y_size = image.shape[0]

        mask_x = range(x_size)
        mask_x = np.tile(mask_x, (y_size, 1))
        mask_y = np.array([range(y_size)]).transpose()
        mask_y = np.tile(mask_y, (1, x_size))

        right_border = offset + angle
        if offset >= 180:
            offset -= 360
            right_border -= 360
        cond = (np.arctan2(mask_y - y_size/2, mask_x - x_size/2) * 180 / np.pi <= offset) | \
                                (np.arctan2(mask_y - y_size/2, mask_x - x_size/2) * 180 / np.pi >= right_border)
        
        if right_border > 180 and offset < 180:
            right_border -= 360
            cond = (np.arctan2(mask_y - y_size/2, mask_x - x_size/2) * 180 / np.pi <= offset) & \
                                (np.arctan2(mask_y - y_size/2, mask_x - x_size/2) * 180 / np.pi >= right_border)
            
        return np.where(~cond)

    def _get_mask(self, image, alv_r_percent, h_shift, v_shift):
        x_size = image.shape[1]
        y_size = image.shape[0]

        r = int(alv_r_percent * min(x_size, y_size) / 2)
        mask_x = range(x_size)
        mask_x = np.tile(mask_x, (y_size, 1))
        mask_y = np.array([range(y_size)]).transpose()
        mask_y = np.tile(mask_y, (1, x_size))

        c0 = y_size / 2 + v_shift
        c1 = x_size / 2 + h_shift
        mask_ids = np.where(((mask_x - c1)**2 + \
                            (mask_y - c0)**2 > r**2) |
                            (mask_x >= x_size) | 
                            (mask_y >= y_size))
        return mask_ids
    
    def _crop(self, image, alv_r_percent, h_shift, v_shift):
        img_height, img_width = image.shape[0], image.shape[1]      
        alvdiam = int(alv_r_percent * min(img_height, img_width))
        alvdiam = alvdiam - alvdiam % 2
        top = max((img_height - alvdiam) // 2 + v_shift, 0)
        bottom = min(top + alvdiam, img_height-1)
        left = max((img_height - alvdiam) // 2 + h_shift, 0)
        right = min(left+alvdiam, img_width-1)
        return image[top:bottom, left:right]


    def _mixup(self, img, gt_res):
        image_second = self._choose_image(1-gt_res, img.shape[0:2])
        mixup_coef = np.random.uniform(self.mixup["percent_range"][0],\
                                        self.mixup["percent_range"][1])
        gt_res = mixup_coef * gt_res + (1 - mixup_coef) * (1 - gt_res)
        img = self._mixup(img, image_second, mixup_coef)
        return np.array((mixup_coef * img + (1 - mixup_coef) * image_second), dtype=np.uint8), gt_res

    def _cutmix(self, img, gt_res):
        image_second = self._choose_image(1-gt_res, img.shape[0:2])
        cut_rad = np.random.uniform(self.cutmix["radius_range"][0],\
                                    self.cutmix["radius_range"][1])
        cut_dist = np.random.uniform(self.cutmix["dist_range"][0],\
                                    self.cutmix["dist_range"][1])
        cut_angle = np.random.uniform(0, 2*np.pi)
        gt_res = gt_res * (1 - cut_rad**2) + (1 - gt_res) * cut_rad**2
        mask = self._get_mask(img, cut_rad, cut_dist*img.shape[0]*np.cos(cut_angle),\
                                                cut_dist*img.shape[0]*np.sin(cut_angle))
        img[mask] = image_second[mask]

        return img

    def _pie(self, img):
        new_img = np.zeros(img.shape, dtype=img.dtype)
        angle = 360 / self.pie["count"]
        offset = np.random.uniform(self.pie["offset"][0],\
                                    self.pie["offset"][1])

        pie_angles = [offset + angle*k for k in range(self.pie["count"])]
        np.random.shuffle(pie_angles)
        offset = np.random.uniform(self.pie["put_offset"][0],\
                                    self.pie["put_offset"][1])
        src_pos = 0
        for num in range(self.pie["count"]):
            img = ndimage.rotate(img, pie_angles[num] - src_pos, reshape=False)
            mask = self._get_sector_mask(img, angle, offset+angle*num)
            new_img[mask] = img[mask]
            src_pos = pie_angles[num]

        return new_img

    def _choose_image(self, image_gt, image_shape):
        opposite_index = np.random.randint(0, len(self.grouped_keys[image_gt]))
        key_second = self.grouped_keys[image_gt][opposite_index]
        index_second = self.markup_keys.index(key_second)
        image_second = cv2.resize(self.images[index_second], image_shape)
        return image_second

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        img_name = self.markup_keys[idx]
        img = np.copy(self.images[idx])

        gt_res = self.markup_values[idx]["gt_result"]

        h_shift = np.random.randint (self.h_shift[0], self.h_shift[1]) if self.random_crop else 0
        v_shift = np.random.randint (self.v_shift[0], self.v_shift[1]) if self.random_crop else 0
        alvrad_rel = np.random.uniform (self.alv_rad_percent[0], self.alv_rad_percent[1]) \
                     if self.random_crop else 0.95

        mask = self._get_mask(img, alvrad_rel, h_shift, v_shift)
        img[mask] = 0

        if self.random_crop:
            img = self._crop(img, alvrad_rel, h_shift, v_shift)

        mixup_banned = False
        if self.augment and hasattr(self, 'cutmix'):
            if np.random.uniform(0, 1) < self.cutmix["probability"]:
                mixup_banned = True
                img, gt_res = self._cutmix(img, gt_res)

        if self.augment and hasattr(self, 'mixup') and not mixup_banned:
            if np.random.uniform(0, 1-self.cutmix["probability"]) < self.mixup["probability"]:
                img, gt_res = self._mixup(img, gt_res)

        if self.augment and hasattr(self, 'pie'):
            if np.random.uniform(0, 1) < self.pie["probability"]:
                img = self._pie(img)

        if self.augment:
            img = self.aug(images = [img])[0]

        img = cv2.resize(img, (self.img_side_size, self.img_side_size))

        sample["image"] = img 

        sample["name"] = img_name
        sample["agg_type"] = np.array(gt_res, dtype = np.double)
        sample["reagent"] = self.markup_values[idx]["reagent"]

        if self.config_info.get("experts_mode", False):
            if self.experts_markup is not None and "expert_votes_positions" in self.experts_markup[img_name]:
                experts_mean = np.mean(self.experts_markup[img_name]["expert_votes_positions"]) / 4
                sample["experts"] = np.array(experts_mean, dtype=np.float32)
                if self.config_info["experts_mode"] == "std":
                    sample["experts_std"] = np.array(np.std(self.experts_markup[img_name]["expert_votes_positions"]) / 4, dtype=np.float32)
                elif self.config_info["experts_mode"] == "dist2integer":
                    sample["experts_dist2integer"] = np.array(0.5 - abs(0.5 - experts_mean), dtype=np.float32)
                elif self.config_info["experts_mode"] == "entropy":
                    sample["experts_entropy"] = np.array(0 if experts_mean == 0 or experts_mean == 1 \
                        else -experts_mean * np.log(experts_mean) - (1 - experts_mean) * \
                        np.log(1 - experts_mean), dtype=np.float32)
                elif self.config_info["experts_mode"] == "average_vote":
                    sample["experts_average_vote"] = sample["experts"]
                elif self.config_info["experts_mode"] == "class":
                    sample["experts_class"] = self.experts_markup[img_name]["expert_class"]

        
        if self.reagents_list is not None:
            sample["meta_reagent_type"] = np.zeros(len(self.reagents_list), dtype = np.float32)
            sample["meta_reagent_type"][self.reagents_list.index(self.markup_values[idx]["reagent"])] = 1.0

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ret_dict = {'image': torch.from_numpy(sample['image'].transpose((2, 0, 1))),
                    'agg_type': torch.from_numpy(sample['agg_type'])}
        
        for item in ['name', 'reagent', "experts", "experts_std", "meta_reagent_type", \
                     "experts_dist2integer", "experts_entropy", "experts_average_vote", "experts_class"]:
            if item in sample:
                ret_dict[item] = sample[item]

        return ret_dict
    
class NormalizeImg(object):
    """Normalize img in sample by max 1"""

    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        image /= image.max()
        ret_dict = {'image': image}
        
        for item in ['agg_type', 'name', 'reagent', "experts", "experts_std", "meta_reagent_type", \
                     "experts_dist2integer", "experts_entropy", "experts_average_vote"]:
            if item in sample:
                ret_dict[item] = sample[item]

        return ret_dict
    
class NormalizeImg(object):
    """Normalize img in sample by max 1"""

class Normalize(object):
    """Normalize ndarrays in sample"""
    def __init__(self, mean, std, inplace=False):
        self.norm_func = T.Normalize(mean, std, inplace)

    def __call__(self, sample):
        ret_dict = {'image': self.norm_func(sample["image"])}
        
        for item in ['agg_type', 'name', 'reagent', "experts", "experts_std", "meta_reagent_type", \
                     "experts_dist2integer", "experts_entropy", "experts_average_vote"]:
            if item in sample:
                ret_dict[item] = sample[item]

        return ret_dict
        
