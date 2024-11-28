import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from utils.metrics.SVLS_2D import get_svls_filter_2d

class LIDCDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None, gt_mode: str='expert1', augment_prob: float=0., keep_names=False):
        self.data_dir = data_dir
        self.transform = transform
        self.gt_mode = gt_mode
        self.augment_prob = augment_prob
        self.keep_names = keep_names

        self.data = pickle.load(open(self.data_dir, 'rb'))
        self.data_idx = list(self.data.keys())

        if self.gt_mode == 'mSVLS':
            self.filter = get_svls_filter_2d(ksize=3, sigma=1., channels=1)

    def __len__(self):
        return len(self.data)
    
    def __getkeyitem__(self, key):
        obj = self.data[key]
        image = obj["image"]
        masks = np.array(obj["masks"], dtype='int')

        if self.gt_mode == 'binary':
            labels = np.round(np.mean(masks, axis=0))
        elif self.gt_mode[:-1] == 'expert':
            expert_id = int(self.gt_mode[-1])
            labels = masks[expert_id]
        elif self.gt_mode == "mOH":
            labels = np.mean(masks, axis=0)
        elif self.gt_mode == "mSVLS":
            masks = torch.Tensor(np.mean(masks, axis=0)).unsqueeze(dim=0)
            labels = self.filter(masks)
        elif self.gt_mode == "count":
            labels = np.sum(masks, axis=0)
        elif self.gt_mode == "all":
            labels = masks
        else:
            raise ValueError(f"gt_mode '{self.gt_mode}' not supported.")

        if self.transform:
            image = self.transform(image)
            labels = self.transform(labels)
        
        image = torch.unsqueeze(torch.Tensor(image), 0)
        if not self.gt_mode == "mSVLS":
            labels = torch.unsqueeze(torch.Tensor(labels), 0)

        if random.random() < self.augment_prob:
            transpose = random.sample(range(1, image.dim()), 2)
            flip = random.randint(2, image.dim() - 1)
            
            image = image.transpose(*transpose).flip(flip)
            labels = labels.transpose(*transpose).flip(flip)

        if self.gt_mode == "count":
            labels = torch.squeeze(labels).long()

        if self.keep_names:
            return image, labels, key
        return image, labels

    def __getitem__(self, idx):
        key = self.data_idx[idx]
        return self.__getkeyitem__(key)
    
    def get_key_list(self):
        return self.data_idx
    
class RIGADataset(Dataset):
    def __init__(self, data_dir: Path, ground_truth: str, sets: list=['BinRushed', 'Magrabia', 'MESSIDOR'], 
                 augment: bool=True, keep_names: bool=False):
        self.data_dir = data_dir
        self.ground_truth = ground_truth
        self.sets = sets
        self.augment = augment
        self.names = keep_names
        self.resize = Resize([256, 256])
        self.images, self.labels, self.keys = self._load_data()

    def _load_file(self, file: str | Path) -> tuple[np.array, list]:
        image = plt.imread(file)

        img_labels = []
        for i in range(6):
            label_file = Path(str(file).replace("/Image/", f"/Rater{i + 1}/"))
            img_labels.append(plt.imread(label_file))

        return image, img_labels

    def _load_data(self) -> tuple[list, list, list]:
        images = []
        labels = []
        keys = []

        for subset in self.sets:
            print(f"Loading {subset}:")
            if subset == 'MESSIDOR':
                for file in tqdm(list((Path(self.data_dir) / "Image" / subset).iterdir())):
                    if file.suffix == '.tif':
                        image, img_labels = self._load_file(file)
                        images.append(image)
                        labels.append(img_labels)
                        keys.append(file)
            else:
                for subdir in tqdm(list((Path(self.data_dir) / "Image" / subset).iterdir())):
                    if subdir.is_dir():
                        for file in subdir.iterdir():
                            if file.suffix == '.tif':
                                image, img_labels = self._load_file(file)
                                images.append(image)
                                labels.append(img_labels)
                                keys.append(file)

        return images, labels, keys
    
    def _augment_data(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        id = image.dim()
        transpose = random.sample(range(id - 2, id), 2)
        flip = random.randint(id - 2, id - 1)
        
        image = image.transpose(*transpose).flip(flip)
        label = label.transpose(*transpose).flip(flip)

        return image, label

    def _get_labels(self, label: list) -> torch.Tensor:
        label_t = []
        for expert in label:
            expert = torch.tensor(np.rollaxis(expert, -1, 0)).long()
            expert = expert.sum(dim=0, keepdim=True)
            expert[expert == 450], expert[expert == 765] = 1, 2 # Convert [0, 150, 255] to [0, 1, 2]
            label_t.append(self.resize(expert))

        if self.ground_truth[:-1] == "expert":
            return label_t[int(self.ground_truth[-1])]
        elif self.ground_truth == "mOH":
            label = torch.stack(label_t).squeeze()
            label = F.one_hot(label)
            label = label.movedim(-1, -3).float()
            return torch.mean(label, dim=0)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:
        image, label = self.images[index], self.labels[index]
        image = np.rollaxis(image, -1, 0)
        image = self.resize(torch.tensor(image).float())

        label = self._get_labels(label)

        if self.augment:
            image, label = self._augment_data(image, label)
        if self.names:
            return image, torch.squeeze(label), str(self.keys[index]).replace(str(self.data_dir), "")
        return image, torch.squeeze(label)