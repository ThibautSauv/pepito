from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms
import torch
import json


def get_class(full_path):
    folder_name = os.path.basename(os.path.dirname(full_path))
    return os.path.basename(folder_name)


LABEL_MAP = {"in": 0, "out": 1}


class PepitoDataset(Dataset):
    """
    Custom torch dataset for Pepito dataset. The data is stored under "./dataset" directory and structured as follows:
    - dataset
        - in
            - 0.jpg
            - 1.jpg
            - ...
        - out
            - 0.jpg
            - 1.jpg
            - ...
    where "in" and "out" are the target classes.

    Args:
        - data_dir (str): the path to the dataset directory.
        - transform (torchvision.transforms.Compose): a composition of torchvision transforms to apply to the data.
    """

    def __init__(self, data_dir, transform=None):
        self.transform = transform

        self.all_paths = glob.glob(os.path.join(data_dir, "*/*.jpg"))

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):

        img_path = self.all_paths[idx]
        img = read_image(img_path)
        resized = transforms.Resize((224, 224))(img).to(torch.float32) / 255.0
        class_name = get_class(img_path)

        if self.transform:
            resized = self.transform(resized)

        return resized, img_path, LABEL_MAP[class_name]


class LimitedPepitoDataset(PepitoDataset):
    """
    Limits the PepitoDataset to a certain number of samples, given in a json file.
    """

    def __init__(self, json_dir, key, transform=None):
        self.transform = transform

        with open(json_dir, "r") as f:
            self.json_data = json.load(f)

        assert key in self.json_data, f"The json file must contain a '{key}' key."
        self.all_paths = self.json_data[key]
