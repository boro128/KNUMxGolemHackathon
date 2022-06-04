import os
import torch
import pandas as pd

from typing import Optional, Callable
from PIL import Image


class MagicDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_dir: str,
        labels_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        self.img_dir = img_dir
        self.labels_file = labels_file
        self.img_list = os.listdir(self.img_dir)
        if labels_file is not None:
            self.img_labels = pd.read_csv(labels_file).loc[:, "category_id"]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = Image.open(img_path)
        label = self.img_labels[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
