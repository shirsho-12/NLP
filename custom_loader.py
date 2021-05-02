from pathlib import Path
from torch.utils.data import Dataset
import cv2
# import torch


class CustomDataLoader(Dataset):
    """Custom Data Loader for variable needs. Pass labelling method based on data"""
    def __init__(self, data_dir, get_label, get_transform, label_transform):
        self.transforms = get_transform
        self.get_label = get_label
        self.data_dir = Path(data_dir)
        self.label_transform = label_transform
        self.data = list(data_dir.glob("**/*"))

    # Change len based on requirement
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.get_label(item)
        img = cv2.cvtColor(cv2.resize(cv2.imread(str(item)), (256, 256)), cv2.COLOR_BGR2RGB)
        return {"image": self.transforms(img), "label": self.label_transform(label)}
