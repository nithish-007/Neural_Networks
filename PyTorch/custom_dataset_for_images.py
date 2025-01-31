import os 
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import transforms
import torch

# from csv
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) ## 25000
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        y_label = self.annotations[index, 1]

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

dataset = CatsAndDogsDataset(csv_file= "cats_dogs.csv",
                             root_dir= "#####",
                             transform= transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
#data loader = .....

# from directory
# after separate data into train and test each has cat and dog
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image

def find_classes(directory:str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")
    
    class_to_dir = {class_name: idx for idx, class_name in enumerate(classes)}

    return classes, class_to_dir

class CatsAndDogsCustomDataset(Dataset):
    def __init__(self, root_dir:str, transform = None) -> None:
        self.path = list(Path(root_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(root_dir)

    def load_images(self, index:int) -> Image.Image:
        image_path = self.path[index]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img = self.load_images(index)
        class_name = self.path[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            img = self.transform(img)

        return (img, class_idx)
    
train_set = CatsAndDogsCustomDataset(root_dir= "train_dir",
                                     transform = transform)


    
