import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torch
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_path, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # PIL.Image.open: RGB, PIL[W, H, C]
        # cv2.imread: BGR, numpy[H, W, C]
        # torchvision.io.read_image: RGB or grayscale, torch.Tensor[C, H, W], uint8 in [0, 255]
        img = read_image(os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]), ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        # return tuple!
        return img, label

def dataloader(img_dir, annotations_path, norm_size, n_types, batch_size, workers=0):
    # img pre-processing
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(norm_size), # PIL/tensor differs
        transforms.ToTensor() # [0, 255] -> [0, 1]
    ])
    transform_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(norm_size), # PIL/tensor differs
        transforms.ToTensor() # [0, 255] -> [0, 1]
    ])
    # label pre-processing 
    target_transform = transforms.Lambda(lambda y: torch.zeros(n_types, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    dataset = CustomImageDataset(img_dir, annotations_path, 
                                 transform=transform_train if 'train' in annotations_path else transform_eval, 
                                 target_transform=target_transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if 'train' in annotations_path else False,
                      num_workers=workers)

if __name__ == '__main__':
    train_dataloader = dataloader(img_dir="./data/train", 
                                  annotations_path="./data/train.csv", 
                                  norm_size=(32, 32),
                                  n_types=24,
                                  batch_size=1)
    imgs, labels = next(iter(train_dataloader))
    print(labels[0])