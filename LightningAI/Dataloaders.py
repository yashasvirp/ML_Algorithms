import os
from git import Repo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import PIL as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Getting dataset from git repository

if not os.path.exists("mnist-pngs"):
    Repo.clone_from("https://github.com/rasbt/mnist-pngs", "mnist-pngs")

# ------------------ Getting CSV files from mnist-pngs ------------------------

df_train = pd.read_csv('mnist-pngs/train.csv')
df_test = pd.read_csv('mnist-pngs/test.csv')

# ------------------ Creating validation set from train data ---------------------

df_train = df_train.sample(frac=1, random_state=123)    # pandas way of shuffling dataset. 

loc = round(df_train.shape[0]*0.9)      # splitting data in 90:10 ratio
df_new_train = df_train.iloc[:loc]
df_new_val = df_train.iloc[loc:]

df_new_train.to_csv('mnist-pngs/new_train.csv', index=None)
df_new_val.to_csv('mnist-pngs/new_val.csv', index=None)


# ----------------- Dataset class ---------------------------------------
class MyDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.img_names = df["filepath"]
        self.labels = df["label"]
    
    def __getitem__(self, index):           # gets a single record
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]


# -------------- Implementing Data loading --------------

if __name__ == "__main__":
    
    data_transforms = {     # applying tranform operations on train and test data
        "train" : transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28,28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),   # normalizing image to [-1,1] range
            ]
        ),
        "test" : transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28,28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),   # normalizing image to [-1,1] range
            ]
        ),
    }

    train_set = MyDataset("mnist-pngs/new_train.csv", "mnist-pngs/", transform=data_transforms["train"])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)    # last param indicates number of processes/CPUs to be created to work parallelly

    val_set = MyDataset("mnist-pngs/new_val.csv", "mnist-pngs/", transform=data_transforms["test"])
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

    test_set = MyDataset("mnist-pngs/test.csv", "mnist-pngs/", transform=data_transforms["test"])
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

