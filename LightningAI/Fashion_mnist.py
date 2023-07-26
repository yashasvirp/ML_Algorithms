# -------------- Loading Dataset -----------------------
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch

def load_mnist(path, kind = 'train'):
    import os
    import gzip
    import numpy as np

    labelPath = os.path.join(path, '%s-labels-idx1-ubyte.gz'%kind)
    imagesPath = os.path.join(path, '%s-images-idx3-ubyte.gz'%kind)

    with gzip.open(labelPath, 'rb') as lp:
        labels = np.frombuffer(lp.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(imagesPath, 'rb') as ip:
        img = np.frombuffer(ip.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return img, labels


class Mydataset(Dataset):
    def __init__(self, data_dir, kind, transform=None) -> None:
        self.transform = transform
        self.images, self.labels = load_mnist(data_dir, kind=kind)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = torch.tensor(img).to(torch.float32)
        img = img/255
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return self.labels.shape[0]

train_ds = Mydataset(data_dir="./fashion-mnist/data/fashion", kind="train")
test_ds = Mydataset(data_dir="./fashion-mnist/data/fashion", kind="t10k")

#-------- Creating validation split------------------

import torch
from torch.utils.data.dataset import random_split

torch.manual_seed(1)
train_dataset, val_dataset = random_split(train_ds, lengths=[55000,5000])

#---------- Dataloaders to load data --------------------------

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=64, shuffle=True)

# ---------------- Implementing model ------------------------

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

# ----------------------- Computing accuracy -----------------------

def accr(model, dataloader):
    
    model = model.eval()
    correct = 0.0
    tot_exmpl = 0

    for index, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():        # same as no_grad() to avoid accumulation of gradients
            op = model(features)
        
        pred = torch.argmax(op, dim = 1)

        ans = labels == pred
        correct += torch.sum(ans)
        tot_exmpl += len(ans)
    
    return correct/tot_exmpl

# -------------- Training Loop -----------------------------------------------

import torch.nn.functional as F

torch.manual_seed(1)
model = PyTorchMLP(784, 10) # 784 vectors with 10 ouput classes

optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

n = 10

loss_list = []
train_acc_list = []; val_acc_list = []

for epoch in range(n):

    model = model.train()

    for i, (features, labels) in enumerate(train_loader):

        op = model(features)
        loss = F.cross_entropy(op, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not i%250:           # Logging
            print(f"Epoch: {epoch+1:03d}/{n:03d}"
              f" | Batch {i:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")
        loss_list.append(loss.item())
    
    train_acc = accr(model, train_loader)
    val_acc = accr(model, val_loader)
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

#------------Model Evaluation-----------------------------------------
train_acc = accr(model, train_loader)
val_acc = accr(model, val_loader)
test_acc = accr(model, test_loader)

print(f"Train accuracy : {train_acc*100:.2f}%")
print(f"Validation accuracy : {val_acc*100:.2f}%")
print(f"Test accuracy : {test_acc*100:.2f}%")