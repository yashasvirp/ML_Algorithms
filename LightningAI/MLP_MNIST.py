# Training MLP using MNIST dataset

# importing dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root="./mnist", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())

#-------- Creating validation split------------------

import torch
from torch.utils.data.dataset import random_split

torch.manual_seed(1)
train_dataset, val_dataset = random_split(train_dataset, lengths=[55000,5000])

#---------- Dataloaders to load data --------------------------

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

#--------- Checking data distribution using Counter -------------------------------------------

from collections import Counter

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

print("\n Training label distribution: ")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\n Validation label distribution: ")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\n Test label distribution: ")
print(sorted(test_counter.items()))

# Above will print in (class label, number of points of each label) format 


#------------------ Zero rule classifier/ Majority class classifier -------------------------------

'''
Just to double check what will be accuracy if we always predict the most frequent class in test data.
This helps to do a sanity check when we have class imbalance datasets.
'''

majority_class = test_counter.most_common(1)[0] # returns the first most common class in test set in a tuple format (element, count)
print("majority class:", majority_class[0])

baseline_accr = majority_class[1]/sum(test_counter.values())
print("Accuracy when always predicting the majority class:")
print(f"{baseline_accr:.2f} ({baseline_accr*100:.2f}%)")

#----- Visual check --------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torchvision

for images, labels in train_loader:
    break

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training images")
plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64], padding=1, pad_value=1.0, normalize=True), (1,2,0)))
plt.show()

#--------------------------------------- Implementing MLP -------------------------------------------------

# print(images.shape) # output : torch.Size([64, 1, 28, 28])      ---> (batchsize,channel,height,width)

# # we reshape the above 28x28 image in a vector using torch.flatten

# torch.flatten(images, start_dim=1)      # we are flattening everything AFTER dimension 1. This will give new shape as torch.Size([64, 784]) which is 64 images of shape 784

class MLP(torch.nn.Module):

    def __init__(self, in_features, out_labels):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, 50),       # First layer (input layer)
            torch.nn.ReLU(),

            torch.nn.Linear(50,25),     # Second layer (Hidden Layer)
            torch.nn.ReLU(),

            torch.nn.Linear(25, out_labels)     # Third layer (output layer)
        )

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)
        op = self.layers(X)
        return op
    
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
model = MLP(in_features=784, out_labels=10) # 784 vectors with 10 ouput classes

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