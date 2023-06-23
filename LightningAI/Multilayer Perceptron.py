import pandas as pd
import numpy as np

df = pd.read_csv("xor.csv")
X = df[["x1","x2"]].values
y = df["class label"].values

# splitting the dataset into train and test using sklearn
# test set will contain only 15% of rows from dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1, stratify=y)

# Now we split validation set FROM train set. It will have only 10% of training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

# y_train = y_train[:,0]
# # visualizing the dataset

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['savefig.dpi'] = 80
# mpl.rcParams['figure.dpi'] = 300

# plt.plot(
#     X_train[y_train[:, 0] == 0, 0],
#     X_train[y_train[:, 0] == 0, 1],
#     marker="D",
#     markersize=10,
#     linestyle="",
#     label="Class 0",
# )

# plt.plot(
#     X_train[y_train[:, 0] == 1, 0],
#     X_train[y_train[:, 0] == 1, 1],
#     marker="^",
#     markersize=13,
#     linestyle="",
#     label="Class 1",
# )

# plt.legend(loc=2)

# plt.xlim([-5, 5])
# plt.ylim([-5, 5])

# plt.xlabel("Feature $x_1$", fontsize=12)
# plt.ylabel("Feature $x_2$", fontsize=12)

# plt.grid()
# plt.show()
#---------------------------------------------------------------------------------------------------------------
# Implementing the perceptron

import torch

class MLP(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()

        self.total_layers = torch.nn.Sequential(

            torch.nn.Linear(in_features, 25),   # 1st hidden layer with inputs and 25 output values (25 neurons/units)
            torch.nn.ReLU(),    # Activation function for first layer

            torch.nn.Linear(25,15),     # 2nd layer with 25 inputs and 15 outputs
            torch.nn.ReLU(),            # Activation for 2nd layer

            torch.nn.Linear(15, out_classes)    # Output layer
        )   # number of outputs from i^th layer should match number of inputs in (i+1)^th layer

    def forward(self, X):
        op = self.total_layers(X)
        return op
    
#---------------------------------------------------------------------------------------------------------------------------
# Defining dataloaders for implementing MLP
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        X = self.features[index]
        y = self.labels[index]
        return X, y
    
    def __len__(self):
        return self.labels.shape[0]

train_ds = MyDataset(X_train, y_train)
val_ds = MyDataset(X_val, y_val)
test_ds = MyDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=False)

#--------------------------------------------------------------------------------------------------------------------------------
# Defining accuracy fn (optional probably)
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

#----------------------------------------------------------------------------------------------------------------------------
# Training Loop

import torch.nn.functional as F

torch.manual_seed(1)        # for reproducibility of results
model = MLP(in_features=2, out_classes=2)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)  # Stochastic GD

num_epochs = 10

for i in range(num_epochs):

    model = model.train()           # setting model in training mode

    for batch_idx, (features, labels) in enumerate(train_loader):
        op = model(features)
        loss = F.cross_entropy(op, labels)      # calculating loss function

        optimizer.zero_grad()
        loss.backward()         # back propogation
        optimizer.step()        # updating weights of model

        ### LOGGING
        print(f"Epoch: {i+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")
    
    train_acc = accr(model, train_loader)
    val_acc = accr(model, val_loader)
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")


#------------Model Evaluation-----------------------------------------
train_acc = accr(model, train_loader)
val_acc = accr(model, val_loader)
test_acc = accr(model, test_loader)

print(f"Train accuracy : {train_acc*100:.2f}%")
print(f"Validation accuracy : {val_acc*100:.2f}%")
print(f"Test accuracy : {test_acc*100:.2f}%")

#-------- Plotting Decision Boundary -----------------------------------

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('D', '^', 'x', 's', 'v')
    colors = ('C0', 'C1', 'C2', 'C3', 'C4')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits = classifier.forward(tensor)
    Z = np.argmax(logits.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    #edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.show()

plot_decision_regions(X_train, y_train, classifier=model)