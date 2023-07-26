import torch

X_train = torch.tensor([258.0,270.0,294.0,320.0,342.0,368.0,396.0,446.0,480.0,586.0]).view(-1,1)
y_train = torch.tensor([236.4,234.4,252.8,298.6,314.2,342.2,360.8,368.0,391.2,390.8])

from matplotlib import pyplot as plt

# plt.scatter(X_train, y_train)
# plt.xlabel("Feature variable")
# plt.ylabel("Target variable")
# plt.show()
# ---------- MLP building (same as MLP before) ----------------

class PytorchMLP(torch.nn.Module):
    def __init__(self, nX):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(nX, 50),    # First Hidden layer
            torch.nn.ReLU(),            # Activation

            torch.nn.Linear(50,25),     # Second Hidden Layer
            torch.nn.ReLU(),            # Activation

            torch.nn.Linear(25, 1),     # only one output
        )
    
    def forward(self, x):
        logits = self.all_layers(x).flatten()   # output of NN stored in Logits
        return logits

# ------------ Normalizing data ----------------------------

X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_norm = (X_train - X_mean)/X_std
y_norm  = (y_train - y_mean)/y_std      # normalize y values as well as predictions are on normalized x values

# -------------- Defining Dataset class ----------------------------

from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.target = y
    
    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.target.shape[0]

train_set = MyDataset(X_norm, y_norm)

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)

# --------------- Training loop --------------------------

import torch.nn.functional as F

torch.manual_seed(1)
model = PytorchMLP(nX=1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 30

loss_list = []
train_acc, val_acc = [], []

for i in range(num_epochs):
    
    model = model.train()
    for batch_id, (nX, ny) in enumerate(train_loader):
        op = model(nX)
        loss = F.mse_loss(op, ny)       # mse loss w.r.t output and actual target(ny)

        optimizer.zero_grad()       # Backprop
        loss.backward()
        optimizer.step()

        if not batch_id % 250:      # Logging
            print(f"Epoch: {i + 1:03d}/{num_epochs:03d} | Batch: {batch_id:03d}/{len(train_loader):03d} | Train loss: {loss:.2f}")
        
        loss_list.append(loss.item())

# -------------- Model Prediction -------------------------
model.eval()
X_range = torch.arange(150,800,0.1).view(-1,1)
X_rnorm = (X_range - X_mean) / X_std        # it is because model was trained on standardized/normalized data

# predict

with torch.no_grad():
    y_rnorm = model(X_rnorm)     # normalized y predictions

# undo the normalization to plot values
y_mlp = y_rnorm * y_std + y_mean

# plotting

plt.scatter(X_train, y_train, label = "Training points")
plt.plot(X_range, y_mlp, color="C1", label = "MLP fit", linestyle="-")
plt.xlabel("Feature")
plt.ylabel("target")
plt.legend()
plt.show()
# plt.savefig("MLP.pdf")    saving plots