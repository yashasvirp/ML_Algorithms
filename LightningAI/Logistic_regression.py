import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Any network will be built as a class and inherits torch.nn.Module
class LogisticRegression(torch.nn.Module): 
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=num_features, out_features= 1) # we have n input parameter and one output feature to be predicted

    
    '''
        without doing linear(x), we could do the following.
        After initializing linear = torch.nn.Linear(n, 1)
        we can get weights as :
            w = linear.weights.detach()
            b = linear.bias.detach()
            logits or z = X.matmul(w.T) + b

        above code is replaced by logits = linear(X)
    '''
    def forward(self, X):
        logits = self.linear(X)         # will directly calculate Z = wx + b
        prob = torch.sigmoid(logits)
        return prob




# Creating a dataloader for minibatch gradient descent
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        # super.__init__()
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)
    
    def __getitem__(self, index):           
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):        # number of examples in our dataset
        return self.labels.shape[0]



s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s,header = None, encoding = 'utf-8')

y = df.iloc[:100,4].values  # extracting classes
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100,[0,2]].values# extracting features as Sepal and Petal lengths

# Standardizing the dataset
X = (X - X.mean(axis=0)) / X.std(axis=0)  # doing (x - mean)/(std deviation) to standardize the data



# Using a dataloader with MyDataset class
train_ds = MyDataset(X, y)
train_loader = DataLoader(dataset=train_ds, batch_size=10, shuffle=True)    



# training the model

import torch.nn.functional as F

torch.manual_seed(1)            # Ensures the results are reproducible
model = LogisticRegression(num_features=2)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05) # we use SGD optimizer that will update all the parameters from model.parameters() i.e. the weights and bias
epochs = 20

for epoch in range(epochs):
    model = model.train()       # setting model in training mode. Good practice to always use this

    for batch_idx, (features, class_labels) in enumerate(train_loader):
        
        prob = model(features)      

        loss = F.binary_cross_entropy(prob, class_labels.view(prob.shape))  # computing logistic regression loss 
                                                            # with predicted values and actual class_labels as parameters

        # backward pass
        optimizer.zero_grad()       # this will prevent accumualtion of gradients over iterations
        loss.backward()         # will compute relevant gradients
        optimizer.step()        # will perform the relevant updates to parameters


        print(f'epoch:{epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss : .2f}')


# computing accuracy
def compute_accuracy(model, dataloader):

    model = model.eval()        # putting the model in evaluation mode

    correct = 0.0
    total = 0.0

    for idx, (features, class_labels) in enumerate(train_loader):

        with torch.no_grad():
            prob = model(features)
        
        pred = torch.where(prob > 0.5, 1, 0)
        y = class_labels.view(pred.shape).to(pred.dtype)

        correct += torch.sum(y == pred)
        total += len(y == pred)
    
    return correct/total

accuracy = compute_accuracy(model, train_loader)
print('Accuracy(%) : ', accuracy*100)





# Visualiziation

def plot_boundary(model):
        w1 = model.linear.weight[0][0].detach()
        w2 = model.linear.weight[0][1].detach()
        b = model.linear.bias[0].detach()

        x1_min = torch.tensor(-20.)
        x2_min = (-(w1*x1_min) - b)/w2          # calculating points for line (ax + by + c = 0 => y = (-c - ax)/b )

        x1_max = torch.tensor(20.)
        x2_max = (-(w1*x1_max) - b)/w2

        return x1_min, x1_max, x2_min, x2_max

x1_min, x1_max, x2_min, x2_max = plot_boundary(model)

plt.plot(X[y==0,0], X[y==0,1], marker = "D", markersize=10, linestyle="", label="class 0") # plotting points where class label y = 0
plt.plot(X[y==1,0], X[y==1,1], marker = "^", markersize=13, linestyle="", label="class 1") # plotting points where class label y = 1
plt.legend(loc=2)
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k") # line after training the model

plt.xlabel("Feature $x_1$", fontsize= 12)
plt.ylabel("Feature $x_2$", fontsize= 12)
plt.grid()
plt.show()


'''
    computation graph is usually stored in memory to perform backward prop or calculate derivatives. Sometimes it is necessary
'''
# with torch.inference_mode():            # this or no_grad() will not store computation graph in memory
#     model = LogisticRegression(2) 