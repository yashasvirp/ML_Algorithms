# Multiclass Logistic Regression using softmax activation function
import torch

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features) -> None:
        super().__init__()
        self.l = torch.nn.Linear(num_features, 3) # 3 output class labels to be predicted (due to one-hot encoding)

    def forward(self, x):
        logits = self.l(x)
        prob = torch.nn.functional.softmax(logits)
        return prob
    

import torch.nn.functional as F
def cross_entropy(net_inputs, y):
    activations = torch.softmax(net_inputs, dim=1)      # dim = 1 ensures all values in one row sum upto 1
    y_onehot = F.one_hot(y)
    loss = -torch.sum(torch.log(activations)*y_onehot, dim = 1)
    avg_loss = torch.mean(loss)
    return avg_loss

torch.manual_seed(1)
net_inputs = torch.rand((4,3))
# print(net_inputs)
y = torch.tensor([0,2,2,1])
print(cross_entropy(net_inputs, y))
# print(F.cross_entropy(net_inputs, y))

