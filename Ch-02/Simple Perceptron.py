import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ScratchPerceptron:
    def __init__(self, num_features):           # initialising weights and biases for number of features
        self.num_features = num_features
        self.weights = [2.0 for _ in range(num_features)] # initializing weights and bias as 1
        self.bias = 2
    
    def __forward(self, X):
        res = self.bias
        
        for i in range(len(X)):
            res += self.weights[i]*X[i] # calculating W_i*X_i + b
        
        return 1 if res > 0 else 0 # return 1 if res > threshold else 0 ---> class prediction

    def __update_weights(self, X, true_y): # learning via errors and updating weights and bias
        
        pred = self.__forward(X)
        error = true_y - pred

        for i in range(len(X)):
            self.weights[i] += error*X[i]
        
        self.bias += error
    
    def train(self, X, y, n_iter=10): # this function is accessible outside the class and will return updated weights and bias
        for _ in range(n_iter):
            error_count = 0
            for i,j in zip(X,y):
                self.__update_weights(i,j)
        
        return (self.weights, self.bias)

    def plot_boundary(self):
        w1, w2 = self.weights[0], self.weights[1]
        b = self.bias

        x1_min = -20
        x2_min = (-(w1*x1_min) - b)/w2          # calculating points for line (ax + by + c = 0 => y = (-c - ax)/b )

        x1_max = 20
        x2_max = (-(w1*x1_max) - b)/w2

        return x1_min, x2_min, x1_max, x2_max

        
s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s,header = None, encoding = 'utf-8')

y = df.iloc[:100,4].values  # extracting classes
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100,[0,2]].values# extracting features as Sepal and Petal lengths


# creating an instance
ppn = ScratchPerceptron(2)
xb1, yb1, xb2, yb2 = ppn.plot_boundary() # line before training
print(xb1, yb1, xb2, yb2)
W, b = ppn.train(X,y)
x1, y1, x2, y2 = ppn.plot_boundary()
print(x1, y1, x2, y2)



print(W)
print(b)


plt.plot(X[y==0,0], X[y==0,1], marker = "D", markersize=10, linestyle="", label="class 0") # plotting points where class label y = 0
plt.plot(X[y==1,0], X[y==1,1], marker = "^", markersize=13, linestyle="", label="class 1") # plotting points where class label y = 1
plt.legend(loc=2)
plt.xlim([0,10])
plt.ylim([0,10])

plt.plot([xb1,xb2],[yb1,yb2], color="r") # line before training the model

plt.plot([x1,x2],[y1,y2], color="k") # line after training the model

plt.xlabel("Feature $x_1$", fontsize= 12)
plt.ylabel("Feature $x_2$", fontsize= 12)
plt.grid()
plt.show()
