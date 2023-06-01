'''Adaptive Linear Neuron - Uses gradient descent.

The weight update is calculated based on all examples in the training dataset (instead of updating the parameters
incrementally after each training example), which is why this approach is also referred to as batch
gradient descent. We will refer to this process as full batch gradient descent.'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class AdalineSGD:
    """ADAptive LInear NEuron classifier with Stochastic Gradient Descent.
        
        Parameters
        ------------
        eta : float        Learning rate (between 0.0 and 1.0)
        n_iter : int        Passes over the training dataset.
        shuffle : bool (default: True) Shuffles training data every epoch if True to prevent cycles.
        random_state : int        Random number generator seed for random weight initialization.

        Attributes
        -----------
        w_ : 1d-array        Weights after fitting.
        b_ : Scalar        Bias unit after fitting.
        losses_ : list        Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta = 0.01, n_iter = 50, shuffle = True, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_inititalized = False
        self.random_state = random_state
    
    def _initialize_weights(self, m):
        '''Initialize wegihts to small random numbers'''
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=m) # assigning random weights in normal distribution
        self.b_ = np.float_(0.) # assigning 0 bias
        self.w_inititalized = True
    
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights'''
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi*error
        self.b_ += self.eta * 2.0 * error
        loss = (error**2)
        return loss
    
    def partial_fit(self, X, y):
        '''Fit training data without re-initializing weights'''
        if not self.w_inititalized:
            self._initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        
        return self


    def fit(self, X, y):
        """Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]    Training vectors, where n_examples is the number of examples and n_features is the number of features.
            y : array-like, shape = [n_examples]     Target values.
            
            Returns
            -------
            self : object
        """

        self._initialize_weights(X.shape[1])
        self.losses_ = [] # tracking losses 

        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            losses = []
            for xi, target in zip(X,y):
                losses.append(self._update_weights(xi,target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        
        return self
    
    def net_input(self, X):
        '''Calculate net input Z = wX + b'''
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        return X

    def predict(self, X):
        '''Return class label after unit step'''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

def plot_decision_regions(X,y,classifier,resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


# Trying Adaline on IRIS dataset

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s,header = None, encoding = 'utf-8')

y = df.iloc[:100,4].values  # extracting classes
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100,[0,2]].values # extracting features as Sepal and Petal lengths


# Plotting loss against number of epochs for different learning rates


# Using standardisation to make gradient descent work better 

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()
