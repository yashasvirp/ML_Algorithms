# Implementing Logistic Regression with Full batch Gradient descent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class LogisticRegressionGD:
    
    """Gradient descent-based logistic regression classifier.
    Parameters
    ------------
    eta : float             Learning rate (between 0.0 and 1.0)
    n_iter : int            Passes over the training dataset.
    random_state : int      Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array     Weights after training.
    b_ : Scalar       Bias unit after fitting.
    losses_ : list    Mean squared error loss function values in each epoch.
    """

    def __init__(self, eta=0.1, n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def net_input(self, X):
        '''Calculating net input z = WX + b'''
        return np.dot(self.w_, X.T) + self.b_

    def activation(self, z):
        '''Computing logistic sigmoid activation'''
        return 1./(1. + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]   Training vectors, where n_examples is the number of examples and n_features is the number of features.
        y : array-like, shape = [n_examples]        Target values.
        
        Returns
        -------
        self : Instance of LogisticRegressionGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = X.shape[1])       # sample weights of row shapes of data, from normal dist. with mean 0 and std dev 0.01
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            ip = self.net_input(X)
            op = self.activation(ip)
            errors = y - op
            self.w_ += self.eta * 2.0 * X.T.dot(errors)/X.shape[0]  # updating weights
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = ((-y.dot(np.log(op)) - (1 - y).dot(np.log(1 - op)))/X.shape[0])
            self.losses_.append(loss)

        return self

    def predict(self, X):
        '''return class label after unit step'''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


def plot_decision_regions(X, y, classifier, test_idx=None,resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.8,c=colors[idx],marker=markers[idx],label=f'Class {cl}',edgecolor='black')
    # highlight test examples
    if test_idx:
    # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],c='none', edgecolor='black', alpha=1.0,linewidth=1, marker='o',s=100, label='Test set')

import pandas as pd

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s,header = None, encoding = 'utf-8')

y = df.iloc[:100,4].values  # extracting classes
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100,[0,2]].values # extracting features as Sepal and Petal lengths

X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

lgr = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lgr.fit(X,y)
plot_decision_regions(X,y,lgr)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
