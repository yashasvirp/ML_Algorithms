
from sklearn import datasets        # Importing dataset from sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

# using train_test_split to split data into train and test sets. 
# test_size = .3 implies 30% of dataset is used for testing
# stratify will ensure that the examples of all class labels are split into proper proportions. It can be checked using np.bincount()
# random_state seed will ensure reproducing the same order of shuffling data every time.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



# Importing Logistic Regression model from sklearn
# Two ways for multiclass classification - ovr and multinomial(default choice)
# many solver parameters are available for optimization other than SGD - lbfgs, newton-cg, liblinear, sag and saga
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

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


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

'''
To predict using Logistic Regression:

In [16]: lr.predict_proba(X_test_std[:3,:])
Out[16]: 
array([[3.81527885e-09, 1.44792866e-01, 8.55207131e-01],
       [8.34020679e-01, 1.65979321e-01, 3.25737138e-13],
       [8.48831425e-01, 1.51168575e-01, 2.62277619e-14]])

       First row - probability of first flower for each class
       second row - probability of second flower for each class, and so on. Highest probability for each sample is given by argmax

In [14]: lr.predict_proba(X_test_std[:3,:]).argmax(axis=1)
Out[14]: array([2, 0, 0]) ----> class for each flower


It can be done using lr.predict() as follows:

In [15]: lr.predict(X_test_std[:3,:])
Out[15]: array([2, 0, 0])

To get an prediction of single example:
scikit-learn expects a two-dimensional array as data input; thus, we have to convert a single row slice into such a
format first. One way to convert a single row entry into a two-dimensional data array is to use NumPy's reshape method.

In [18]: lr.predict(X_test_std[0,:].reshape(1,-1))
Out[18]: array([2])

'''