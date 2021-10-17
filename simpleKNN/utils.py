import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# plots the decision boundary of the model and the scatterpoints
def plot_classifier(model, X, y):
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.figure()
    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred.astype(int), # unsigned int causes problems with negative sign... o_O
                cmap=plt.cm.RdBu, alpha=0.6)

    plt.scatter(x1[y==0], x2[y==0], color="b", label="class 0")
    plt.scatter(x1[y==1], x2[y==1], color="r", label="class 1")
    plt.legend()


# Computes the element with the maximum count
def mode(y):
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


# Computes the Euclidean distance between rows of 'X' and rows of 'Xtest
def euclidean_dist_squared(X, Xtest):
    X_norms_sq = np.sum(X ** 2, axis=1)
    Xtest_norms_sq = np.sum(Xtest ** 2, axis=1)
    dots = X @ Xtest.T

    return X_norms_sq[:, np.newaxis] + Xtest_norms_sq[np.newaxis, :] - 2 * dots
