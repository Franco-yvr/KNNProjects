#!/usr/bin/env python
import os
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import utils
from knn import KNN


def load_dataset(filename):
    with open(Path(".", "data", filename), "rb") as f:
        return pkl.load(f)


def main():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    # i being the number of nearest neighbour required in calculations
    for i in [1,3,10]:
        model = KNN(k = i)
        model.fit(X, y)

        print("for k = ", i)
        y_hat_train = model.predict(X)
        err_train = np.mean(y_hat_train != y)
        print(f"Knn training error: {err_train:.3f}")

        y_hat_test = model.predict(X_test)
        err_valid = np.mean(y_hat_test != y_test)
        print(f"Knn validation error: {err_valid:.3f}")

    model = KNN(k=4)
    model.fit(X, y)
    utils.plot_classifier(model,X, y)

    fname = os.path.join(".", "figs", "knn_with_k=4.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)
    plt.clf()


def cross_validate():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    train_result = np.zeros(len(ks))
    cv_results = np.zeros(len(ks))
    test_result = np.zeros(len(ks))

    # cross-validation
    fold_number = 10
    for index, i in enumerate(ks):
        fold_results = np.zeros(fold_number)
        for fold in range(fold_number):
        
            # build mask array
            mask_array = np.ones(X.shape[0], dtype=bool)
            bottom = round(fold / fold_number * X.shape[0])
            top = round((fold+1) / fold_number * X.shape[0])
            mask_array[bottom:top] = False

            # Select the fold with mask array
            x_train = X[mask_array,:]
            y_train = y[mask_array]
            x_valid = X[~mask_array,:]
            y_valid = y[~mask_array]

            # fit and predict fold
            model = KNN(k = i)
            model.fit(x_train, y_train)
            y_hat_train = model.predict(x_valid)
            err_train = np.mean(y_hat_train != y_valid)

            # save the results
            fold_results[fold] = err_train
        cv_results[index] = np.mean(fold_results)

    # graph train, cv and test error for each k
    for index, i in enumerate(ks):
        model = KNN(k=i)
        model.fit(X, y)
        y_hat_train = model.predict(X_test)
        err_test = np.mean(y_hat_train != y_test)
        test_result[index] = err_test

    for index, i in enumerate(ks):
        model = KNN(k=i)
        model.fit(X, y)
        y_hat_train = model.predict(X)
        err_test = np.mean(y_hat_train != y)
        train_result[index] = err_test
    print("The k values are:", ks)
    print("The train result is:", train_result)
    print("The cross-validation result is:", cv_results)
    print("The test result is:",test_result)

    # plot the results
    plt.plot(ks, train_result, label="train error")
    plt.plot(ks, cv_results, label = "cross-validation error")
    plt.plot(ks, test_result, label = "test error")
    plt.title("Error Rates for each value K")
    plt.xlabel("K values")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(axis='x')
    plt.grid(axis='y')
    fname = os.path.join(".", "figs", "errors.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


if __name__ == "__main__":
    main()
    cross_validate()
