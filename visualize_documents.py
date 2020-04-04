"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Assignment 3
B. Chan, S. Wei, D. Fleet

This file visualizes the document dataset by reducing the dimensionality with PCA
"""

import matplotlib
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from pca import PCA

def main(dataset):
    documents = dataset['data'].astype(np.float)

    # NOTE: MATLAB is really fast for this compared to numpy!
    pca = PCA(documents)
    low_dim_data = pca.reduce_dimensionality(documents, 3)

    classes = np.unique(dataset['labels'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for class_i in classes:
        class_i_data = low_dim_data[dataset['labels'].flatten() == class_i]
        ax.scatter(class_i_data[:, 0],
                   class_i_data[:, 1],
                   class_i_data[:, 2],
                   s=1)

    plt.show()


if __name__ == "__main__":
    dataset = pickle.load(open("data/BBC_data.pkl", "rb"))
    main(dataset)
