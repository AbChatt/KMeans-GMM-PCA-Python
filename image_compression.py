"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Assignment 3
B. Chan, S. Wei, D. Fleet

This file compresses images using PCA
"""

import matplotlib
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from pca import PCA

def main(data, K, num_display):
    assert K > 0
    assert num_display > 0
    image_shape = data.shape[1:]
    images = data.reshape(-1, np.product(image_shape))

    pca = PCA(images)

    low_dim_data = pca.reduce_dimensionality(images, K)

    reconstructed_images = pca.reconstruct(low_dim_data)

    images_to_show = np.random.choice(range(images.shape[0]), size=num_display)
    fig, axes = plt.subplots(nrows=2, ncols=num_display, figsize=(12, 8))

    axes[0, 0].set_ylabel("Original", size='large')
    axes[1, 0].set_ylabel("Reconstruction", size='large')
    for idx, image_i in enumerate(images_to_show):
        axes[0, idx].set_yticklabels('')
        axes[0, idx].set_xticklabels('')
        axes[0, idx].tick_params(axis='both',
                                 which='both',
                                 bottom=False,
                                 top=False,
                                 labelbottom=False)
        axes[0, idx].imshow(images[image_i, :].reshape(image_shape))

        axes[1, idx].set_yticklabels('')
        axes[1, idx].set_xticklabels('')
        axes[1, idx].tick_params(axis='both',
                                 which='both',
                                 bottom=False,
                                 top=False,
                                 labelbottom=False)
        axes[1, idx].imshow(reconstructed_images[image_i, :].reshape(image_shape))

    plt.show()

    with open("data/compressed_eye_image_data.pkl", "wb") as f:
        pickle.dump({
          "low_dim_data": low_dim_data,
          "V": pca.V,
          "mean": pca.mean
        }, f)


if __name__ == "__main__":
    # You can change the seed to check other images.
    seed = 100
    np.random.seed(seed)

    data = pickle.load(open("data/eye_image_data.pkl", "rb"))
    K = 100
    num_display = 10
    main(data, K, num_display)