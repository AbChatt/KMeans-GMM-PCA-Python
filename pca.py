"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Assignment 3
B. Chan, S. Wei, D. Fleet

===========================================================

 COMPLETE THIS TEXT BOX:

 Student Name:
 Student number:
 UtorID:

 I hereby certify that the work contained here is my own


 ____________________
 (sign with your name)

===========================================================
"""

import numpy as np

class PCA:
    def __init__(self, X):
        """ This class represents PCA with components and mean given by data.

        TODO: You will need to implement the methods of this class:
        - _compute_components: ndarray -> ndarray
        - reduce_dimensionality: ndarray, int -> ndarray
        - reconstruct: ndarray -> ndarray

        Implementation description will be provided under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Dimension of low-dimensional representation of input features.
             NOTE: K >= 1

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        """

        # Mean of each column, shape: (D, )
        self.mean = np.mean(X, axis=0)
        self.V = self._compute_components(X)

    def _compute_components(self, X):
        """ This method computes the PCA directions (one per column) given data.

        NOTE: Use np.linalg.eigh to compute the eigenvectors

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

        Output:
        - V (ndarray (shape: (D, D))): The matrix of PCA directions (one per column) sorted in descending order.
        """
        assert len(X.shape) == 2, f"X must be a NxD matrix. Got: {X.shape}"
        (N, D) = X.shape

        # ====================================================
        # TODO: Implement your solution within the box
        
        # ====================================================

        assert V.shape == (D, D), f"V shape mismatch. Expected: {(D, D)}. Got: {V.shape}"
        return V

    def reduce_dimensionality(self, X, K):
        """ This method reduces the dimensionality of X to K-dimensional using the precomputed mean and components.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - K (int): Number of dimensions for the low-dimensional input data.

        Output:
        - low_dim_X (ndarray (shape: (N, K))): A NxD matrix consisting N K-dimensional low-dimensional representation of input data.
        """
        assert len(X.shape) == 2, f"X must be a NxD matrix. Got: {X.shape}"
        (N, D) = X.shape
        assert D > 0, f"dimensionality of representation must be at least 1. Got: {D}"
        assert K > 0, f"dimensionality of representation must be at least 1. Got: {K}"

        # ====================================================
        # TODO: Implement your solution within the box
        
        # ====================================================

        assert low_dim_X.shape == (N, K), f"low_dim_X shape mismatch. Expected: {(N, K)}. Got: {low_dim_X.shape}"
        return low_dim_X

    def reconstruct(self, low_dim_X):
        """ This method reconstruct X from low-dimensional input data using the precomputed mean and components.

        NOTE: The K is implicitly defined by low_dim_X.

        Args:
        - low_dim_X (ndarray (shape: (N, K))): A NxD matrix consisting N K-dimensional low-dimensional representation of input data.

        Output:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional reconstructed input data.
        """
        assert len(low_dim_X.shape) == 2, f"low_dim_X must be a NxK matrix. Got: {low_dim_X.shape}"
        (N, K) = low_dim_X.shape
        assert K > 0, f"dimensionality of representation must be at least 1. Got: {K}"
        D = self.mean.shape[0]

        # ====================================================
        # TODO: Implement your solution within the box
        
        # ====================================================

        assert X.shape == (N, D), f"X shape mismatch. Expected: {(N, D)}. Got: {X.shape}"
        return X
