import numpy as np
import pandas as pd

class SVD:
    """Class for SVD dimensionality reduction technique"""

    def __init__(self, mat, k):
        data_matrix = np.array(mat)
        self.k = k
        self.U, self.S = self.__get_object_factor_matrix(data_matrix, k)
        self.V = self.__get_feature_factor_matrix(data_matrix, k)

    def __get_object_factor_matrix(self, data_matrix, k):
        
        # Compute the covariance matrix
        covariance_matrix = np.dot(data_matrix, data_matrix.T)

        # Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top-k eigenvectors
        non_zero_eigenvectors = eigenvectors[:, eigenvalues > 1e-10]

        #Select Top K latent semantics
        selected_eigenvectors = non_zero_eigenvectors[:, :k]
        S = np.diag(np.sqrt(eigenvalues))

        return np.real(selected_eigenvectors), np.real(S)
    

    def __get_feature_factor_matrix(self, data_matrix, k):
        
        # Compute the covariance matrix
        covariance_matrix = np.dot(data_matrix.T, data_matrix)

        # Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top-k eigenvectors
        non_zero_eigenvectors = eigenvectors[:, eigenvalues > 1e-10]

        #Select Top K latent semantics
        selected_eigenvectors = non_zero_eigenvectors[:, :k]

        return selected_eigenvectors

    def get_latent_semantics_data(self):
        ''' returns a matrix of top-k latent features and latent semantics data for analysis'''
        return self.V, (self.U, self.S, self.V)
    
    def get_object_id_weight_pair(self, ids):
        '''returns dataframe of object_id, weight pairs in the decreasing order'''

        # Create DataFrame for sorted
        df = pd.DataFrame(self.U, index = ids)

        # Created sorted df based on all columns in the decreasing order
        sorted_df =  df.sort_values(by=list(range(self.k)), axis=0, ascending=False)

        # Create object-id-weight pairs in decreasing order, considering the first column as the weight
        object_id_weight_pairs = [(id, df.loc[:, 0].loc[id]) for id in sorted_df.index]

        # Create a Pandas DataFrame for the sorted pairs
        df = pd.DataFrame(object_id_weight_pairs, columns=['Object_ID', 'Weight'])

        return df