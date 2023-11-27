import numpy as np
import pandas as pd
import random

class SVD:
    """Class for SVD dimensionality reduction technique"""

    def __init__(self, mat, k):
        data_matrix = np.array(mat)
        self.k = k
        self.U, self.S = self.__get_object_factor_matrix(data_matrix, k)
        self.V = self.__get_feature_factor_matrix(data_matrix, k)
        # self.U, self.S, self.V = np.linalg.svd(mat, full_matrices=False)
        self.U = self.U[:,:k]
        self.S = self.S[:k]
        self.V = self.V[:k,:]
#         print(self.choose_sign(data_matrix))
        (self.U,self.V) = self.choose_sign(data_matrix)
        self.U = self.U @ np.diag(self.S)
#         print(self.U,self.V)
        
        
    def change_sign_c(self,matrix, column_index):
        for row in matrix:
            row[column_index] = -row[column_index]
    def change_sign_r(self,matrix,row_index) :
        matrix[row_index] = [-element for element in matrix[row_index]]

    def choose_sign(self,data_matrix) :
        ls = self.get_latent_semantics_data()
        U,S,V = ls[1][0].copy(), ls[1][1].copy(), ls[1][2].copy()
        config = (U,V)
        A_ = U @ np.diag(S) @ V
    #     print(A_)
        config_cost = np.linalg.norm(A_ - data_matrix)
    #     seq = range(1,k)
        for xx in range(1,1000) :  
#             print(xx,config_cost)
#             print(config)
            n_u,n_v = random.randint(1,self.k-1),random.randint(1,self.k-1)
            l_u,l_v = random.sample(range(1, self.k+1), n_u),random.sample(range(1, self.k+1), n_v)
            U_,V_ = U.copy(),V.copy()

            for i in l_u :
                self.change_sign_c(U_,i-1)
            for j in l_v :
                self.change_sign_r(V_,j-1)

            A_ = U_ @ np.diag(S[:self.k]) @ V_
            cc = np.linalg.norm(A_ - data_matrix)
            if(cc < config_cost) :
#                 print(cc)
                config_cost = cc
                config = (U_,V_)
#                 print(config)
#         print(config_cost)
#         print(config)
        return config


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
        # non_zero_eigenvectors = eigenvectors[:, eigenvalues > 1e-10]

        #Select Top K latent semantics
        selected_eigenvectors = eigenvectors[:, :k]
        S = np.sqrt(eigenvalues)

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
        # non_zero_eigenvectors = eigenvectors[:, eigenvalues > 1e-10]

        #Select Top K latent semantics
        selected_eigenvectors = eigenvectors[:, :k]

        return np.array(selected_eigenvectors).T

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