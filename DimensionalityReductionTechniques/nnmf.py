import time
import numpy as np
from pandas import DataFrame

class NNMF:
    """Class for dimensionality reduction using NNMF"""

    def __nnmf(self, mat, k, n_iterations, debug):
        def updateH(W, H, V):
            a, b = W.T.dot(V), W.T.dot(W).dot(H) + 1e-10 # adding small double to avoid division by zero
            return H*(a / b)
        
        def updateW(W, H, V):
            a, b = V.dot(H.T), W.dot(H).dot(H.T) + 1e-10 # adding small double to avoid division by zero
            return W*(a / b)
        
        def calculate_loss(V, W, H):
            # square of Euclidian distance between V and WH
            return sum((V - W.dot(H)).flatten()**2)
        
        V = np.array(mat)
        # input matrix dimensions
        n, m = V.shape

        # initial random values
        W = np.abs(np.random.randn(1, n, k))[0] # dim: n * k
        H = np.abs(np.random.randn(1, k, m))[0] # dim: k * m
        loss = calculate_loss(V, W, H) # initial loss

        start_time = time.time()
        for i in range(n_iterations):
            H = updateH(W, H, V)
            W = updateW(W, H, V)
            
            new_loss = sum((V - W.dot(H)).flatten()**2)
            if debug:
                print("Loss after %d iterations: %d. Delta loss: %d." % (i, new_loss, loss-new_loss))
            loss = new_loss
        end_time = time.time()
        if debug:
            print("Ran NNMF in %f seconds" %(end_time - start_time))

        return W, H

    def __init__(self, mat, k, n_iterations= 100, debug = False):
        self.W, self.H = self.__nnmf(mat, k, n_iterations, debug)
    
    def get_latent_semantics_data(self):
        ''' returns a matrix of top-k latent features and latent semantics data for analysis'''
        return self.H, (self.W, self.H)
    
    def get_object_id_weight_pair(self, object_ids):
        '''returns dataframe of object_id, weight pairs in the decreasing order'''

        # L2 norm for each row of matrix W
        object_weights = np.linalg.norm(self.W, axis = 1)

        # Create object-id-weight pairs in decreasing order
        object_id_weight_pairs = [(object_ids[i], object_weights[i]) for i in range(len(object_ids))]

        # Sort the pairs by weight in decreasing order
        object_id_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        return DataFrame(object_id_weight_pairs, columns=['Object_ID', 'Weight'])