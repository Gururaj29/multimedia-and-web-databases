import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation

class LDA:
    """Class for dimensionality reduction using LDA"""

    def __init__(self, mat, k):
        
        self.k = k

        # Check if the input matrix contains negative values
        if (np.array(mat) < 0).any():
            # Normalize the input matrix using Min-Max scaling
            print("Using normalization as data contain negative value")
            min_max_scaler = MinMaxScaler()
            normalized_mat = min_max_scaler.fit_transform(np.array(mat))
        else:
            # Use the original data if it doesn't contain negative values
            normalized_mat = np.array(mat)

        self.lda_model = LatentDirichletAllocation(n_components=k, random_state=0)
        self.latent_features = self.lda_model.fit_transform(normalized_mat)

    def get_latent_semantics_data(self):
        ''' returns a matrix of top-k latent features and latent semantics data for analysis'''
        return self.lda_model.components_, (self.lda_model, self.latent_features)
      
    def get_object_id_weight_pair(self, ids):
        '''returns dataframe of object_id, weight pairs in the decreasing order'''

         # Create DataFrame for sorted
        df = pd.DataFrame(self.latent_features, index = ids)

        # Created sorted df based on all columns in the decreasing order
        sorted_df =  df.sort_values(by=list(range(self.k)), axis=0, ascending=False)

        # Create object-id-weight pairs in decreasing order, considering the first column as the weight
        object_id_weight_pairs = [(id, df.loc[:, 0].loc[id]) for id in sorted_df.index]

        # Create a Pandas DataFrame for the sorted pairs
        df = pd.DataFrame(object_id_weight_pairs, columns=['Object_ID', 'Weight'])

        return df