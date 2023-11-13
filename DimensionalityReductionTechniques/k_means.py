# from sklearn.cluster import KMeans
## mentioned not to use inbuilt library

# k-means steps
## randomly initialize cluster centers
## for each data point, compute the euclidean distance from each centroid
#### assign to nearest centroid cluster
## adjust centroid as mean of cluster
## repeat till loss stabilized


import random
import numpy as np
import pandas as pd


class KMeans:
    """Class for dimensionality reduction using K-means"""

    def __init__(self, mat, k):
        self.mat = np.array(mat)  # feature matrix of size mxn
        self.k = k  # number of clusters
        self.centroids = np.array(
            [self.mat[i] for i in random.sample(range(self.mat.shape[1]), self.k)]
        )  # randomly initialized k cluster centers
        self.assigned_clusters = np.zeros((self.mat.shape[0]))  # assigned centroids
        self.mat_k = self.get_fd_latent_semantics()

    def assign_clusters(self):
        flag = True
        sse = [np.inf]

        while flag:
            node_cluster_assigned = np.zeros((self.mat.shape[0],))

            # for each point, calculate euclidean distance to each centroid
            # assign to nearest centroid

            for i in range(self.mat.shape[0]):
                euclidean_dist = np.inf
                for j in range(self.k):
                    if (
                        np.linalg.norm(np.subtract(self.mat[i], self.centroids[j]))
                        < euclidean_dist
                    ):
                        euclidean_dist = np.linalg.norm(
                            np.subtract(self.mat[i], self.centroids[j])
                        )
                        node_cluster_assigned[i] = j

            # new centroids
            self.centroids = np.zeros((self.k, self.mat.shape[1]))
            num_nodes_per_cluster = np.zeros((self.k,))
            for i in range(self.mat.shape[0]):
                self.centroids[int(node_cluster_assigned[i])] += self.mat[i]
                num_nodes_per_cluster[int(node_cluster_assigned[i])] += 1

            self.centroids /= num_nodes_per_cluster.reshape(-1, 1)

            # calculate sse
            i_sse = 0
            for i in range(self.mat.shape[0]):
                i_sse += np.linalg.norm(
                    np.subtract(
                        self.mat[i], self.centroids[int(node_cluster_assigned[i])]
                    )
                )

            # breaking condition - if sse change is less than 5%
            if sse[-1] - i_sse < 0.05 * sse[-1]:
                sse.append(i_sse)
                flag = False

            sse.append(i_sse)

        self.assigned_clusters = node_cluster_assigned

    def get_fd_latent_semantics(self):
        mat_k = np.zeros((self.mat.shape[0], self.k))
        for i in range(self.mat.shape[0]):
            for j in range(self.k):
                mat_k[i][j] = np.linalg.norm(
                    np.subtract(self.mat[i], self.centroids[j])
                )

        return mat_k

    def get_latent_semantics_data(self):
        ''' returns a matrix of top-k latent features and latent semantics data for analysis'''
        return self.centroids, (self.centroids, self.mat_k)

    def get_object_id_weight_pair(self, ids):
        """returns dataframe of object_id, weight pairs in the decreasing order"""

        # Create DataFrame for sorted
        df = pd.DataFrame(self.mat_k, index=ids)

        # Created sorted df based on all columns in the decreasing order
        sorted_df = df.sort_values(by=list(range(self.k)), axis=0, ascending=False)

        # Create object-id-weight pairs in decreasing order, considering the first column as the weight
        object_id_weight_pairs = [(id, df.loc[:, 0].loc[id]) for id in sorted_df.index]

        # Create a Pandas DataFrame for the sorted pairs
        df = pd.DataFrame(object_id_weight_pairs, columns=["Object_ID", "Weight"])

        return df
