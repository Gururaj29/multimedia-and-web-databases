import util
from collections import deque
import numpy as np
import heapq
from FeatureDescriptors import similarity_measures

class DBScan:
    '''
        implements DBScan algorithm and also gives C significant clusters
    '''
    def __init__(self, data, c, min_samples, eps=10):
        self.c = c
        self.min_samples = min_samples
        self.eps = eps
        self.clusters = self.__dbscan(data)
        self.centroids = self.__calculateCentroids(data)
    
    def __dbscan(self, data):
        cluster_labels = {i:0 for i in data}  # 0 denotes an undefined label
        cluster_id = 0  # Start with cluster_id as 1, as 0 is used for undefined

        for point_idx in data:
            if cluster_labels[point_idx] != 0:  # Ignore already processed points
                continue

            neighbors = self.__getNeighbors(data, point_idx, self.eps)

            # If the point is not a core point, mark as noise (temporarily)
            if len(neighbors) < self.min_samples:
                cluster_labels[point_idx] = -1
            else:
                # Increment the cluster ID, because this is the start of a new cluster
                cluster_id += 1
                # Start expanding cluster from the core point
                self.__growCluster(
                    data, cluster_labels, point_idx, neighbors, cluster_id, self.eps, self.min_samples
                )

        return cluster_labels
    
    def __getNeighbors(self, data, point_idx, eps):
        neighbors = []
        for idx, current_point in data.items():
            if util.euclidean_distance(data[point_idx], current_point) < eps:
                neighbors.append(idx)
        return neighbors

    def __growCluster(self, data, labels, point_idx, neighbors, cluster_id, eps, min_samples):
        # Label initial point as a core point of the cluster
        labels[point_idx] = cluster_id
        neighbors = deque(
            neighbors
        )  # Convert list to a deque for efficient popleft operation
        while neighbors:
            neighbor_idx = neighbors.popleft()
            if labels[neighbor_idx] == -1:  # Previously labeled as noise
                labels[neighbor_idx] = cluster_id  # Noise becomes edge point
            elif labels[neighbor_idx] == 0:  # Not yet visited
                labels[neighbor_idx] = cluster_id  # Label as part of the cluster
                neighbor_neighbors = self.__getNeighbors(data, neighbor_idx, eps)
                # If neighbor is a core point, append its neighbors to the queue for further checking
                if len(neighbor_neighbors) >= min_samples:
                    for n_neighbor_idx in neighbor_neighbors:
                        if labels[n_neighbor_idx] <= 0:  # Not yet visited or noise
                            neighbors.append(n_neighbor_idx)
    
    def __calculateCentroids(self, data):
        cluster_points = {}
        for id, point in data.items():
            cluster_id = self.clusters[id]
            if cluster_id == -1: continue
            if cluster_id not in cluster_points: cluster_points[cluster_id] = []
            cluster_points[cluster_id].append(point)
        return {cluster_id: np.mean(cluster_points[cluster_id], axis=0) for cluster_id in cluster_points}

    def getCSignicantClusters(self, point):
        most_significant_clusters = heapq.nlargest(self.c, self.centroids, key=lambda x: similarity_measures.l2_norm(point, self.centroids[x]))
        return most_significant_clusters
