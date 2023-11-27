import heapq
from collections import Counter
from FeatureDescriptors import similarity_measures

class NN:
    '''
        k-NN classifier
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        '''
            Stores the train data
            X:  
        '''
        self.train_data = [(x, y) for x, y in zip(X, Y)]
    
    def predict(self, X):
        '''
            predicts the class of the new object by taking the mode of the k-closest neighbour labels
        '''
        k_closest_neighbors = heapq.nlargest(self.k, self.train_data, key=lambda x: similarity_measures.l2_norm(X, x[0]))
        label_counter = Counter(map(lambda x: x[1], k_closest_neighbors))
        return label_counter.most_common(1)[0][0]
