import random
from functools import reduce
import numpy as np
import operator
import heapq
from FeatureDescriptors import similarity_measures

class HashBuckets(dict):
    def __getitem__(self, __key):
        return super().get(__key, set())
    
class Hash:
    def __init__(self, vectors, w, dim):
        self.w = w
        self.random_line = self.__get_random_line(dim)
        self.buckets = self.__prepare_buckets(vectors)
    
    def __get_random_line(self, dim):
        vec = np.zeros(dim) # dim has to be greater than or equal to 2
        vec[0], vec[1] = random.random(), random.random()
        random.shuffle(vec)
        return vec / np.linalg.norm(vec) # return unit vector

    def __get_bucket_id(self, vector):
        projection = vector.dot(self.random_line) # since random line is a unit vector, dot product is the projection
        return projection//self.w

    def __prepare_buckets(self, vectors):
        buckets = HashBuckets()
        for id in vectors:
            buckets[self.__get_bucket_id(vectors[id])] |=  {id,}
        return buckets

    def getBucket(self, bucket_id):
        return self.buckets[bucket_id]

    def search(self, query_vector):
        return Hash.Search(self, self.__get_bucket_id(query_vector))

    class Search:
        def __init__(self, hash, start):
            self.searchRange = []
            self.direction = None
            self.start = start
            self.hash = hash

        def fetchNext(self):
            left, right = (-1, 0), (0, 1)
            if not self.searchRange:
                self.searchRange = (self.start, self.start)
                return self.hash.getBucket(self.start)
            if not self.direction:
                # select random direction in the beginning of expansion
                self.direction = random.choice([left, right])
            next_bucket_id = self.searchRange[0] - 1 if self.direction is left else self.searchRange[1] + 1
            self.searchRange = (self.searchRange[0]+self.direction[0], self.searchRange[1]+self.direction[1])
            self.direction = left if self.direction is right else right # toggle the direction
            return self.hash.getBucket(next_bucket_id)

class HashLayer:
    def __init__(self, h, vectors, w, dim):
        self.hashFunctions = [Hash(vectors, w, dim) for i in range(h)]
    
    def getHashFunctions(self):
        return self.hashFunctions

    def search(self, query_vector):
        return HashLayer.Search(self, query_vector)

    class Search:
        def __init__(self, hashLayer, query_vector):
            self.hashSearches = [hash.search(query_vector) for hash in hashLayer.getHashFunctions()]
            self.searchResults = [set() for i in range(len(self.hashSearches))]
        
        def getLayerSearchResults(self):
            return reduce(operator.and_, self.searchResults)
        
        def getOverallNumberOfImagesFetched(self):
            return sum([len(results) for results in self.searchResults])
        
        def getUniqueSearchImages(self):
            return reduce(operator.or_, self.searchResults)

        def fetchNext(self):
            hashId = 0
            for hashSearch in self.hashSearches:
                while True:
                    # run loop until we find some new objects
                    hashResults = hashSearch.fetchNext()
                    if len(hashResults) != 0:
                        break
                self.searchResults[hashId] |= hashResults
                hashId += 1
            return self.getLayerSearchResults()

class LSH:
    def __init__(self, data, dim, l=10, h=10, w=0.001):
        self.hashLayers = [HashLayer(h, data, w, dim) for i in range(l)]
    
    def search(self, query_vector, k):
        return LSH.Search(query_vector, k, self.hashLayers)
    
    class Search:
        def __init__(self, query_vector, k, hashLayers):
            self.query_vector = query_vector
            self.hashLayers = hashLayers
            self.layerSearches = [layer.search(query_vector) for layer in self.hashLayers]
            self.k = k
            self.results = set()
            self.exclude_set = set()
    
        def fetchNext(self):
            layerResults = [search.fetchNext() for search in self.layerSearches]
            return reduce(operator.or_, layerResults)
        
        def getOverallNumberOfImagesFetched(self):
            return sum([layerSearch.getOverallNumberOfImagesFetched() for layerSearch in self.layerSearches])
        
        def getUniqueNumberofImagesFetched(self):
            return len(reduce(operator.or_, map(lambda x: x.getUniqueSearchImages(), self.layerSearches)))
        
        def getFilteredResults(self, exclude_set=()):
            exclude_set = exclude_set if exclude_set else self.exclude_set
            return self.results - self.exclude_set

        def search(self, exclude_set=set()):
            self.exclude_set = exclude_set
            while len(self.getFilteredResults()) < self.k:
                # iterate until we get at least k results
                self.results = self.fetchNext()
            return self.getFilteredResults()
        
        def bestKResults(self, data, exclude_set=set()):
            if not self.results: self.search()
            return heapq.nlargest(self.k, self.getFilteredResults(exclude_set), key=lambda x: similarity_measures.l2_norm(data[x], self.query_vector))
        
        def printSearchAnalytics(self):
            if not self.results: self.search()
            print("Overall number of images fetched by all hash functions: ", self.getOverallNumberOfImagesFetched())
            print("Unique number of images fetched by all hash functions: ", self.getUniqueNumberofImagesFetched())
            print("Overall number of images returned by LSH: ", len(self.results))
            print("Overall number of images excluded in the final search result: ", len(self.exclude_set))
        




