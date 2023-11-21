import random
from functools import reduce
import numpy as np
import operator

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

            def fetchNext(self):
                hashId = 0
                for hashSearch in self.hashSearches:
                    while True:
                        # run loop while we find some new objects
                        hashResults = hashSearch.fetchNext()
                        if len(hashResults) != 0:
                            break
                    self.searchResults[hashId] |= hashResults
                    hashId += 1
                return self.getLayerSearchResults()

class LSH:
    def __init__(self, data, h, l, w, dim):
        self.hashLayers = [HashLayer(h, data, w, dim) for i in range(l)]
    
    def search(self, data, query_vector, k, show_images=False):
        return self.__search(data, query_vector, k)
    
    def __getSearchResults(self, layerSearchResults):
        return reduce(operator.or_, layerSearchResults)

    def __search(self, data, query_vector, k):
        layerSearches = [layer.search(query_vector) for layer in self.hashLayers]
        remainingQueryResults = k        
        while remainingQueryResults >= 0:
            # iterate until we get at least k results
            layerResults = [search.fetchNext() for search in layerSearches]
            results = self.__getSearchResults(layerResults)
            remainingQueryResults = k - len(results)
        # if we received more than k results, sort them based on L2 norm and take the closest k
        bestKResults = sorted([(object_id, np.linalg.norm(data[object_id] - query_vector)) for object_id in results], key=lambda x: x[1])[:k]
        return [i[0] for i in bestKResults]
        




