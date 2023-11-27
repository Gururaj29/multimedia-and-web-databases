from util import Constants
import random

class SVM:
    def __init__(self):
        return
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        # TODO
    
    def predict(self, X):
        # TODO: using random classification for now
        p = 0.2 # probability of irrelevance
        if random.random() <= p:
            return random.choice([Constants.VeryIrrelevant, Constants.Irrelevant])
        else:
            return random.choice([Constants.Relevant, Constants.VeryRelevant])