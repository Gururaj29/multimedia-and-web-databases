import random
class PPR:
    '''
        Placeholder class for personalized pagerank classifier
    '''
    def __init__(self, random_jump_probability = 0.15):
        self.p =  random_jump_probability
        return

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        return random.choice(self.Y)
