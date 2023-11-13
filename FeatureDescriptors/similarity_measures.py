# libraries
import numpy

# internal packages
from util import Constants

# l1 norm distance measure, since we want similarity measure output is the negation of the l1 norm
def l1_norm(v1, v2):
    return (sum([abs(v1[i]-v2[i]) for i in range(len(v1))])) * -1

# l2 norm distance measure, since we want similarity measure output is the negation of the l2 norm
def l2_norm(v1, v2):
    return (numpy.linalg.norm(numpy.array(v1) - numpy.array(v2))) * -1

# l max norm distance measure, since we want similarity measure output is the negation of the l max norm
def l_max(v1, v2):
    return (max([abs(v1[i]-v2[i]) for i in range(len(v1))])) * -1

# intersection similarity measure
def intersection(v1, v2):
    min_count, max_count = 0, 0
    for i in range(len(v1)):
        min_count += min(v1[i], v2[i])
        max_count += max(v1[i], v2[i])
    return min_count/max_count

# cosine similarity measure
def cosine_similarity(v1, v2):
    return numpy.dot(v1,v2)/(numpy.linalg.norm(v1)*numpy.linalg.norm(v2))

# map of similarity measure name to the function
similarity_function_map = {
    Constants.L1_NORM: l1_norm,
    Constants.L2_NORM: l2_norm,
    Constants.L_MAX: l_max,
    Constants.INTERSECTION: intersection,
    Constants.COSINE_SIMILARITY: cosine_similarity
}

def get_similarity_measure_func(similarity_measure_id):
    return similarity_function_map[similarity_measure_id]