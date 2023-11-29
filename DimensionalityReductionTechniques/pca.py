import numpy as np

def inherenet_dimentionalityPCA(features) :
    features = np.array(features)
    
    cov_matrix = np.cov(features.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    np.sort(eigenvalues)
    total_varience = np.sum(eigenvalues**2)

    var_explained = 0
    iter = 0
    while(var_explained <= .99) :
        var_explained += eigenvalues[iter]**2/total_varience
        iter+=1

    return iter, var_explained