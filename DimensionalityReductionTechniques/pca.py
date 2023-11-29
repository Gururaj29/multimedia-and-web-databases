import numpy as np
import pandas as pd
import random
from DimensionalityReductionTechniques import svd

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
        # print(var_explained)
        iter+=1
    # var = np.sum(cov_matrix.diagonal())
    # iter = 2
    # while(iter < 100) :
    #     svd_ = svd.SVD(features,iter)
    #     ls_data = svd_.get_latent_semantics_data()
    #     cov_matrix_new = ls_data[1][0] @ ls_data[1][2]
    #     new_var = np.sum(cov_matrix_new.diagonal())
    #     varience_preserve = (var - new_var)*100/var
    #     print(varience_preserve)
    #     if(varience_preserve > 99) :
    #         return iter,varience_preserve
    #         break
    #     iter+=1

    return iter, var_explained