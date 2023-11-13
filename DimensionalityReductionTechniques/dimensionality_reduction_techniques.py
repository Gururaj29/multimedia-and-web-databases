from DimensionalityReductionTechniques import k_means
from DimensionalityReductionTechniques import lda
from DimensionalityReductionTechniques import nnmf
from DimensionalityReductionTechniques import svd
from util import Constants

def DRT(drt, mat, k):
    if drt == Constants.SVD:
        return svd.SVD(mat, k)
    elif drt == Constants.KMeans:
        return k_means.KMeans(mat, k)
    elif drt == Constants.LDA:
        return lda.LDA(mat, k)
    elif drt == Constants.NNMF:
        return nnmf.NNMF(mat, k)
    return None