import os
import math
import numpy as np

from PIL import Image

class Constants:
    """class for all the constants used across this application"""

    # feature descriptor constants
    COLOR_MOMENTS = "color_moments"
    HOG = "hog"
    ResNet_AvgPool_1024 = "resnet_avgpool_1024"
    ResNet_Layer3_1024 = "resnet_layer3_1024"
    ResNet_FC_1000 = "resnet_fc_1000"
    ResNet_SoftMax_1000 = "resnet_sm_1000"

    # dimensionality reduction technique constants
    SVD = "svd"
    NNMF = "nnmf"
    LDA = "lda"
    KMeans = "k_means"
    CP = "cp"

    # latent semantic constants
    LS1 = "ls1"
    LS2 = "ls2"
    LS3 = "ls3"
    LS4 = "ls4"

    # similarity matrix constants
    LL = "label_label"

    # similarity measure constants
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    L_MAX = "l_max"
    COSINE_SIMILARITY = "cosine_similarity"
    INTERSECTION = "intersection"

    #storage variables
    PATH_REPO = os.getcwd()    

    #Caltech Data Path
    CALTECH_DATASET_LOCATION = os.path.join(PATH_REPO, "Data")

    #feature vectors
    FEAT_VECTORS_LOCATION = os.path.join(PATH_REPO,"Outputs","features")
    FEAT_COLOR_MOMENTS_LOCATION = os.path.join(FEAT_VECTORS_LOCATION,COLOR_MOMENTS)
    FEAT_HOG_LOCATION = os.path.join(FEAT_VECTORS_LOCATION,HOG)
    FEAT_RESNET_LAYER3_1024_LOCATION = os.path.join(FEAT_VECTORS_LOCATION,ResNet_Layer3_1024)
    FEAT_RESNET_FC_1000_LOCATION = os.path.join(FEAT_VECTORS_LOCATION,ResNet_FC_1000)
    FEAT_RESNET_AVGPOOL_1024_LOCATION = os.path.join(FEAT_VECTORS_LOCATION, ResNet_AvgPool_1024)
    FEAT_RESNET_SOFTMAX_1000_LOCATION = os.path.join(FEAT_VECTORS_LOCATION, ResNet_SoftMax_1000)

    #labels
    LBL_FEAT_VECTORS_LOCATION = os.path.join(PATH_REPO,"Outputs","labels")
    LBL_COLOR_MOMENTS_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION,COLOR_MOMENTS)
    LBL_HOG_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION,HOG)
    LBL_RESNET_LAYER3_1024_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION,ResNet_Layer3_1024)
    LBL_RESNET_FC_1000_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION,ResNet_FC_1000)
    LBL_RESNET_AVGPOOL_1024_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION, ResNet_AvgPool_1024)
    LBL_RESNET_SOFTMAX_1000_LOCATION = os.path.join(LBL_FEAT_VECTORS_LOCATION, ResNet_SoftMax_1000)

    # task_outputs
    TASK_0_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_0")
    TASK_1_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_1")
    TASK_2_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_2")
    TASK_3_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_3")
    TASK_7_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_7")
    TASK_8_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_8")
    TASK_9_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_9")
    TASK_10_LOCATION = os.path.join(PATH_REPO, "Outputs", "tasks", "task_10")


def fd_cli_to_constants(fd):
    fd_cli_to_constants_map = {
        "CM": Constants.COLOR_MOMENTS,
        "HOG": Constants.HOG,
        "AvgPool": Constants.ResNet_AvgPool_1024,
        "L3": Constants.ResNet_Layer3_1024,
        "FC": Constants.ResNet_FC_1000,
        None: Constants.ResNet_SoftMax_1000,
        "RESNET": Constants.ResNet_SoftMax_1000,
    }
    return fd_cli_to_constants_map[fd]

def drt_cli_to_constants(drt):
    drt_cli_to_constants_map = {
        "SVD": Constants.SVD,
        "kmeans": Constants.KMeans,
        "LDA": Constants.LDA,
        "NNMF": Constants.NNMF,
        None: Constants.CP
    }
    return drt_cli_to_constants_map[drt]

def cube_root(x) :
    if(x >= 0) :
        return x**(1./3.)
    else :
        return -((-x)**(1./3.))
    
def chk(i,j,n,m) :
    if(i >= 0 and j >= 0 and i<n and j<m) :
        return True
    else :
        return False
    
def normalize(bin_) :
    bin_ = list(bin_)
    norm_ = 0
    for x in bin_ :
        norm_ += x**2
    norm_ = math.sqrt(norm_)
    #print(norm_)
    for i in range(len(bin_)) :
        bin_[i] = bin_[i] / norm_
    return bin_

def check_image(image) :
    img_array = np.array(image)
    #print(len(img_array.shape))
    if(len(img_array.shape) == 3) :
        return True
    return False