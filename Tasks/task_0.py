from util import Constants
from MDS import mds
from DimensionalityReductionTechniques import pca

def Execute(arguments, db):
    execute_internal(arguments, db)

def execute_internal(arguments, db) :
    image_fd_data = db.get_feature_descriptors(Constants.ResNet_FC_1000)
    image_data = [image_fd_data[i] for i in image_fd_data]
    id, var_preserve = pca.inherenet_dimentionalityPCA(image_data)
    # dim, sa, new_image_data = mds.MDS(image_data)
    print("Inherent dimensionality of even numbered images :", id)

    label_fd_data = db.get_label_feature_descriptors(Constants.ResNet_FC_1000)
    label_data = [label_fd_data[i] for i in label_fd_data]
    id, var_preserve = pca.inherenet_dimentionalityPCA(label_data)
    # dim, sa, new_label_data = mds.MDS(label_data)
    print("Inherent dimensionality of unique labels of even numbered images :", id)