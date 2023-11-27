from util import Constants
from DimensionalityReductionTechniques import dimensionality_reduction_techniques as drt
from Tasks import task_util
import numpy as np
import torchvision
from heapq import nlargest
import util

def Execute(arguments, db):
    execute_internal(arguments['k'], db)

def execute_internal(k, db) :

    # imagenet_data = torchvision.datasets.Caltech101(root=Constants.CALTECH_DATASET_LOCATION, download=True)

    #Internal Parameters
    fd = Constants.ResNet_Layer3_1024 
    drt_c = Constants.SVD
    
    train_labels_fds = db.get_label_feature_descriptors(fd,True)
    # print(np.array(train_labels_fds.values()))

    matrix = []
    for label in train_labels_fds:
        matrix.append(train_labels_fds[label])
    

    # print( np.array(list(train_labels_fds.values())).shape)
    drt_ = drt.DRT(drt_c, np.array(matrix),k)
    
    latent_semantics_data = drt_.get_latent_semantics_data()
    ids = train_labels_fds.keys()

    # 4000 * 1000 
    test_image_fds = db.get_feature_descriptors(fd, False)

    label_k_dict = dict(zip(ids, task_util.get_imgs_k_mat(latent_semantics_data,drt_c)))

    image_label_dict = dict()
    cnt = 0
    for image_id in test_image_fds :
        cnt+=1
        # 1*1000
        feat_vec = test_image_fds[image_id]
        # 1*k
        feat_vec_k = task_util.get_one_k_mat(latent_semantics_data,drt_c, k, feat_vec)

        # imagenet_datsa[image_id][0].show()

        #compare with each label in label_k_dict
        similarity_dict = dict()
        for label in train_labels_fds :
            similarity_dict[label] = task_util.cosine_similarity(feat_vec_k,label_k_dict[label])
            # similarity_dict[label] = task_util.cosine_similarity(feat_vec,train_labels_fds[label])

    

        image_label_dict[image_id] = max(similarity_dict.items(), key = lambda x: x[1])[0]
        # image_label_dict[image_id] = nlargest(10,similarity_dict,key = similarity_dict.get)
        # if(cnt > 50) :
        #     break
    # print(image_label_dict)
    # print(image_label_dict)
    util.AnalysePredictions(db, image_label_dict)
        

    return