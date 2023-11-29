from util import Constants
from DimensionalityReductionTechniques import dimensionality_reduction_techniques as drt
from Tasks import task_util
import numpy as np
import util
import os 
import csv

def Execute(arguments, db):
    execute_internal(arguments['k'], db)

def execute_internal(k, db) :

    # Internal Parameters
    fd = Constants.ResNet_Layer3_1024 
    drt_c = Constants.SVD
    
    train_labels_fds = db.get_label_feature_descriptors(fd,True)

    matrix = []
    for label in train_labels_fds:
        matrix.append(train_labels_fds[label])
    
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

        # compare with each label in label_k_dict
        similarity_dict = dict()
        for label in train_labels_fds :
            similarity_dict[label] = task_util.cosine_similarity(feat_vec_k,label_k_dict[label])
    
        image_label_dict[image_id] = max(similarity_dict.items(), key = lambda x: x[1])[0]
    
    if(not os.path.exists(Constants.TASK_1_LOCATION)) :
        os.makedirs(Constants.TASK_1_LOCATION)

    true_labels = db.get_id_label_dict(False)

    with open(os.path.join(Constants.TASK_1_LOCATION,"task1_k_"+str(k)+".csv"), 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'MostLikelyLabel', 'TrueLabel']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key in image_label_dict:
            writer.writerow({'image_id': key, 'MostLikelyLabel': image_label_dict[key], 'TrueLabel': true_labels[key]})

    util.AnalysePredictions(db, image_label_dict)