import numpy as np
import sys 
sys.path.append('..')
from Code import util, storage
from collections import defaultdict
import json
import os

def getLabelFeat(feat, id_labels):
    feat_mean = {}
    for label in id_labels:
        img_ids = id_labels[label]
        lab_all = []
        for id in img_ids:
            if id in feat:
                lab_all.append(feat[id])
        lab_all = np.array(lab_all)
        mean_lab_all = lab_all.mean(axis=0)
        feat_mean[label] = mean_lab_all
    return feat_mean

def load_dict(path,label_vec_dict) :
    for key,val in label_vec_dict.items() :

        with open(os.path.join(path, key+'.txt' ), 'w') as file:
            json.dump(val.tolist(), file)


db = storage.Database()

feat_desc = db.feature_descriptors

# feat_desc[util.Constants.COLOR_MOMENTS] = storage.Database.get_feature_descriptors(util.Constants.COLOR_MOMENTS)
# feat_desc[util.Constants.HOG] = storage.Database.get_feature_descriptors(util.Constants.HOG)
# feat_desc[util.Constants.ResNet_AvgPool_1024] = storage.Database.get_feature_descriptors(util.Constants.ResNet_AvgPool_1024)
# feat_desc[util.Constants.ResNet_FC_1000] = storage.Database.get_feature_descriptors(util.Constants.ResNet_FC_1000)
# feat_desc[util.Constants.ResNet_Layer3_1024] = storage.Database.get_feature_descriptors(util.Constants.ResNet_Layer3_1024)
label_dict = db.get_id_label_dict()

id_labels_dict = defaultdict(list)

for key,val in label_dict.items():
    id_labels_dict[val].append(key)

feat_mean_col_moments = getLabelFeat(feat_desc[util.Constants.COLOR_MOMENTS], id_labels_dict)
feat_mean_hog = getLabelFeat(feat_desc[util.Constants.HOG], id_labels_dict)
feat_mean_resnet_avgpool = getLabelFeat(feat_desc[util.Constants.ResNet_AvgPool_1024], id_labels_dict)
feat_mean_resnet_l3 = getLabelFeat(feat_desc[util.Constants.ResNet_Layer3_1024], id_labels_dict)
feat_mean_resnet_fc = getLabelFeat(feat_desc[util.Constants.ResNet_FC_1000], id_labels_dict)
feat_mean_resnet_sm = getLabelFeat(feat_desc[util.Constants.ResNet_SoftMax_1000], id_labels_dict)

load_dict(util.Constants.LBL_COLOR_MOMENTS_LOCATION,feat_mean_col_moments)
load_dict(util.Constants.LBL_HOG_LOCATION,feat_mean_hog)
load_dict(util.Constants.LBL_RESNET_LAYER3_1024_LOCATION,feat_mean_resnet_l3)
load_dict(util.Constants.LBL_RESNET_FC_1000_LOCATION,feat_mean_resnet_fc)
load_dict(util.Constants.LBL_RESNET_AVGPOOL_1024_LOCATION,feat_mean_resnet_avgpool)
load_dict(util.Constants.LBL_RESNET_SOFTMAX_1000_LOCATION, feat_mean_resnet_sm)