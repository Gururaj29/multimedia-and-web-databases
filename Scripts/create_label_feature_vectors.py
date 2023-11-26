import numpy as np
import util, storage
from collections import defaultdict
import json
import os
import pathlib

def getLabelFeat(db, fd, id_labels, isEvenNumbered):
    feat = db.get_feature_descriptors(fd, isEvenNumbered)
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

def load_dict(db, fd, id_labels, isEvenNumbered):
    datasetType = util.Constants.DatasetTypeTrain if isEvenNumbered else util.Constants.DatasetTypeTest
    path = os.path.join(os.getcwd(), "Outputs", datasetType, "labels", fd)
    label_vec_dict = getLabelFeat(db, fd, id_labels, isEvenNumbered)
    for key,val in label_vec_dict.items():
        with open(os.path.join(path, key+'.txt' ), 'w') as file:
            json.dump(val.tolist(), file)

def load_labels(db, fds, isEvenNumbered=True):
    label_dict = db.get_id_label_dict(isEvenNumbered)

    id_labels_dict = defaultdict(list)

    for key,val in label_dict.items():
        id_labels_dict[val].append(key)

    for fd in fds:
        load_dict(db, fd, id_labels_dict, isEvenNumbered)

def run_script():
    fds = [
        util.Constants.COLOR_MOMENTS,
        util.Constants.HOG,
        util.Constants.ResNet_AvgPool_1024,
        util.Constants.ResNet_FC_1000,
        util.Constants.ResNet_Layer3_1024,
        util.Constants.ResNet_SoftMax_1000
    ]
    for fd in fds:
        pathlib.Path(os.path.join(os.getcwd(), "Outputs", util.Constants.DatasetTypeTrain, "labels", fd)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(os.getcwd(), "Outputs", util.Constants.DatasetTypeTest, "labels", fd)).mkdir(parents=True, exist_ok=True)
    
    db = storage.Database()

    load_labels(db, fds, isEvenNumbered=True)
    load_labels(db, fds, isEvenNumbered=False)