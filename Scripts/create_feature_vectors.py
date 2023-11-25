from FeatureDescriptors.color_moments import get_ColorMoments
from FeatureDescriptors.HOG import get_HOG
from FeatureDescriptors.resnet_features import resnet50_feats
from FeatureDescriptors.resnet_softmax_features import resnet50_sm
import util
import torch,torchvision
import os
import numpy as np
import json
from tqdm import tqdm
import pathlib

# Reading from a JSON file
imagenet_data = torchvision.datasets.Caltech101(root=util.Constants.CALTECH_DATASET_LOCATION, download=True)

def get_path(fd, evenNumbered=True):
    datasetType = util.Constants.DatasetTypeTrain if evenNumbered else util.Constants.DatasetTypeTest
    path = os.path.join(os.getcwd(), 'Outputs', datasetType, 'features', fd)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

def extract_feature_vectors(isEvenNumbered=True):
    start_index = 0 if isEvenNumbered else 1
    for i in tqdm(range(start_index,len(imagenet_data),2)):
        image = imagenet_data[i][0]
        label = imagenet_data[i][0].filename.split('/')[-2]

        #check if image has 3 channels or not
        if(util.check_image(imagenet_data[i][0]) == False) :
            image = image.convert('RGB')

        #get color_moments of given image
        color_moments = get_ColorMoments(image)

        #print(color_moments)
        
        #get HOG vectors of given image
        hog_ = get_HOG(image)
        
        #get resnet vectors 
        avg_pool_1024, l3_1024, fc_1000 = resnet50_feats(image)
        l3_1024 = np.array(l3_1024)

        # get resnet50 feature vectors using Softmax activation
        rn_sm_1000 = resnet50_sm(image)

        #save features to respective files
        with open(os.path.join(get_path(util.Constants.COLOR_MOMENTS, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(color_moments.tolist(), file)
        with open(os.path.join(get_path(util.Constants.HOG, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(hog_, file)
        with open(os.path.join(get_path(util.Constants.ResNet_AvgPool_1024, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(avg_pool_1024, file)
        with open(os.path.join(get_path(util.Constants.ResNet_Layer3_1024, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(l3_1024.tolist(), file)
        with open(os.path.join(get_path(util.Constants.ResNet_FC_1000, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(fc_1000.tolist(), file)
        with open(os.path.join(get_path(util.Constants.ResNet_SoftMax_1000, isEvenNumbered),str(i)+"_"+label+'.txt'), 'w') as file:
            json.dump(rn_sm_1000.tolist(), file)

def process_odd_numbered_images():
    extract_feature_vectors(isEvenNumbered=False)

def process_even_numbered_images():
    extract_feature_vectors(isEvenNumbered=True)

def run_script():
    process_even_numbered_images()
    process_odd_numbered_images()