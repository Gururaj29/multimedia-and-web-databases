from color_moments import get_ColorMoments
from HOG import get_HOG
from resnet_features import resnet50_feats
from resnet_softmax_features import resnet50_sm
from Code import util
import torch,torchvision
import os
import numpy as np
import json
from tqdm import tqdm

# Reading from a JSON file
imagenet_data = torchvision.datasets.Caltech101(root=util.Constants.CALTECH_DATASET_LOCATION, download=True)

for i in tqdm(range(0,len(imagenet_data),2)):
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
    with open(os.path.join(util.Constants.FEAT_COLOR_MOMENTS_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(color_moments.tolist(), file)
    with open(os.path.join(util.Constants.FEAT_HOG_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(hog_, file)
    with open(os.path.join(util.Constants.FEAT_RESNET_AVGPOOL_1024_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(avg_pool_1024, file)
    with open(os.path.join(util.Constants.FEAT_RESNET_LAYER3_1024_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(l3_1024.tolist(), file)
    with open(os.path.join(util.Constants.FEAT_RESNET_FC_1000_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(fc_1000.tolist(), file)
    with open(os.path.join(util.Constants.FEAT_RESNET_SOFTMAX_1000_LOCATION,str(i)+"_"+label+'.txt'), 'w') as file:
        json.dump(rn_sm_1000.tolist(), file)