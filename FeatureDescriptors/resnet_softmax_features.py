import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor
from collections import OrderedDict 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resnet50_sm(image):

    imgs = image.resize((224, 224))
    transform = transforms.Compose([
        ToTensor()
    ])
    imgs_tensor = transform(imgs).unsqueeze(0)
    model = models.resnet50(pretrained=True)
    model.to(device)

    with torch.no_grad():
        rn_feature = model(imgs_tensor)
    rn_sm_1000 = nn.Softmax()(rn_feature)

    rn_sm_1000 = rn_sm_1000.detach().numpy()
    rn_sm_1000 = rn_sm_1000[0]

    return rn_sm_1000