import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor
from collections import OrderedDict 
import numpy as np

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

class MultiOutputModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out
    

model = MultiOutputModel(output_layers = [6,8,9])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def resnet50_feats(image):
    avg_pool_1024, l3_1024, fc_1000 = [], [], []
    if 0:
        return avg_pool_1024, l3_1024, fc_1000
    imgs = image.resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor()# ,
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imgs_tensor = transform(imgs)
    out, layerOut = model(imgs_tensor.unsqueeze(0))
    
    avg_pool_1024 = []
    if 'avgpool' in layerOut:
        avg_feats = layerOut['avgpool']
        avg_feats = avg_feats.detach().numpy()

        for i in range(0, len(avg_feats[0]), 2):
            val = (avg_feats[0][i][0][0] + avg_feats[0][i+1][0][0]) /2
            avg_pool_1024.append(val)
    
    l3_1024 = []
    if 'layer3' in layerOut:
        l3_feats = layerOut['layer3']
        l3_feats = l3_feats.detach().numpy()
        for i in range(len(l3_feats[0])):
            grid = l3_feats[0][i]
            l3_1024.append(np.mean(grid))
    
    fc_1000 = []
    if 'fc' in layerOut:
        fc_feats = layerOut['fc']
        fc_feats = fc_feats.detach().numpy()
        fc_1000 = fc_feats[0]
        
    return avg_pool_1024, l3_1024, fc_1000