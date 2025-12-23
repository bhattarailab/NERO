# load both vit and resnet18

import os
import timm
import torch
from torchvision.models import resnet18
import torch.nn as nn

def get_model(args):
    weights_path = args.weights

    if args.model_arch=="resnet18":
        # model = resnet18()
        # model.fc = nn.Linear(512, args.num_classes)
        # model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False))
        model = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)

    elif args.model_arch=="deit":            
        model=timm.create_model('deit_small_patch16_224.fb_in1k', pretrained = False, num_classes = args.num_classes, checkpoint_path = weights_path)
    
    model.eval()
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    return model
