import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.data_loader import get_loader

class LRPModel(torch.nn.Module):
    def __init__(self, model, args):
        super(LRPModel, self).__init__()
        self.activations = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.args = args

        if args.model_arch=='resnet18':
            self.model.avgpool.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """Stores activations of the penultimate layer."""
        self.activations = output.clone()

    def forward(self, images , epsilon=0):
        
        output = self.model(images.to(self.device))
        output = torch.nn.functional.softmax(output, dim=1)
        
        if self.args.model_arch=="resnet18":
            
            activations = self.activations.view(images.shape[0], -1)
            weight_fc = self.model.fc.weight.to(self.device)
            bias = self.model.fc.bias.unsqueeze(dim=0)

        elif self.args.model_arch=="deit":
            activations = self.model.forward_features(images.to(self.device))[:,0]
            weight_fc = self.model.head.weight.to(self.device)
            bias = self.model.head.bias.unsqueeze(dim=0)
        
        out_rel = []
        out_feat = []
        activations = torch.cat((activations, torch.ones( (activations.shape[0],1)).to(self.device)), dim=1)
        weight_fc = torch.cat((weight_fc, bias.T), dim=1)
        for i in range(activations.shape[0]):
            z = activations[i].unsqueeze(dim=0) * weight_fc
            z_d= z.sum(dim=1, keepdim=True).T + epsilon 
            out = output[i].unsqueeze(dim=0) @(z/(z_d.T))
            out_feat.append(activations[i].detach().cpu().detach().numpy())
            out_rel.append(out.detach().cpu().numpy().squeeze(0))
        return out_feat, out_rel

   

def get_relevancy_score(args, model):    
    lrp_model = LRPModel(model, args)

    train_loader, id_loader, ood_loader = get_loader(args)
    loaders = [train_loader, id_loader, ood_loader]
    scores = []
    labels_all = []
    for loader in loaders:
        features = []
        relevancies = []
        labels = []
        with torch.no_grad():
            for batch_images, classes in loader:
                feature, relevancy = lrp_model.forward(batch_images)  # Get relevance scores
                features.extend(feature)
                relevancies.extend(relevancy)
                labels.extend(classes.detach().numpy().tolist())
        features_relevancy = np.concatenate((np.expand_dims(features, axis=0), np.expand_dims(relevancies, axis=0)), axis=0)
        
        scores.append(features_relevancy)
        labels_all.append(labels)
    return scores, labels_all