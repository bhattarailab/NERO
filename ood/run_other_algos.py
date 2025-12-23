import torch
import os
from utils.data_loader import get_loader
from utils.args_loader import get_args
from utils.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
import cv_uncertainty as unc
import random

def initialize_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = get_args()

initialize_seed(args.seed)

START = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model=get_model(args)
model.to(device)

batch_size = args.batch_size

UNC_METHODS = [
    unc.MSP(),
    unc.Odin(), 
    unc.EnergyQuant(),
    unc.EntropyQuant(),
    unc.MaxLogit(),
   
    unc.Mahalanobis(),
    unc.Neco(), 
    unc.ViM(),
    unc.ReAct_Energy(),
    unc.GradNorm()
]

model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, id_loader, ood_loader = get_loader(args)

for uncertainty in UNC_METHODS:
    print(f"\t\t\t{uncertainty.name} - {time.time()-START}s")
    
    if uncertainty.name in ['neco', 'vim', 'mahalanobis']:
        scores_in, scores_ood = uncertainty.quantify(
                                        model=model,
                                        train_loader = train_loader,
                                        val_loader = id_loader,
                                        ood_loader = ood_loader,
                                        model_name = args.model_arch,
                                        )
    elif uncertainty.name == 'fdbd':
        scores_in, scores_ood = uncertainty.quantify(
                                        model=model,
                                        train_loader = train_loader,
                                        val_loader = id_loader,
                                        ood_loader = ood_loader,
                                        model_name = args.model_arch,
                                        id_dataset = args.in_dataset
                                        )
    
    
    else:

        scores_in = uncertainty.quantify(
            data_loader=id_loader,
            model=model,
            model_name = args.model_arch,
            num_class = args.num_classes
        )
        scores_ood = uncertainty.quantify(
            data_loader=ood_loader,
            model=model,
            model_name = args.model_arch,
            num_class = args.num_classes
        )
    
    
    DATA = [0 for _ in range(len(scores_in))] + [
        1 for _ in range(len(scores_ood))
    ]
    aucs = roc_auc_score(DATA, scores_in.tolist() + scores_ood.tolist())
    num_ind = len(scores_in)
    recall_num = int(np.floor(0.95 * num_ind))
    thresh = np.sort(scores_in)[recall_num]
    fpr = np.sum(scores_ood <= thresh)/len(scores_ood)
    
    with open(args.base_dir, 'a') as file:
        file.write(f"{uncertainty.name}: AUC = {round(aucs.mean()*100, 2):.2f} FPR@95TPR = {round(fpr.mean()*100, 2):.2f}\n")    