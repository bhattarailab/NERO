import csv
import os
import torch
import random
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from utils.args_loader import get_args
from utils.model_loader import get_model
from utils.args_loader import get_args
from utils.feature_relevancy_extract import *

def initialize_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pca_projection(train_features, id_feature, ood_feature, n_components):
    pca = PCA(n_components=n_components)
    feature_matrix_reduced = pca.fit_transform(train_features)  # Shape: (N, n_components)
    reduced_features = feature_matrix_reduced
    id_features_reduced = pca.transform(id_feature)
    ood_features_reduced =pca.transform(ood_feature)
    return reduced_features, id_features_reduced, ood_features_reduced

def rel_dist(train, valid, ood, labels):
    
    rel_class = []
    for cls in np.unique(labels):
        rel_class.append(np.expand_dims(train[cls == labels].mean(axis=0), axis=0)) # number of filters, 1
    scores = []
    for data in [valid, ood]:
        rels_diff = []
        for class_rels in rel_class:
            diff = np.linalg.norm(data-class_rels, ord=1, axis=1)
            rels_diff.append(diff) #3, 600    
        scores.append(np.min(np.asarray(rels_diff), axis=0))
    
    return scores



def relevanacy_algo(args, train, valid, ood, train_labels):
    
    k, n_component = 175, np.int32(2*train[1][:, :-1].shape[-1]/3)
    k_smallest_indices_id = np.argsort(valid[1][:, :-1], axis=1)[:, :k]
    samples_idx_id = np.arange(valid[1][:, :-1].shape[0])[:, None]
    k_features_id = valid[0][:, :-1][samples_idx_id, k_smallest_indices_id]    
        
    k_smallest_indices_ood = np.argsort(ood[1][:, :-1], axis=1)[:, :k]
    samples_idx_ood = np.arange(ood[1][:, :-1].shape[0])[:, None]
    k_features_ood = ood[0][:, :-1][samples_idx_ood, k_smallest_indices_ood]       
        
    data = [train[1][:, :-1], valid[1][:, :-1], ood[1][:, :-1]]
    data= pca_projection(*data, n_component)
        
    id_features = np.linalg.norm(k_features_id, axis=1, ord=1)/np.linalg.norm(valid[0][:,:-1], axis=1, ord=1)
    ood_features = np.linalg.norm(k_features_ood, axis=1, ord=1)/np.linalg.norm(ood[0][:,:-1], axis=1, ord=1)

    scaling_train = rel_dist(train[1], train[1], train[1],train_labels)
    scaling_term_bias = scaling_train[0].mean()/np.abs(train[1][:,-1]).mean()
        
    relevancy_diff = rel_dist(*data, train_labels)

    scores_in = (relevancy_diff[0] + scaling_term_bias*np.abs(valid[1][:,-1])) * id_features
    scores_ood = (relevancy_diff[1] + scaling_term_bias*np.abs(ood[1][:,-1]))  * ood_features

    scores_in, scores_ood = np.asarray(scores_in), np.asarray(scores_ood)
    
    DATA = [0 for _ in range(len(scores_in))] + [
            1 for _ in range(len(scores_ood))
        ]
    aucs = roc_auc_score(DATA, scores_in.tolist() + scores_ood.tolist())
    num_ind = len(scores_in)
    recall_num = int(np.floor(0.95 * num_ind))
    thresh = np.sort(scores_in)[recall_num]
    fpr = np.sum(scores_ood <= thresh)/len(scores_ood)
    with open(args.base_dir, 'a') as file:
        file.write(f" AUC = {round(aucs.mean()*100, 2):.2f} FPR@95TPR = {round(fpr.mean()*100, 2):.2f}\n")


if __name__=="__main__":
    
    args = get_args()
    initialize_seed(args.seed)
    model = get_model(args)
    feat_rel, labels = get_relevancy_score(args, model)
    train, valid, ood= feat_rel[0],  feat_rel[1],  feat_rel[2]
    labels = np.asarray(labels[0])
    relevanacy_algo(args, train, valid, ood, labels)
