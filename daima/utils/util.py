
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
import numpy as np
from option import getargs
args = getargs()
class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.ones(feature_size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1,keepdim=True)
        return  self.a_2 * (x - mean) / (std +self.eps) + self.b_2





#精度获取
# metrics
def get_acc(pred, label):
    return accuracy_score(label, pred)

def get_sen(pred, label):
    return recall_score(label, pred,average='macro')

def get_spe(pred, label):
    if(np.sum(label)>len(label)):
        return 0
    else:
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        return tn / (tn + fp)


def get_pre(pred, label):
    return precision_score(label, pred,average='macro')

def get_f1(pred, label):
    return f1_score(label, pred,average='macro')

def get_auc(prob, label):
    #print(prob.shape)
    prob = torch.tensor(prob)
    score = prob[:, 1]
    if (np.sum(label) > len(label)):
        score = prob.reshape(-1, args.output_size)
        return roc_auc_score(label, score, multi_class='ovr')
    else:
        return roc_auc_score(label, score)


def get_bac(pred, label):
    return balanced_accuracy_score(label, pred)

def evaluate(pred, prob, label):
    acc = get_acc(pred, label)
    sen = get_sen(pred, label)
    spe = get_spe(pred, label)
    pre = get_pre(pred, label)
    f1 = get_f1(pred, label)
    auc = get_auc(prob, label)
    bac = get_bac(pred, label)
    return acc, sen, spe, pre, f1, auc, bac
def upper_triangular_vectors(A):
    batch_size, w, h = A.shape
    triu_indices = torch.triu_indices(w, h,1)
    # 用上三角索引提取每个矩阵的上三角元素，并拉成向量
    return A[:, triu_indices[0], triu_indices[1]]

