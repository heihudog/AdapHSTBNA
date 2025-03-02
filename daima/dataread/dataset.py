import scipy
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import tensor, float32
from random import shuffle
import numpy as np
from torch.utils.data import Dataset
from utils import util
from option import getargs
args = getargs()
def dataread(train = True):
    data_path = args.data_path
    mat_data = scipy.io.loadmat(data_path)

    data = []
    if args.dataset == 3:
        raw_data = np.array(mat_data['timeseries'])
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        data=np.transpose(data,(0,2,1))
        labels = np.array(mat_data['label']).reshape(-1)
        print(labels.shape)
        #zx1=mat_data["ecg_all"]
        labels[labels == 1] = 0
        labels[labels == 4] = 1
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        print(labels.shape)

    elif args.dataset == 4  :
        raw_data = np.array(mat_data['AAL2'])
        #print(raw_data)
        data = raw_data[0]
        # for item in raw_data:
        #     data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab']).squeeze()
        labels[labels == -1] = 0
    elif args.dataset == 5:
        raw_data = np.array(mat_data['AAL'])
        data = raw_data[0]
# for item in raw_data:
#     data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab']).squeeze()
        labels[labels == -1] = 0

    else:
        raw_data = np.array(mat_data['AAL'])[0]
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)

        labels = np.array(mat_data['lab'][0])
        print(labels.shape)



    train_data,test_data,train_label,test_label = train_test_split(data, labels, test_size=0.15, random_state=42,stratify=labels)  #42
    if(train == True):
        return train_data, train_label
    if(train == False):
        return {  tensor(test_data, dtype=float32),  tensor(test_label)}


class Load_Data(Dataset):

    def __init__(self, k_fold=None):
        # data_path: path to the sample file, label_path: path to the label file
        # k_fold: the number of folds, 'None' indicates no folding
        # data should be a shape of (batch,_,_)
        self.data_dict = {}

        self.data,self.label = dataread()
        #self.label = labels # labels should consist of 0 and 1

        for id in range(self.data.shape[0]):
            self.data_dict[id] = self.data[id, :, :]

        self.full_subject_list = list(self.data_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
        self.k = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    # set folds
    def set_fold(self, fold, train=True):
        # fold: the fold to use
        # train: 'True' indicates using the training set, 'False' indicates using the validation set.
        assert self.k_fold is not None
        self.k = fold
        train_idx,test_idx = list(self.k_fold.split(self.full_subject_list, self.label))[fold]

        if train: shuffle(train_idx)
        if not train:shuffle(test_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        X = self.data_dict[subject]
        y = self.label[subject]
        #y   = np.eye(2)[y]

        return {'id': subject, 'X': tensor(X, dtype=float32), 'y': y}


