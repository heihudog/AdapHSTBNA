import torch
import torch.nn as nn
from option import getargs
import utils.backonemodel as backonemodel
args = getargs()

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.con = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.con)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.con(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels,  coff_embedding=4, num_subset=0,kernel_size=1):#num_subset=3
        super(unit_gcn, self).__init__()
        self.C = out_channels
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        for m in self.modules():
           if isinstance(m, nn.Conv2d):
               conv_init(m)
           elif isinstance(m, nn.BatchNorm2d):
               bn_init(m, 1)
        bn_init(self.bn, 1) #1e-6
        self.a = None
        epsilon = False
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0

    def forward(self, x):
        N, C1, T, V = x.size()
        C = self.C
        if(self.num_subset == 1):
            A2 = x.permute(0, 3, 1, 2).reshape(N * V, T, C)
            if (self.a == None):
                self.a = backonemodel.Attention(T, T, C, 1, 0.5).to(args.device)
                A = self.a(A2,A2)
                print("aaaaaaa",A.shape)
                x = (torch.matmul(A2, A) ).reshape(N, V, T, C).permute(0, 3, 2, 1)  #
        y = x

        y = self.bn(y)
        a = 0
        return self.relu(y),a

#L1 L2 L3 L4 L5  输入的图卷积层
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels,  stride = 1, residual=True,kernel=6,num_set=2, p = True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels,  num_subset=num_set,kernel_size=kernel)
        self.tcn1 = tcn(in_channels, out_channels, stride=stride,kernel_size=kernel)
        self.tcn2 = tcn(out_channels, out_channels, stride=stride,kernel_size=kernel)
        self.relu = nn.ReLU()

    def forward(self, x):
        _,C,_,_ = x.shape
        x = self.tcn1(x)
        x = self.tcn2(x)
        y,a = self.gcn1(x)
        return self.relu(y),a




config = getargs()
class SModel(nn.Module):
    def __init__(self, num_class=2, num_point=90, num_person=6,  in_channels=1,kernel=1,num_set=2,layer = 1,T = 32,T1 = 64):
        super(SModel, self).__init__()
        self.T = T
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l14 = TCN_GCN_unit(1, T,  stride=1, kernel=kernel, num_set=1)

    def forward(self, x):
        N, C, T, V, M = x.size()  #v是脑区个数，c是通道数，t是时间维度。  1  116  175  1
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  #（N,M,C,T,V）  (N*M,C,T,V)
        x, a = self.l14(x)
        x = x.mean(2)#结果就会变成（N*M,C,V）

        return x,a