import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import getargs
import utils.backonemodel as backonemodel
from utils.pool import  Percentile


args = getargs()




def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


#L1 L2 L3 L4 L5  输入的图卷积层
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, residual=True,kernel=6,num_set=2):
        super(TCN_GCN_unit, self).__init__()
        self.tcn1 = unit_tcn(in_channels, out_channels, stride=stride,kernel_size=kernel)
        self.relu = nn.ReLU()
        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=kernel, stride=stride)

    def forward(self, x):
        x = self.tcn1(x) #+ self.residual(x) #(N*M,C,T,V)
        return self.relu(x)

class Model(nn.Module):
    def __init__(self, num_class=2, num_point=90, num_person=6,  in_channels=1,kernel=1,num_set=2):
        super(Model, self).__init__()
        # A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(1, 12,  residual=False,kernel=kernel,num_set=num_set)
        self.l2 = TCN_GCN_unit(12, 12, kernel=kernel,num_set=num_set)
        self.l4 = TCN_GCN_unit(12, 12, kernel=kernel,num_set=num_set)
        self.l5 = TCN_GCN_unit(12, 32,  stride=2,kernel=kernel,num_set=num_set)
        self.l6 = TCN_GCN_unit(32, 32, kernel=kernel,num_set=num_set)
        #self.l8 = TCN_GCN_unit(32, 64,  stride=2,kernel=kernel,num_set=num_set)


        self.fc = nn.Linear(64, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()  #v是脑区个数，c是通道数，t是时间维度。  1  116  175  1
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  #（N,M,C,T,V）  (N*M,C,T,V)
        x= self.l1(x)
        x = self.l2(x)
        x = self.l4(x)
        x = self.l5(x)
        x= self.l6(x)
        #x = self.l8(x)
        x = x.mean(1)#结果就会变成（N*M,C,V）
        return x


