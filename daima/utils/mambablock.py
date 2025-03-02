import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from einops import rearrange, repeat, einsum


from option import getargs
config = getargs()


class ModelArgs:
    def __init__(self, d_model: int, n_layer: int,features:int):
        self.d_model = d_model
        self.n_layer = n_layer
        self.features = features
        self.__post_init__()
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    K: int = 3
    A: torch.Tensor = None
    feature_size: int = None
    d_inner: int
    dt_rank : int
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)




class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        #self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size)
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x0 = x
        x1 = self.norm(x)
        #x2 = self.kfgn(x1)
        x3 = self.mixer(x1)
        output = x3

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
       # self.kfgn = kfgn


        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias).to(config.device)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        ).to(config.device)

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False).to(config.device)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True).to(config.device)

        A = repeat(torch.arange(1, args.d_state + 1).to(config.device), 'n -> d n', d=args.d_inner)#这里的A是自定义的，和x没什么关系，有的改进点是让A = linear(x)

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner)).to(config.device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias).to(config.device)


    def forward(self, x):
        (b, l, d) = x.shape  # 输入是三个维度 l 和d
        x_and_res = self.in_proj(x)  #先映射，为了映射出两个维度
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner],#d_inner = 2 n
                                   dim=-1)  # 切分成不同的向量，把最后一个维度切成两个self.args.d_inner

        x = rearrange(x, 'b l d_in -> b d_in l')  # b  l self.args.d_inner
        x = self.conv1d(x)[:, :, :l] # 得到是 b din l  取前l个
        x = rearrange(x, 'b d_in l -> b l d_in')


        x = F.silu(x)  # 激活函数

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output


    def ssm(self, x):
        # 输入的x是三维度 (b, l, d)

        (d_in, n) = self.A_log.shape  # A是一个由参数组成的矩阵，状态矩阵

        A = -torch.exp(self.A_log.float())#(d_in, n)
        D = self.D.float()  # D朱焕成float形式

        x_dbl = self.x_proj(x)  # 把x映射成args.dt_rank + args.d_state * 2
        #为什么这样分 ，为了从x初始化 delta B C


        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  #B  和 D 都是取自于x的映射
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)  softplus是一个激活函数，具体形式为 ln(1+ex)

        # 这里的delta是从x中分离的 前dt rank
        y = self.selective_scan(x, delta, A, B, C, D)  # 输入(b, l, d)   (b, l,dt_rank)

        return y



    def selective_scan(self, u, delta, A, B, C, D):

        (b, l, d_in) = u.shape#这里面 u是x ，delta是x映射后 分出来的一部分

        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        # In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        temp_adj_padded = torch.ones(d_in, d_in, device=config.device)#这里可以再改一改
        delta_p = torch.matmul(delta, temp_adj_padded)

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))  # 知道这个是爱因斯坦求和，这里大家理解为矩阵相乘就可以了
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n') #这里和原文有所不同

        x = torch.zeros((b, d_in, n), device=config.device)  #这里的x看作是隐藏层
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]   #i 代表第i个时间点，这里面少了x，为什么，因为已经在deltaB——u里乘完了
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in') #
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output