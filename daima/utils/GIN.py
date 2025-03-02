import torch
from torch import nn
from einops import rearrange
from option import getargs
args = getargs()
class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm_num = 116,epsilon=False ):
        super().__init__()
        self.norm_num = norm_num
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.conv1d = nn.Conv1d(input_dim,input_dim,kernel_size=115)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(norm_num), nn.LeakyReLU())


    def forward(self, v, a):
        v_aggregate = torch.matmul(a, v) #向量相乘
        b, n = v_aggregate.shape[0], v_aggregate.shape[1]  # 取出长和宽
        v_aggregate += self.epsilon * v # assumes that the adjacency matr
        v_aggregate = rearrange(v_aggregate, 'b n c -> (b n) c')

        v_aggregate = rearrange(v_aggregate, '(b n) c -> b n c', b=b, n=n)
        v_combine = self.mlp(v_aggregate)

        return v_combine




class LayerGIN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        b, n = v.shape[0], v.shape[1]
        v_aggregate = torch.matmul(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_aggregate = rearrange(v_aggregate, 'b n c -> (b n) c')
        v_combine = self.mlp(v_aggregate)
        v_combine = rearrange(v_combine, '(b n) c -> b n c', b=b, n=n)
        return v_combine



class LayerGIN_N(nn.Module):
    def __init__(self, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0

    def forward(self, v, a):
        b, n = v.shape[0], v.shape[1]
        v_aggregate = torch.matmul(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        return v_aggregate

