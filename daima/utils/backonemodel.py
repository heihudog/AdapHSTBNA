import math
from entmax import entmax15
import torch
from torch import nn
from .util import  LayerNorm


vaild_lens = None
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
       # self.attention_weights = entmax15(scores)
        #att_output = entmax15(torch.bmm(scores,values))
        self.attention_weights = nn.functional.softmax(scores,dim= -1)
        att_output = torch.bmm(self.dropout(self.attention_weights),values)
        return  self.attention_weights,att_output
        #return torch.bmm(self.dropout(self.attention_weights), values)
class DotProductAttention1(nn.Module):
    """缩放点积注意力"""
    def forward(self, queries, keys):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores,dim= -1)
        return  self.attention_weights




def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(X.shape[0], X.shape[2], -1)


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class Attention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention1()
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, valid_lens = None):
        queries = self.W_q(queries.permute(0, 2, 1))
        keys = self.W_k(keys.permute(0, 2, 1))

        a = self.attention(queries.permute(0, 2, 1), keys.permute(0, 2, 1))
        #a = batch_adj(queries.permute(0, 2, 1))

        return a
# @save

#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens = None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        a,output = self.attention(queries, keys, values, valid_lens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        return a,output

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, dropout, normalized_shape,**kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        #print("XXXXX",X.shape,Y.shape)
        return self.ln(self.dropout(Y) + X)


#位置编码
#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

#这个是transformer层
class TransformerLayer(nn.Module):
    def __init__(self, key_size, query_size, value_size,hidden_size, num_heads, norm_num, dropout): #norm_num是第二个维度，这里可能是脑区也可能是时间序列维数
        super(TransformerLayer, self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.self_attention = MultiHeadAttention(key_size, query_size, value_size,hidden_size, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(norm_num)
        self.addnorml =  AddNorm(dropout,hidden_size)
    def forward(self, src,vaild_lens = None):
        a,src2= self.self_attention(src, src, src,vaild_lens)
        src2 = self.norm1(src2)
        #src3 = torch.matmul(src2,src)
        X = self.addnorml(src2, src)
        return X,a






class TransformerPositionLayer(nn.Module):
    def  __init__(self, key_size, query_size, value_size,hidden_size, num_heads, norm_num, dropout=0.4): #norm_num是第二个维度，这里可能是脑区也可能是时间序列维数
        super(TransformerPositionLayer, self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.self_attention = MultiHeadAttention(key_size, query_size, value_size,hidden_size, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(norm_num)
        self.addnorml =  AddNorm(dropout,hidden_size)
        self.position = PositionalEncoding(hidden_size,dropout)
        # self.ffn = PositionWiseFFN(
        #     ffn_num_input, ffn_num_hiddens, hidden_size)
    def forward(self, src,vaild_lens = None):
        src = self.position(src)
        a,src2= self.self_attention(src, src, src,vaild_lens)
        src2 = self.norm1(src2)
        #src3 = torch.matmul(src2,src)
        X = self.addnorml(src2, src)
        #X = self.addnorml(X, self.ffn(X))
        # src = self.norm1(X)
        # src = src + src0
        return X,a
class TransformerSTLayer(nn.Module):
    def  __init__(self, key_size, query_size, value_size,hidden_size, num_heads, norm_num, dropout=0.4): #norm_num是第二个维度，这里可能是脑区也可能是时间序列维数
        super(TransformerSTLayer, self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.self_attention = MultiHeadAttention(key_size, query_size, value_size,hidden_size, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(norm_num)
        self.addnorml =  AddNorm(dropout,hidden_size)
        self.position = PositionalEncoding(hidden_size,dropout)
    def forward(self, k,q,v,vaild_lens = None):
        #src = self.position(x)
        a,src2= self.self_attention(k, q, v,vaild_lens)
        src2 = self.norm1(src2)
        X = self.addnorml(src2, v)
        return X,a


#################################################PC attention
def f_correltaion(x,y):
    batch_size = x.shape[0]
    n = x.shape[1]
    features = x.shape[2]
    correlation_list = []
    # 逐个计算每个人的皮尔逊相关系数
    for i in range(batch_size):
        x_i = x[i, :, :]  # 获取第 i 个人的数据
        y_i = x[i,:,:]
        x_i_reshaped = x_i.squeeze()  # 重新排列成 (n, features) 的形状
        y_i_reshaped = y_i.squeeze()
        correlation_matrix_i = pearson_correlation(x_i_reshaped,y_i_reshaped)
        correlation_list.append(correlation_matrix_i)
    # 将列表转换为张量
    correlation_tensor = torch.stack(correlation_list)
    return correlation_tensor
def pearson_correlation(tensor,tensor1):
    # 将 tensor 转换为浮点型
    tensor = tensor.float()
    tensor1 = tensor1.float()
    # 计算每个节点的均值
    mean = torch.mean(tensor, dim=1, keepdim=True)
    mean1 = torch.mean(tensor1,dim=1 , keepdim= True)
    # 减去均值进行中心化
    tensor_centered = tensor - mean  # 116  175
    tensor_centered1 = tensor1 - mean1 # 116  175

    # # 7. 计算皮尔逊相关系数

    # 计算每个节点的标准差
    std = torch.std(tensor, dim=1, keepdim=True)#dim = 0 沿着列作相关系数    dim = 1 沿着行作相关系数
    std1 = torch.std(tensor1,dim=1,keepdim=True)
    # 计算标准化后的矩阵
    tensor_normalized = tensor_centered / std
    tensor_normalized1 = tensor_centered1/std1
    # 计算节点之间的相关系数矩阵
    correlation_matrix = torch.mm(tensor_normalized, tensor_normalized1.t()) / tensor.size(1)
    return correlation_matrix

class PCattention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout,norm_num, bias=False, **kwargs):
        super(PCattention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.norm_num = norm_num
        self.dp = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.ln = LayerNorm(norm_num)

    def forward(self, queries, keys ):
        q = self.dp(self.W_q(queries))
        k = self.dp(self.W_k(keys))
        a =self.ln(f_correltaion(queries, keys))
        a1 = self.ln(f_correltaion(q,k))
        return self.ln(a + a1)/2


class PCnetwork(nn.Module):
    def __init__(self, key_size, query_size, value_size,hidden_size, num_heads, norm_num, dropout):
        super(PCnetwork,self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(norm_num)
        self.addnorml = AddNorm(dropout,hidden_size)
        self.PCattention = PCattention(key_size, query_size, value_size,hidden_size, num_heads,dropout,norm_num )
    def forward(self, src, vaild_lens=None):
        a = self.PCattention(src, src)
        return  a





#输入脑网络信息
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()


    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)


    def forward(self, input, percentiles):#此处是进行稀疏化的地方
        input = torch.flatten(input) # find percentiles for flattened axis#把input拉成向量
        input_dtype = input.dtype
        input_shape = input.shape#把input的形式保存下来
        if isinstance(percentiles, int):
            percentiles = (percentiles,) #转换成列表形式的张量 也就是两维度
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)#把输入的形状更新一下，类似于拉直的操作
        in_sorted, in_argsort = torch.sort(input, dim=0) #进行排序，选返回张量和排序的索引
        positions = percentiles * (input.shape[0]-1) / 100#找到
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)


    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

def construct_adjacent_matrix(pc, sparsity):
    p = Percentile()
    thresholded = (pc > p(pc, 100 - sparsity))
    _i = thresholded.nonzero(as_tuple=False)
    _v = torch.ones(len(_i))
    _i = _i.T
    return torch.sparse.FloatTensor(_i, _v, (pc.shape[0], pc.shape[1]))





def batch_adj(X):
    result = []
    for x in X:
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        result.append(c)
    return  torch.stack(result, dim=0)
