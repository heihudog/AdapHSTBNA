"""
From https://github.com/vlukiyanov/pt-dec
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
from torch.nn.functional import softmax
from torch.autograd import Variable
import entmax
class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal=True,
        freeze_center=False,
        project_assignment=True
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        self.BatchNorm1d = nn.BatchNorm1d(self.embedding_dimension)
        self.softmax = nn.Softmax()
        self.kl = nn.KLDivLoss(reduction='batchmean')

        if cluster_centers is None:  #如果没有聚类中心的话，自己创建一个
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)

        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        initial_cluster_centers[j], initial_cluster_centers[i])
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                    torch.norm(initial_cluster_centers[i], p=2)

            initial_cluster_centers =orthogonal_cluster_centers

        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center))






    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        if self.project_assignment:  #这里就是true
            node_num = batch.size(1)
            batch_size = batch.size(0)
            assignment = batch@self.cluster_centers.T  #计算batch每个样本和样本中心的距离，计算的结果是个矩阵， 每行代表一个样本，每列是一个聚类中心，矩阵的元素表示到聚类中心的距离
            # prove
            assignment = torch.pow(assignment, 2)  #每个元素做平方，可能是类似于标准化的操作吧

            norm = torch.norm(self.cluster_centers, p=2, dim=-1) #聚类中心也标准化一下

            soft_assign = assignment/norm
            soft_assign = soft_assign.view(batch_size, node_num, -1)
            lossp = self.lortho_loss(entmax.entmax15(self.cluster_centers)) #可能会梯度爆炸
            lossd = self.max_kl_div_loss(self.cluster_centers)
           # print("cccccccccccccccccccccccccc",entmax.entmax15(soft_assign,dim = 2)[0][0][0],entmax.entmax15(soft_assign,dim = 1)[1][0][0])
            #loss =  lossd+lossp
            #print("loss2距离",lossd,"loss1正交",lossp)
           # print(print(self.max_kl_div_loss(self.cluster_centers.unsqueeze(0))))
          #  return softmax(soft_assign, dim=-1),lossp,lossd #按照到每个聚类中心的距离做一下标准化
            return entmax.entmax15(soft_assign,dim = 2), lossp, lossd
        else:

            norm_squared = torch.sum(
                (batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            lossp = torch.tensor(0.0, requires_grad=True)
            lossd =torch.tensor(0.0, requires_grad=True)
            return numerator / torch.sum(numerator, dim=1, keepdim=True),lossd,lossp

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers.

        :return: FloatTensor [number of clusters, embedding dimension]
        """
        return self.cluster_centers


    def loss(self, assignment):
        flattened_assignment = assignment.view(assignment.size(0),-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)#使用的时kl散度计算损失

    def lortho_loss(self,P):
        """
        计算一个 batch 的 Lortho 损失函数
        :param P: 输入矩阵 P，形状为 (batch_size, 64, 178)
        :return: batch 的 Lortho 损失值
        """
        k, _ = P.size()
        I_k = torch.eye(k).to(P.device)  # 创建 k x k 单位矩阵
        loss = 0


        P_i = P
        P_T_i = P_i.t()  # P_i 的转置178  64
        PP_T_i = torch.mm(P_i, P_T_i)  # P_i × P_T_i
        diff = (PP_T_i - I_k)@(PP_T_i - I_k)  # P_i × P_T_i - I_k
        loss = torch.norm(diff, p='fro')  # Frobenius 范数
        return loss/k


    def kl_divergence(self,p, q):
        return torch.sum(p * torch.log(p / (q+ 1e-10 )+ 1e-10), dim=-1)


    def max_kl_div_loss(self,inputs):
        k, q = inputs.size()
        total_kl = 1e-10
        epsilon = 1e-10
        loss = 0
        l = 0



        for j in range( 0, k-1):
            for i in range(j+1,k):
            #inputs = torch.clip(inputs, epsilon, 1.0)
            #kl_ij = self.kl(inputs[ 0], inputs[j])
            #kl_ji = nn.functional.cosine_similarity(inputs[ j], inputs[ 0],dim=0) #这是三角距离
                kl_ji = nn.functional.pairwise_distance(inputs[j],inputs[i])#使用欧式距离进行相似性度量
                total_kl += ( kl_ji)
                l = l+1

        #print(2 * total_kl / (k * (k - 1)))
        loss = (1/(total_kl))/l


        return loss
