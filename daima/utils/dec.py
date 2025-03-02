"""
From https://github.com/vlukiyanov/pt-dec
"""


from typing import Tuple
from utils.cluster import *
from utils.GIN import LayerGIN as GIN

class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True,spatial = False,norm_num = 116,assignments : Optional[torch.Tensor] = None
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder  #如果encoder使用的是transformer
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.spatial = spatial
        self.norm_num =  norm_num
        self.gin = GIN(hidden_dimension,hidden_dimension,hidden_dimension,norm_num)
        #self.assignment = ClusterAssignment(
        #    cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment
        #)#此处是聚类函数


        self.assignment = ClusterAssignment(
            cluster_number, hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center,
            project_assignment=project_assignment
        )  # 此处是聚类函数
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')#size_average=False




    def forward(self, batch: torch.Tensor,a = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        #print(batch,batch[0].shape,batch[1].shape)
        #batch = batch.transpose(1,2)
        if(self.spatial == False):
            node_num = batch.size(1)
            batch_size = batch.size(0)

            # [batch size, embedding dimension]
            flattened_batch = batch.reshape(batch_size, -1)  # 就是拉直
            encoded = self.encoder(flattened_batch)
            mae_loss = nn.MSELoss()
            lossm = mae_loss(flattened_batch, encoded)
            # [batch size * node_num, hidden dimension]
            #encoded = encoded.view(batch_size * node_num, -1)

            # [batch size * node_num, cluster_number]

            assignment, loss1,loss2 = self.assignment(encoded)
            # [batch size, node_num, cluster_number]

            assignment = assignment.view(batch_size, node_num, -1)





            # 计算重构损失

            lossp = self.loss(assignment)
            # [batch size, node_num, hidden dimension]
            encoded = encoded.view(batch_size, node_num, -1)
            # Multiply the encoded vectors by the cluster assignment to get the final node representations
            # [batch size, cluster_number, hidden dimension]

            node_repr = torch.bmm(assignment.transpose(1, 2), encoded)  #
            loss3 = lossp
            loss4 = self.lortho_loss(assignment.transpose(1, 2))
            #print("loss3,loss4", self.loss(self.softmax(assignment)), self.lortho_loss(assignment.transpose(1, 2)))
        # self.lortho_loss(assignment.transpose(1, 2)) +
        # node_repr = torch.bmm(encoded,assignment.transpose(1, 2))

        if(self.spatial == True ):
            node_num = batch.size(1)
            batch_size = batch.size(0)
            fea = batch.view(batch_size, node_num, -1)
            #fea = nn.functional.layer_norm(self.gin(fea, a), normalized_shape=fea.shape[-2:])
            flattened_batch = fea.reshape(batch_size, -1)  # 就是拉直
            encoded = self.encoder(flattened_batch)
            mae_loss = nn.MSELoss()
            lossm = mae_loss(flattened_batch, encoded)
            encoded = encoded.view(batch_size , node_num, -1)
            assignment, loss1,loss2 = self.assignment(encoded)
            lossp = self.loss(assignment)

            print("aaaaaaaaaaaaaaaaa",assignment.permute(0,2,1)[0][0][0])

            encoded = encoded.view(batch_size, node_num, -1)
            node_repr = torch.bmm(assignment.transpose(1, 2), encoded)  #


            loss3 =  lossp


        return node_repr, loss1,loss2,loss3,lossm
#（batch_size , cluster_num , node_num）


    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """

        weight = (batch ** 2) / (torch.sum(batch, 0)+0.000000001)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(batch, 0), torch.sum(weight, 1), "aaaa")
        return (weight.t() / (torch.sum(weight, 1)+0.000000001)).t()   #（-1，assignment.size(0)，assignment.size(-1)）


    def loss(self, assignment):

        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(torch.log(flattened_assignment+0.000000001), target)# / flattened_assignment.size(0) # 使用的时kl散度计算损失



    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tens or of dtype float
        """
        return self.assignment.get_cluster_centers()




    def kl_divergence(self, p, q):
        return torch.sum(p * torch.log(p / (q + 1e-10) + 1e-10), dim=-1)

    def max_kl_div_loss(self, inputs):
        batch, k, q = inputs.size()
        total_kl = 0

        for i in range(k):
            for j in range(i + 1, k):
                kl_ij = self.kl_divergence(inputs[:, i, :], inputs[:, j, :])
                kl_ji = self.kl_divergence(inputs[:, j, :], inputs[:, i, :])
                total_kl += (kl_ij + kl_ji) / 2

        loss = 1 / total_kl.mean()
        return loss

