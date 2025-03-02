import torch
from utils import GIN,backonemodel,dec
from torch import nn
from utils.pool import  DiffPool,Percentile  #稀疏化
from utils import CTA,mambablock
from option import getargs

args = getargs()
j = 0
class transpose_dimensions(nn.Module):
    def __init__(self,transpose = True):
        self.transpose = transpose

    def forward(self,input_tensor):
        output_tensor = input_tensor.transpose(0, 2, 1)
        return output_tensor


class HBGinTransformer(nn.Module):
    def __init__(self,args,input_size,hidden_size,output_size,num_heads,norm_num,ratio,dropout,ratio1 = 0.4):
        super(HBGinTransformer,self).__init__()
        self.input_size = input_size
        self.hidden_size=  hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.norm_num = norm_num
        self.ratio =ratio
        self.ratio1 = ratio1
        self.softmax = nn.Softmax(dim=-1)
        self.dp1 = nn.Dropout(p=dropout)
        args = args
        print(ratio)

        self.mamba_blocks = nn.ModuleDict()
        mamba_args = {}
        self.attention = nn.ModuleDict()
        self.TGL = nn.ModuleDict()
        self.Sencoder = nn.ModuleDict()
        self.Sdec = nn.ModuleDict()
        new_n = args.norm_num
        T = args.input_size
        #T_out = int(T/(args.layer+1))
        T_out = 88#int(T *ratio1)
        ii = 0
        for i in range(args.layer):  #64  64
            new_n1 = new_n
            T_1 =T
            if(ii==0):
                new_n = int(args.ratio * new_n)
                ii = -1
            else:
                new_n = int( args.ratio * new_n)

            T = int(T - (args.input_size - T_out) / (args.layer))
            if(i==(args.layer - 1)):   # 16 90 64
                mamba_args[f'{i}'] = mambablock.ModelArgs(d_model=new_n1, n_layer=i, features=T_out)
                self.mamba_blocks[f'mamba{i}'] = mambablock.MambaBlock(mamba_args[f'{i}'])

                #self.attention[f'{i}'] = backonemodel.PCnetwork(T_1, T_1, T_1, T_1,
                                                              #  num_heads, new_n1, dropout)#.to(args.device)

                self.TGL[f'{i}'] = CTA.SModel(num_point=new_n1, num_person=1, kernel=7, layer=i,T = T_out,T1 = T_1)


                self.Sencoder[f'{i}'] = nn.Sequential(
                    nn.Linear(T_out * new_n1, new_n1 * 16),  # 3904
                    nn.ReLU(),
                    nn.Linear(new_n1 * 16, new_n1 * 16),
                    nn.ReLU(),
                    nn.Linear(new_n1 * 16, new_n1 * T_out),
                )
                self.Sdec[f'{i}'] = dec.DEC(new_n, hidden_dimension=T_out, encoder=self.Sencoder[f'{i}'],
                                            orthogonal=True, freeze_center=False, project_assignment=True, spatial=True,norm_num = new_n1)

            else:

                mamba_args[f'{i}'] = mambablock.ModelArgs(d_model=new_n1, n_layer=1, features=T_out)#args.input_size)
                self.mamba_blocks[f'mamba{i}'] = mambablock.MambaBlock(mamba_args[f'{i}'])

               # self.attention[f'{i}'] = backonemodel.PCnetwork(T_1, T_1, T_1, T_1,
                                                              #  num_heads, new_n1, dropout)#.to(args.device)

                self.TGL[f'{i}'] = CTA.SModel(num_point=new_n1, num_person=1, kernel=7, layer=i,T=T,T1 = T_1)

                self.Sencoder[f'{i}'] = nn.Sequential(
                    nn.Linear(T * new_n1, new_n1  * 16),  # 3904
                    nn.GELU(),
                    nn.Linear(new_n1 * 16, new_n1 * 16),
                    nn.GELU(),
                    nn.Linear(new_n1 * 16, new_n1 * T),
                )
                self.Sdec[f'{i}'] = dec.DEC(new_n, hidden_dimension=T, encoder=self.Sencoder[f'{i}'],
                                            orthogonal=True, freeze_center=False, project_assignment=True, spatial=True,norm_num = new_n1)




        self.dim_reduction1 =nn.Sequential(
                nn.Linear(T_out, int(T_out/2)),
                #nn.BatchNorm1d(int(T_out/2)),
                nn.ReLU()
            )


        self.ln300t2 = nn.Sequential(
            nn.Linear((int(T_out/2)*int(new_n )),30), #int((args.cluster*(args.cluster+1))/2-args.cluster) +6*args.cluster
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 16),
            nn.BatchNorm1d(16),
            #nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, args.output_size),
            nn.Softmax(dim = -1)
            )



    def HB(self,inputs,i):


        Z = self.mamba_blocks[f'mamba{i}'](inputs)
        a = backonemodel.batch_adj(Z.permute(0, 2, 1))
        a[torch.isnan(a)] = 0.0
        a = Percentile.batch_adjacent_matrix(a, 10).to_dense() #此处修改过
        if(i == 0):
            print(a.shape)
            torch.save(a,"a.pt")
        Z = Z.unsqueeze(1).unsqueeze(4)  # batch   1 175 64 1
        x, a1 = self.TGL[f'{i}'](Z)  # 64,116 捕获空间信息   116   64
        x = x.permute(0, 2, 1)  # 64 64
        x = nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],))
        fea, loss1, loss2,loss3,loss4 = self.Sdec[f'{i}'](x, a)
        fea = fea.permute(0, 2, 1)
        loss = loss3 + loss1 + loss4+ loss2
        return fea ,loss

    def forward(self,inputs,order = "11"):
        if(order == '11'): #0.71
            args = getargs()
            inputs = nn.functional.layer_norm(inputs, normalized_shape=inputs.shape[-2:])
            losss =  {}
            for i in range(args.layer):
                print(i)
                inputs, loss1 = self.HB(inputs,i)
                losss[f'{i}'] = (1/args.layer)*loss1
            fea = inputs.permute(0,2,1)
            bz, _, _, = fea.shape
            node_feature = self.dim_reduction1(fea)
            fea_out1 = node_feature.reshape((bz, -1))
            outputs = self.ln300t2(fea_out1)
            loss = (1/args.layer)*sum(losss.values())#+0.1*loss2  #  (1/args.layer)*
            return outputs, loss




