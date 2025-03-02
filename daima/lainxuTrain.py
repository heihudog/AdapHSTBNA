import csv
import time

import numpy as np
import torch

from model import AdapHBNA
from option import getargs
from trainNaowangluo import train


def ltrain(args1):
    allaccu = []
    bestacc = []
    all_result = []
    args = args1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # args.k_fold = 10
    # fold_list = args.k_fold
    for i in range(args.k_fold):
        i = int(i)
        fold = i

        # 在 for 循环中不断修改 layer 值

        device = torch.device(args.device)
        # train_loader, test_loader = abideread.load_abide(args.batch_size)  #直接使用划分

        # train_loader,test_loader = KfoldAbideRead.load_abide(fold,i)
        lr = args.lr
        model_name = f"model{fold}"
        model_name = AdapHBNA.HBGinTransformer(args,args.input_size, args.hidden_size, args.output_size,
                                                       args.num_heads,
                                                       args.norm_num, args.ratio, args.dropout,args.ratio1).to(args.device)
        # print(model_name.__dict__,"/n/n/n")
        # def reset_parameters(model):
        #     for layer in model.children():
        #         if isinstance(layer, nn.Linear):
        #             nn.init.xavier_uniform_(layer.weight)
        #             nn.init.zeros_(layer.bias)
        #
        #
        # model[model_name] =reset_parameters(model)
        init_para = model_name.state_dict()

        if not args.eval:
            acc_max, accu = train(args, fold, model_name, init_para)

        allaccu.append(accu)
        bestacc.append(acc_max)

    a = []
    array = np.array(allaccu)
    a.append(array[:, 0])
    bestacc1 = np.array(bestacc)
    a.append(np.mean(array[:, 0], axis=0))
    a.append(np.std(array[:, 0]))
    bestacc.append(np.mean(bestacc1, axis=0))
    # bestacc.append(np.std(bestacc1))

    # bestacc = np.array(bestacc)
    print('Accuracy summary:')
    print(a)
    print(bestacc)
    import csv  # 调用数据保存文件
    import pandas as pd  # 用于数据输出

    canshu = [getargs()]
    list = [time.localtime()]  # 列表头名称
    with open('test.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list)
        writer.writerows([canshu])
        writer.writerows([array])
        writer.writerows([a])


ratio = 0.01
for i in range(1, 10):
        # 修改 args.layer 值

    ratio =  0.01* 0.5**i
    args1 = getargs(lr  = ratio,A = 4)
    # with open('test.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(float(ratio))

    ltrain(args1)


ratio1 = 0.01
for i in range(1, 10):
        # 修改 args.layer 值

    ratio1 =  0.01* 0
    args2 = getargs(lr  = ratio1,A = 6)
    # with open('test.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(float(ratio))

    ltrain(args2)



for i in range(1, 10):
        # 修改 args.layer 值

    ratio =  0.0001
    args1 = getargs(lr  = ratio,A = 4)
    # with open('test.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(float(ratio))

    ltrain(args1)
# for i in range(8, 17):
#         # 修改 args.layer 值
#
#     ratio =  1e-4
#     args1 = getargs(ratio  = ratio,A = 2)
#     # with open('test.csv', 'a', newline='') as file:
#     #     writer = csv.writer(file)
#     #     writer.writerows(float(ratio))
#
#     ltrain(args1)


# for i in range(4, 17):
#     # 修改 args.layer 值
#
#     ratio = i * 0.05
#     args1 = getargs(ratio=ratio,A = 2)
#     # with open('test.csv', 'a', newline='') as file:
#     #     writer = csv.writer(file)
#     #     writer.writerows(float(ratio))
#
#     ltrain(args1)

