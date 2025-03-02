import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
from dataread import dataset
from model import AdapHBNA  as  HBGinTNaowangluo
import torch
import time
import sklearn.metrics as metrics
import torch.nn as nn
from tqdm import tqdm
from option import getargs
from utils import util
import csv
def train(args,fold,model,init):

    args = getargs()
    device = args.device
    min_lr =args.lr# 6e-5#1e-6+ 5e-7#args.lr#3e-7  0.00025
    max_lr = args.lr*1.2
    current_val = 1
    acc_out = []
    losses = []
    testloss=  []
    test_out = []
    tlosses = []
    val_accs = []
    dataset1 = dataset.Load_Data(k_fold=args.k_fold)
    labels = torch.tensor(dataset1.label).to(torch.long)
    print(labels)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)

    #dataloader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler)
    dataloader = torch.utils.data.DataLoader(dataset1, batch_size=args.minibatch_size, shuffle=False)
    model = model
    lr = args.lr
    model.load_state_dict(init)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)  # 不设置l2损失
#    scheduler = StepLR(opt, 10, gamma=0.1)

    def adjust_learning_rate(train_accuracy, val_accuracy, optimizer, current_val,current_train,mean_prediction):
        for param_group in optimizer.param_groups:
            print("LR",param_group['lr'])
            current_lr = param_group['lr']
            if train_accuracy > val_accuracy:

                new_lr = 0.8*current_lr
                if mean_prediction.item()  == 1:
                    new_lr = new_lr * 0.8
                if mean_prediction.item()  == 0:
                    new_lr = new_lr * 0.8
            elif train_accuracy < val_accuracy:
                new_lr = current_lr * 1.1

            else:
                new_lr = current_lr
            param_group['lr'] = max(min(new_lr, max_lr), min_lr)


    loss_entro = nn.CrossEntropyLoss()

    best_test_acc = 0
    b = 0
    c = 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        tloss = 0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        val_pred = []
        val_true = []
        Prob_v = []
        val_loss = 0
        idx = 0
        total_time = 0.0
        dataset1.set_fold(fold=fold, train=True)

        for _,data in enumerate(tqdm(dataloader, ncols=80, desc=f'fold:{fold} | epoch:{epoch}')):
            print(f"Epoch {epoch + 1}, Learning rate: {opt.param_groups[0]['lr']}")
            data, label = data['X'].to(args.device), data['y'].to(args.device).squeeze()
            print(data.shape)
            batch_size = data.size()[0]
            c += 1
            b +=  label.sum()
            start_time = time.time()
            logits,loss1 = model(data)
            label = label.to(torch.float32)

            loss =  loss_entro(logits, label.long()).to(device) +loss1.to(device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            value11, preds = torch.max(logits.data, 1)

            count += batch_size
            train_loss += loss.cpu().detach().numpy().item() * batch_size
            tloss = loss_entro(logits, label.long()).cpu().detach().numpy().item()
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1

        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (
        epoch, train_loss * 1.0 / count, train_acc)

        print("outstroutstroutstroutstroutstroutstr",outstr)


        losses.append((train_loss * 1.0 / count))
        tlosses.append((tloss))

        #####################
        #val
        dataset1.set_fold(fold=fold, train=False)
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader, ncols=80, desc=f'fold:{fold} | epoch:{epoch}')):
                data, label = data['X'].to(args.device), data['y'].to(args.device).squeeze()
                batch_size = data.size()[0]
                start_time = time.time()

                logits, loss1 = model(data)
                logits = logits.squeeze(1)
                label = label.to(torch.float32)
                end_time = time.time()
                total_time += (end_time - start_time)
                value11, preds = torch.max(logits.data, 1)

                count += batch_size
                val_loss += loss.cpu().detach().numpy().item() * batch_size
                val_true.append(label.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())
                Prob_v1 = logits.softmax(1).tolist()
                Prob_v.extend(Prob_v1)
                idx += 1

                #

            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)
            val_acc = metrics.accuracy_score(val_true, val_pred)

            val_accs.append(val_acc)
            evaluation_metrics_v = util.evaluate(pred=val_pred, prob=Prob_v, label=val_true)
            print(val_accs)
            print("labellabellabellabellabel",label)
            print("predspredspredspredspreds",preds)
            mean_prediction =torch.mean(preds.float())
            outstr = 'Val %d,  val acc: %.6f' % (
                epoch,val_acc)

            if(epoch == 0):
                current_val = 0
                current_train = 0

            print(outstr,  current_val)
            adjust_learning_rate(train_acc, val_acc, opt,current_val,current_train,mean_prediction)
            current_val = val_acc
            current_train = train_acc
            if val_acc >= best_test_acc:
                        best_test_acc = val_acc
                        best_model = model.state_dict()

        print("best_test_accbest_test_accbest_test_acc",best_test_acc)





    # Test
    ####################
    with (torch.no_grad()):
        test_loss = 0.0
        count = 0.0
        model.load_state_dict(best_model)
        t = time.strftime('%Y-%m-%d%H-%M-%S', time.localtime())
        filename = os.path.basename(args.data_path)
        PATH = './save/model/  '+ t+'%s %d %d' % (filename,best_test_acc,fold) + '.pth'
        torch.save(best_model, PATH)
        model.eval()
        total_time = 0.0
        data, label = dataset.dataread(train = False)
        data = data.to(device)
        label = label.to(device)
        print(label.shape)

        batch_size = data.size()[0]
        start_time = time.time()
        # logits, lossd, lossp = model(data)
        logits, loss1 = model(data)
        logits = logits.squeeze(1)
        label = label.to(torch.float32)
        value22, preds = torch.max(logits.data, 1)
        print(logits)
        print("labellabellabellabel",label)
        print("predspredspredspredspreds",preds)
        # print('test 第',fold,'次',label)
        Prob = logits.softmax(1).tolist()
        end_time = time.time()
        total_time += (end_time - start_time)
        count += batch_size
        test_loss1 = loss_entro(logits, label.long()).detach()
        test_loss += test_loss1.item() * batch_size

        label = label.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()


        print('test total time is', total_time)

        test_acc = metrics.accuracy_score(label, preds)
        evaluation_metrics = util.evaluate(pred=preds, prob=Prob, label=label)
        outstr = 'Test  loss: %.6f,  test acc: %.6f' % ( test_loss * 1.0 / count, test_acc)

        test_out.append(test_acc)
        acc_out.append(evaluation_metrics)
        testloss.append((test_loss * 1.0 / count))
        print(outstr)


    plt.figure(1)

    y1=  losses
    plt.plot(np.arange(0, len(losses)),y1,'r')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if (fold == args.k_fold - 1):
        t = time.strftime('%Y-%m-%d%H-%M-%S', time.localtime())
        plt.savefig('./save/trainpicture/' + t + '1.png')
    plt.figure(2)

    plt.plot(np.arange(0, args.epochs), val_accs)
    plt.title('acc Function')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    if(fold  == args.k_fold-1):
        t = time.strftime('%Y-%m-%d%H-%M-%S', time.localtime())
        plt.savefig('./save/trainpicture/' + t + '2.png')

    if(fold == args.k_fold-1):
        plt.show()
    else:
        plt.show(block=False)
    return evaluation_metrics_v,evaluation_metrics



if __name__ == "__main__":
    allaccu = []
    bestacc = []
    all_result = []
    args = getargs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for i in range(args.k_fold  ):
        i = int(i)
        fold = i
        args = getargs()
        device = torch.device(args.device)
        lr = args.lr
        model_name = f"model{fold}"
        model_name = HBGinTNaowangluo.HBGinTransformer(args,args.input_size, args.hidden_size, args.output_size, args.num_heads,
                                                  args.norm_num, args.ratio, args.dropout).to(args.device)
        init_para =  model_name.state_dict()

        if not args.eval:
            acc_max,accu = train(args,fold, model_name,init_para)

        allaccu.append(accu)
        bestacc.append(acc_max)

    a = []
    array = np.array(allaccu)
    a.append(array[:,0])
    bestacc1 = np.array(bestacc)
    a.append(np.mean(array[:,0], axis=0))
    a.append(np.std(array[:,0]))
    bestacc.append(np.mean(bestacc1, axis=0))
    print('Accuracy summary:')
    print(a)

    canshu = [getargs()]
    list = [time.localtime()]  # 列表头名称
    with open('test.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list)
        writer.writerows([canshu])
        writer.writerows([array])
        writer.writerows([a] )



