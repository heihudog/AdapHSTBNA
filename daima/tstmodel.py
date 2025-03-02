import os
from dataread import dataset
from model import HBGinTNaowangluoNYU as HBGinTNaowangluo
import torch
import time
import sklearn.metrics as metrics

from option import getargs
from utils import util
args = getargs()
device = args.device

with (torch.no_grad()):
    test_loss = 0.0
    count = 0.0

    model = HBGinTNaowangluo.HBGinTransformer(args,args.input_size, args.hidden_size, args.output_size, args.num_heads,
                                                   args.norm_num, args.ratio, args.dropout).to(args.device)
    model_dict = torch.load('save\model\  2025-01-2220-36-41UM116 0 0.pth')
    model.load_state_dict(model_dict,strict=False)
    init_para = model.state_dict()
    t = time.strftime('%Y-%m-%d%H-%M-%S', time.localtime())
    filename = os.path.basename(args.data_path)

    model.eval()
    total_time = 0.0
    data, label = dataset.dataread(train = False)
    print(data.shape)
    data = data.to(device)
    label = label.to(device)
    batch_size = data.size()[0]
    start_time = time.time()
    logits, loss1 = model(data)
    logits = logits.squeeze(1)
    label = label.to(torch.float32)
    value22, preds = torch.max(logits.data, 1)
    print(logits)
    print(preds)
    Prob = logits.softmax(1).tolist()
    end_time = time.time()
    total_time += (end_time - start_time)
    count += batch_size

    # 记录每个被试之间的距离
    label = label.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()

    print('test total time is', total_time)

    test_acc = metrics.accuracy_score(label, preds)
    evaluation_metrics = util.evaluate(pred=preds, prob=Prob, label=label)

    outstr = 'Test  loss: %.6f,  test acc: %.6f' % (test_loss * 1.0 / count, test_acc)


    print(outstr)

