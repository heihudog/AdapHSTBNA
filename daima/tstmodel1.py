import os
from dataread import dataset
from model import HBGinTNaowangluoNYU  as HBGinTNaowangluo
import torch
import time
import sklearn.metrics as metrics

from option import getargs
from utils import util
args = getargs()
device = args.device

def testmodel(model_path):
    with (torch.no_grad()):
        count = 0.0

        model =   HBGinTNaowangluo.HBGinTransformer(args,args.input_size, args.hidden_size, args.output_size, args.num_heads,
                                                  args.norm_num, args.ratio, args.dropout).to(args.device)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict,strict=False)
        init_para = model.state_dict()
        t = time.strftime('%Y-%m-%d%H-%M-%S', time.localtime())
        filename = os.path.basename(args.data_path)

        model.eval()
        total_time = 0.0
        data, label = dataset.dataread(train=False)
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

        label = label.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        print('test total time is', total_time)

        test_acc = metrics.accuracy_score(label, preds)
        evaluation_metrics = util.evaluate(pred=preds, prob=Prob, label=label)
        return logits.softmax(1)[:,1].cpu(),test_acc




#m1 = 'save\model\  2025-02-2120-16-31NYU116(1) 0 1.pth'
m1 = 'save\mdd\  2025-02-2703-00-11SITE25 0 2.pth'
testmodel(m1)

preds = []
labels = []
m = []





file_path = 'a.pt'  # 替换为实际的 .pt 文件路径
loaded_data = torch.load(file_path).cpu().numpy().mean(0)
loaded_data = loaded_data[:90, :90]
#print(loaded_data.shape)

import numpy as np


def process_matrix(input_matrix):
    mat = np.array(input_matrix)
    n = mat.shape[0]
    result = np.zeros_like(mat)
    np.fill_diagonal(result, np.diag(mat))
    upper_rows, upper_cols = np.triu_indices(n, k=1)
    if upper_rows.size > 0:
        upper_values = mat[upper_rows, upper_cols]
        sorted_upper_indices = np.argsort(upper_values)[::-1]
        take_upper = min(10, len(sorted_upper_indices))
        selected_upper = sorted_upper_indices[:take_upper]
        result[upper_rows[selected_upper], upper_cols[selected_upper]] = mat[
            upper_rows[selected_upper], upper_cols[selected_upper]]

    lower_rows, lower_cols = np.tril_indices(n, k=-1)
    if lower_rows.size > 0:
        lower_values = mat[lower_rows, lower_cols]
        sorted_lower_indices = np.argsort(lower_values)[::-1]
        take_lower = min(10, len(sorted_lower_indices))
        selected_lower = sorted_lower_indices[:take_lower]
        result[lower_rows[selected_lower], lower_cols[selected_lower]] = mat[
            lower_rows[selected_lower], lower_cols[selected_lower]]

    return result.tolist()



loaded_data = process_matrix(loaded_data)

for row in loaded_data:
    print(row)

np.savetxt('Mdd.txt', loaded_data)







