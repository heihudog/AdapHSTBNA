import argparse
global A
NYU = 1
UM116 = 2  #0.0006   1e-5
ADNI = 3
nyu200 = 4  #1e-4 6e-6
um200 = 5
SITE1 = 8
SITE6 = 6 # 0.00003  3

A = 7
def getargs(ratio = 0.6,ratio1 = 0.4,A = A,lr = 0.00006):  #116  0.5
    parser = argparse.ArgumentParser(description='xx')
    parser.add_argument('--exp_name', type=str, default='H undec conv down brain', metavar='N',
                            help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=14, metavar='batch_size',
                            help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=7, metavar='batch_size',
                            help='Size of batch)')
    parser.add_argument('--minibatch_size', type=int, default=14, metavar='batch_size',
                            help='Size of minibatch_size)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=0,
                            help='Use SGD')
    parser.add_argument('--lr', type=float, default=lr,metavar='LR',  #太高时，验证集几乎全是1，太低时几乎全是0
                            help='learning rate (default: 0.006,0.001 0.1 if using sgd)')
    parser.add_argument('--layer', type=int, default=3, help='layer of model')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='LR',
                            help='weight_decay  (default: 0.0001,0.001 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
    parser.add_argument('--device', type=str, default='cuda',
                            help='enables CUDA training')
    parser.add_argument('--no_cuda', type=bool, default='False',
                            help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                            help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout rate')
    parser.add_argument('--mult_num', type=int, default=16, help=' ')
    parser.add_argument('--ratio', type=float, default = ratio , help='0  -  0.5')
    parser.add_argument('--num_pooling', type=int, default=2, help='')
    parser.add_argument('--k_fold', default=5, help='fold = 0,1,2,3,4')
    parser.add_argument('--cluster',type = int, default=16, help='num of roi')#16的时候很不错
    parser.add_argument('--dataset', type=int, default=A, help='num of roi')
    parser.add_argument('--model_path', type=str, default='', metavar='N')
    parser.add_argument('--Num', type=int, default=0, help=' ')
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--Gbias', type=bool, default=False, help='if bias ')
    if(A == 1):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/NYU116(1)")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=175, help=' ')
        parser.add_argument('--input_size', type=int, default=175, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' NUM OF ROI')
    if(A ==2):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/UM116")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=295, help=' ')
        parser.add_argument('--input_size', type=int, default=295, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help='NUM OF ROI ')
    if(A ==3):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/adni2")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=137, help=' ')
        parser.add_argument('--input_size', type=int, default=137, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' NUM OF ROI')
    if(A ==4):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/NYUcc200")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=175, help=' ')
        parser.add_argument('--input_size', type=int, default=175, help=' ')
        parser.add_argument('--norm_num', type=int, default=200, help='NUM OF ROI ')
    if(A ==5):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/UMcc200")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=295, help=' ')
        parser.add_argument('--input_size', type=int, default=295, help=' ')
        parser.add_argument('--norm_num', type=int, default=200, help=' NUM OF ROI')

    if(A ==8):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/MDD/SITE1")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=1833, help=' ')
        parser.add_argument('--input_size', type=int, default=1833, help=' ')
        parser.add_argument('--norm_num', type=int, default=3200, help=' NUM OF ROI')
    if(A ==6):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/MDD/SITE21") #20时  lay 2  0.00006比较好
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=230, help='232  20      230  21 ')
        parser.add_argument('--input_size', type=int, default=230, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help='NUM OF ROI ')
    if (A == 7):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/MDD/SITE25")  # 20时  lay 2  0.00006比较好
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=230, help='232  20      230  21 ')
        parser.add_argument('--input_size', type=int, default=230, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' NUM OF ROI')
    argvs = parser.parse_args()
    return argvs

