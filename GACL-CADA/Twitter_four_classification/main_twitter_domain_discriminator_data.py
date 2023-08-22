# --------- GACL meathod ------------


import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
#twitter pheme
# from Process.process import * 
from Process.process_early_data import *
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping_data import *
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold_early import *
# from Process.rand5fold1 import *
from others.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import random
from torch.autograd import Function
from domain_discriminator import *

class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    # def attack(self, epsilon=0.3, emb_name='hard_fc1.'): # T15: epsilon = 0.2
    def attack(self, epsilon=0.2, emb_name='hard_fc1.'): 
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if norm != 0 and not th.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc1.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GCN_Net(th.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc=th.nn.Linear(2*out_feats,4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats) # optional
    def forward(self, data):
        init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x, data.edge_index, data.edge_index2
        
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2 
        x2 = th.cat((x2_g, x2_t), 1)
        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = th.mm(x, x_T)
        x_norm = th.norm(x, p=2, dim=1)
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())
        
        # t = 0.3 # pheme: t = 0.6
        t=0.3
        cos_matrix = (dot_matrix / norm_matrix) / t
        cos_matrix = th.exp(cos_matrix)
        diag = th.diag(cos_matrix)
        cos_matrix_diag = th.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = th.ne(y_matrix, y_matrix_T).float()
        #y_matrix_list = y_matrix.chunk(3, dim=0)
        #y_matrix = y_matrix_list[0]
        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)
        #neg_matrix = neg_matrix_list[0]
        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2,dim=0)
        #print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        #print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        pos_matrix = pos_matrix_list[0]
        #print('pos shape: ', pos_matrix.shape, pos_matrix)
        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        #print('neg shape: ', neg_matrix.shape)
        div = pos_matrix / neg_matrix 
        div = (th.sum(div, dim=1)).unsqueeze(1)  
        div = div / batchsize
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

        task_predict = self.fc(x) #(128, 4)
        task_predict = F.log_softmax(task_predict, dim=1)

    
        return task_predict,cl_loss,x, y

#val train test
def train_GCN(x_train, x_val,x_test,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter, method,seeds):

    
    # fgm = FGM(model)
    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    for para in model.hard_fc2.parameters():
        para.requires_grad = False
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # optional ------ S1 ----------
    for para in model.hard_fc1.parameters():
        para.requires_grad = True
    for para in model.hard_fc2.parameters():
        para.requires_grad = True
    #optimizer_hard = th.optim.Adam(model.hard_fc.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_hard = th.optim.SGD([{'params': model.hard_fc1.parameters()},
                                    {'params': model.hard_fc2.parameters()}], lr=0.001)

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
     

    for epoch in range(n_epochs): 
        if dataname=='Twitter15':
            traindata_list, val_list,testdata_list = loadData(dataname, x_train,x_val, x_test,method, droprate=0.1) # T15 droprate = 0.1
        elif dataname=='Twitter16':
            traindata_list,val_list, testdata_list = loadData(dataname, x_train,x_val, x_test,method, droprate=0.4) # T15 droprate = 0.1
            
        # codeDic={} #twitter pheme
        # traindata_list, testdata_list = loadData(dataname, x_train, x_test,codeDic, droprate=0.1) # T15 droprate = 0.1
        
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)

        val_loader = DataLoader(val_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        NUM=1
        beta=0.001
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            # Batch_100_data.to(device)

            src_predict,cl_loss, src_feature,src_labels= model(Batch_data)
            src_label_loss=F.nll_loss(src_predict,src_labels) 

            loss=src_label_loss+0.001*cl_loss
            avg_loss.append(loss.item())
            ##------------- S1 ---------------##
            '''
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            avg_loss.append(loss.item())
            optimizer.step()
            epsilon = 3
            loss_ad = epsilon/(finalloss + 0.001*cl_loss)
            print('loss_ad: ', loss_ad)
            optimizer_hard.zero_grad()
            loss_ad.backward()
            optimizer_hard.step()
            '''
            ##--------------------------------##

            ##------------- S2 ---------------##
            optimizer.zero_grad()
            loss.backward()
            fgm.attack()
            src_predict,cl_loss, src_feature,src_labels= model(Batch_data)
            src_label_loss=F.nll_loss(src_predict,src_labels) 

            loss_adv=src_label_loss+0.001*cl_loss
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            ##--------------------------------##

            _, pred = src_predict.max(dim=-1)
            correct = pred.eq(src_labels).sum().item()
            train_acc = correct / len(src_labels) 
            
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,loss.item(),train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
            #print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        #twitter
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        tqdm_val_loader = tqdm(val_loader)
        for Batch_data in tqdm_val_loader:
            Batch_data.to(device)
            val_out, val_cl_loss, val_feature,y = model(Batch_data)

            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            #twitter
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss), np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        #twitter
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        
        print('results:', res)
      
        if epoch > 25 and method=='c':
           early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_Acc1),np.mean(temp_val_Acc2), np.mean(temp_val_Acc3), np.mean(temp_val_Acc4),  model,  dataname,seeds,method)
           
        accs =np.mean(temp_val_accs)

    
        Acc1=np.mean(temp_val_Acc1)
        
        Acc2=np.mean(temp_val_Acc2)
        
        Acc3=np.mean(temp_val_Acc3)
        
        Acc4=np.mean(temp_val_Acc4)
        
        if early_stopping.early_stop and method=='c':
            print("Early stopping")
            accs=early_stopping.accs
           
            Acc1=early_stopping.Acc1
            Acc2 = early_stopping.Acc2
            Acc3 = early_stopping.Acc3
            Acc4 = early_stopping.Acc4
            break
    
    if method=='b':
        th.save(model.state_dict(),
            './model_all_domain/'+dataname+'/event_division'+'/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')

    accs,Acc1, Acc2, Acc3, Acc4, =test_GCN(x_test, x_train,x_val,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds)
    #twitter
    return accs,Acc1, Acc2, Acc3, Acc4
    
def test_GCN(x_test, x_train,x_val,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds):
    model = GCN_Net(768, 64, 64).to(device)
    if method=='c':
        state_dict=th.load('./model_all_domain/'+dataname+'/random_division'+'/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')
    elif method=='b':
        state_dict=th.load( './model_all_domain/'+dataname+'/event_division'+'/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')
        
    model.load_state_dict(state_dict)
    fgm = FGM(model)
    temp_val_losses = []
    temp_val_accs = []
    temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
    temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
    temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
    temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    traindata_list,traindata_100_list, testdata_list = loadData(dataname, x_train, x_val,x_test,method, droprate=0) # traindata_list：类实例化对象   testdata_list：类实例化对象 
    
    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
    tqdm_test_loader = tqdm(test_loader)
    for Batch_data in tqdm_test_loader:
        Batch_data.to(device)
            
        val_out, val_cl_loss, val_feature,y = model(Batch_data)
        val_loss = F.nll_loss(val_out, y)
        temp_val_losses.append(val_loss.item())
        _, val_pred = val_out.max(dim=1) 
        correct = val_pred.eq(y).sum().item()
        val_acc = correct / len(y) 

        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
        temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
        temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
        temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
        temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
        temp_val_accs.append(val_acc)

        #twitter
    res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        
    print('results:', res)
      

    # accall=np.mean(temp_val_Acc_all)       
    accs =np.mean(temp_val_accs)

    
    Acc1=np.mean(temp_val_Acc1)
   
    Acc2=np.mean(temp_val_Acc2)
    
    Acc3=np.mean(temp_val_Acc3)
    
    Acc4=np.mean(temp_val_Acc4)


        
    #twitter
    return accs,Acc1, Acc2,  Acc3,  Acc4


def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True
##---------------------------------main---------------------------------------


scale = 1
lr=0.0005 * scale
weight_decay=1e-4
patience=10
n_epochs=100
batchsize=120
datasetname=sys.argv[3] # (1)Twitter15  (2)pheme  (3)weibo
# datasetname='Twitter15'
iterations=1
#model="GCN"

# device = th.device('cuda:4' if th.cuda.is_available() else 'cpu')
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

if datasetname=='Twitter16':
    data_path = './data/twitter16/'
    laebl_path = './data/Twitter16_label_All.txt'
elif datasetname=='Twitter15':
    data_path = './data/twitter15/'
    laebl_path = './data/Twitter15_label_All.txt'

method=sys.argv[2]
type=sys.argv[1]
# method='b' #
if method=='c':
    path='./model_all_domain/'+datasetname+'/random_division/'
elif method=='b':
    path='./model_all_domain/'+datasetname+'/event_division/'
path_other=''
# type = 'train'

if type == 'train' and method=='b':
    print('TRAIN')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)

        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_0, acc1_0,acc2_0, acc3_0,  acc4_0 = train_GCN(
            
            fold0_x_train_8,
            fold0_x_100_train,
            fold0_x_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '20')
        setup_seed(30)
        print('seed=30, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_1, acc1_1,  acc2_1, acc3_1, acc4_1 = train_GCN(
            
            fold0_x_train_8,
            fold0_x_100_train,
            fold0_x_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '30')
        setup_seed(40)
        print('seed=40, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_2, acc1_2,acc2_2, acc3_2,acc4_2 = train_GCN(
            
            fold0_x_train_8,
            fold0_x_100_train,
            fold0_x_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '40')
        setup_seed(50)
        print('seed=50, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_3, acc1_3,acc2_3, acc3_3, acc4_3 = train_GCN(
            
            fold0_x_train_8,
            fold0_x_100_train,
            fold0_x_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '50')
        setup_seed(60)
        print('seed=60, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_4, acc1_4,  acc2_4, acc3_4, acc4_4= train_GCN(
            
            fold0_x_train_8,
            fold0_x_100_train,
            fold0_x_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '60')

        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)


    print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                    sum(ACC2) / iterations, sum(ACC3) / iterations,
                                                                    sum(ACC4) / iterations))

elif type == 'train' and method=='c':
    print('TRAIN')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)

        fgm = FGM(model)

        fold0_test,fold0_val,fold0_train,\
           fold1_test,fold1_val,fold1_train,\
           fold2_test,fold2_val,fold2_train,\
           fold3_test,fold3_val,fold3_train,\
           fold4_test, fold4_val,fold4_train= load5foldData(datasetname,data_path,laebl_path,4)
    
        
        print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
        print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold1_val))
        print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold2_val))
        print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold3_val))
        print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold4_val))

        accs_0, acc1_0,  acc2_0, acc3_0, acc4_0 = train_GCN(
            
            fold0_train,
            fold0_val,
            fold0_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '20')
       
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        accs_1, acc1_1,  acc2_1, acc3_1, acc4_1 = train_GCN(
            
            fold1_train,
            fold1_val,
            fold1_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '30')
      
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)

        accs_2, acc1_2, acc2_2,acc3_2,  acc4_2 = train_GCN(
            
            fold2_train,
            fold2_val,
            fold2_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '40')
      
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
       
        accs_3, acc1_3,acc2_3, acc3_3,  acc4_3 = train_GCN(
            
            fold3_train,
            fold3_val,
            fold3_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '50')
       
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        
        accs_4, acc1_4, acc2_4, acc3_4, acc4_4 = train_GCN(
            
            fold4_train,
            fold4_val,
            fold4_test,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '60')

        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)

    print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                    sum(ACC2) / iterations, sum(ACC3) / iterations,
                                                                    sum(ACC4) / iterations))

if type == 'test' and method=='b':
    print('TRAIN')
    test_accall,test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [],[]

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        state_dict=th.load(path+'GCN20_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0, acc4_0, pre4_0, rec4_0, F4_0 = test_GCN(
            fold0_x_test,
            fold0_x_train_8,
            fold0_x_100_train,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '20')
        setup_seed(30)
        print('seed=30, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        state_dict=th.load(path+'GCN30_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 ,acc3_1, pre3_1, rec3_1, F3_1, acc4_1, pre4_1, rec4_1, F4_1 = test_GCN(
            fold0_x_test,
            fold0_x_train_8,
            fold0_x_100_train,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '30')
        setup_seed(40)
        print('seed=40, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        state_dict=th.load(path+'GCN40_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 ,acc3_2, pre3_2, rec3_2, F3_2, acc4_2, pre4_2, rec4_2, F4_2 = test_GCN(
            fold0_x_test,
            fold0_x_train_8,
            fold0_x_100_train,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '40')
        setup_seed(50)
        print('seed=50, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        state_dict=th.load(path+'GCN50_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3, acc3_3, pre3_3, rec3_3, F3_3, acc4_3, pre4_3, rec4_3, F4_3 = test_GCN(
           fold0_x_test,
            fold0_x_train_8,
            fold0_x_100_train,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '50')
        setup_seed(60)
        print('seed=60, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)
        state_dict=th.load(path+'GCN60_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm = FGM(model)
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4, acc4_4, pre4_4, rec4_4, F4_4 = test_GCN(
            fold0_x_test,
            fold0_x_train_8,
            fold0_x_100_train,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '60')

        # test_accall.append((accs_all0+accs_all1+accs_all2+accs_all3+accs_all4)/5)
        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4) / 5)
        PRE4.append((pre4_0 + pre4_1 + pre4_2 + pre4_3 + pre4_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4) / 5)
        REC4.append((rec4_0 + rec4_1 + rec4_2 + rec4_3 + rec4_4) / 5)
        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        F4.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

    print("Twitter:|Total_ALL_ Accuracy: {:.4f}|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}|pre4: {:.4f}"
          "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|rec4: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|F4: {:.4f}".format(sum(test_accall)/iterations,sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                    sum(ACC2) / iterations, sum(ACC3) / iterations,
                                                                    sum(ACC4) / iterations,sum(PRE1) / iterations,
                                                                    sum(PRE2) / iterations,sum(PRE3) / iterations,
                                                                    sum(PRE4) / iterations,
                                                                    sum(REC1) / iterations, sum(REC2) / iterations,
                                                                    sum(REC3) / iterations, sum(REC4) / iterations,
                                                                    sum(F1) / iterations, sum(F2) / iterations,
                                                                    sum(F3) / iterations, sum(F4) / iterations))

elif type == 'test' and method=='c':
    print('TRAIN')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768, 64, 64).to(device)

        fgm = FGM(model)

        fold0_test,fold0_val,fold0_train,\
           fold1_test,fold1_val,fold1_train,\
           fold2_test,fold2_val,fold2_train,\
           fold3_test,fold3_val,fold3_train,\
           fold4_test, fold4_val,fold4_train= load5foldData(datasetname,data_path,laebl_path,4)
    
        
        print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
        print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold0_val))
        print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold0_val))
        print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold0_val))
        print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold0_val))

        accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0, acc4_0, pre4_0, rec4_0, F4_0 = test_GCN(
            fold0_test,
            fold0_train,
            fold0_val,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '20')
       
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 ,acc3_1, pre3_1, rec3_1, F3_1, acc4_1, pre4_1, rec4_1, F4_1 = test_GCN(
            fold1_test,
            fold1_train,
            fold1_val,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '30')
      
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)

        accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 ,acc3_2, pre3_2, rec3_2, F3_2, acc4_2, pre4_2, rec4_2, F4_2 = test_GCN(
            fold2_test,
            fold2_train,
            fold2_val,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '40')
      
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
       
        accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3, acc3_3, pre3_3, rec3_3, F3_3, acc4_3, pre4_3, rec4_3, F4_3 = test_GCN(
            fold3_test,
            fold3_train,
            fold3_val,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '50')
       
        model = GCN_Net(768, 64, 64).to(device)
        fgm = FGM(model)
        
        accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4, acc4_4, pre4_4, rec4_4, F4_4 = test_GCN(
            fold4_test,
            fold4_train,
            fold4_val,

            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter, method, '60')

        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4) / 5)
        PRE4.append((pre4_0 + pre4_1 + pre4_2 + pre4_3 + pre4_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4) / 5)
        REC4.append((rec4_0 + rec4_1 + rec4_2 + rec4_3 + rec4_4) / 5)
        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        F4.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

    print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}|pre4: {:.4f}"
          "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|rec4: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|F4: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                    sum(ACC2) / iterations, sum(ACC3) / iterations,
                                                                    sum(ACC4) / iterations,sum(PRE1) / iterations,
                                                                    sum(PRE2) / iterations,sum(PRE3) / iterations,
                                                                    sum(PRE4) / iterations,
                                                                    sum(REC1) / iterations, sum(REC2) / iterations,
                                                                    sum(REC3) / iterations, sum(REC4) / iterations,
                                                                    sum(F1) / iterations, sum(F2) / iterations,
                                                                    sum(F3) / iterations, sum(F4) / iterations))

elif type == 'test_data':
    print('test_data')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    th.backends.cudnn.enabled = False
    len_train=0
    len_test=0
    for iter in range(iterations):
        l1=l2=l3=l4=0
        l5=l6=l7=l8=0
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['non-rumor'], ['false'], ['true'], ['unverified']
        if datasetname=='Twitter15':
            label_path1 = './data/label_15.json'
        elif datasetname=='Twitter16':
            label_path1 = './data/label_16.json'
        with open(label_path1, encoding='utf-8') as f:
            json_inf = json.load(f)
        print('The len of file_list: ', len(json_inf))
        setup_seed(20)
        print('seed=20, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train= load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
        setup_seed(30)
        print('seed=30, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
       
        setup_seed(40)
        print('seed=40, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))

        setup_seed(50)
        print('seed=50, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
    
        setup_seed(60)
        print('seed=60, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))

        l1=l1/5
        l2=l2/5
        l3=l3/5
        l4=l4/5
        l5=l5/5
        l6=l6/5
        l7=l7/5
        l8=l8/5
        # print(int(l1),int(l2),int(l4),int(l3))
        # print(int(l5),int(l6),int(l8),int(l7))
        print(round(l1),round(l2),round(l4),round(l3))
        print(round(l5),round(l6),round(l8),round(l7))






