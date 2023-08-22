
import sys,os
from matplotlib import pyplot as plt

from numpy.matrixlib.defmatrix import matrix
from sklearn import manifold
sys.path.append(os.getcwd())
from Process.process_pheme_early_domain_data import *
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class_data import*
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold_pheme_early_domain import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import random
# from Process.mmd_distance import similarity_discriminator
from torch.autograd import Function
from domain_discriminator import *
from test_TSNE import plot_embedding
from torch.utils.data import ConcatDataset
import math
class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=False) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)
       
    def forward(self, x):

        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

    
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
        
        # ====================================add========================================

        self.fc=th.nn.Linear(2*out_feats,2)

        # self.fc=th.nn.Linear(4*out_feats,2)
        
        # ====================================add========================================

        self.hard_fc1 = hard_fc(out_feats, out_feats)
        

        
        

        self.conv3 = GCNConv(5, 32)
        self.conv4 = GCNConv(32, out_feats)
        # self.conv3 = GCNConv(5, 10)
        # self.conv4 = GCNConv(10, out_feats)
        



        





    def forward(self, data):
        # ====================================add========================================
      
        init_x, edge_index1, edge_index2= data.x, data.edge_index, data.edge_index2
        
        
        #print(data.id)
        #print(init_x.shape)
        #print(edge_index1.shape)
        # ======== embedding ============
        # print(init_x.shape)
        x1 = self.conv1(init_x, edge_index1) # anchor
        x1 = F.relu(x1)
        #x1 = F.dropout(x1, training=self.training)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        #print(x.shape) # 6124*64

        # pooling
        x1 = scatter_mean(x1, data.batch, dim=0) # (120, 64)
        
        
        # ====================================add========================================

        x1_g = x1
      
        x1 = self.hard_fc1(x1) 
        x1_t = x1
        #x1 = F.relu(x1)
        x1 = th.cat((x1_g, x1_t), 1)  #2*64 & 4*64



        # ======== embedding ============
        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        #x2 = F.dropout(x2, training=self.training)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0) # (120, 64)
        #x2 = self.proj2(x2)
        
        
        # ====================================add========================================

        x2_g = x2
        
        
       
        #print(x.shape)
        #for l, layer in enumerate(self.hard_fc1):
        #    x2 = layer(x2) #
        x2 = self.hard_fc1(x2) 
        x2_t = x2
        #x1 = F.relu(x1)
        x2 = th.cat((x2_g, x2_t), 1)

        #x2_hard = self.hard_fc(x2) #dim=64
        #x2_hard = F.relu(x2_hard)
        
        

        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)


        # ==================- cl_loss -====================
        
        x_T = x.t()
        dot_matrix = th.mm(x, x_T)


        x_norm = th.norm(x, p=2, dim=1) # shape: 2*batchsize
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())
        
        
        t = 0.8 #
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
        
      
        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        
        pos_matrix_list = cos_matrix.chunk(2, dim=0)
        p1_list = pos_matrix_list[0].chunk(2, dim=1)
        p2_list = pos_matrix_list[1].chunk(2, dim=1)
        p1 = th.diag(p1_list[1]).unsqueeze(1)
        p2 = th.diag(p2_list[0]).unsqueeze(1)
        #pos_matrix = th.cat((p1, p2), 0)
        pos_matrix = p1

        '''
        pos_matrix_list = cos_matrix.chunk(3, dim=0)
        p_list = pos_matrix_list[0].chunk(3, dim=1)
        p = th.diag(p_list[1])
        pos_matrix = p.unsqueeze(1)
        #print('pos_matrix.shape: ', pos_matrix.shape)
        #print('neg_matrix.shape: ', neg_matrix.shape)
        '''


        div = pos_matrix / neg_matrix
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

         
        # ======================- CE_loss -========================
        task_predict = self.fc(x)
        task_predict = F.log_softmax(task_predict, dim=1)
        #print(x.shape)
        # ======================- hard_CE_loss -========================



        return task_predict,cl_loss,x,y




#val train test
def train_GCN(x_val, x_train,x_test,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds):
    # ====================================add========================================

    # fgm = FGM(model)
    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    for para in model.hard_fc1.parameters():
        #print(para)
        para.requires_grad = True
    #optimizer_hard = th.optim.Adam(model.hard_fc.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_hard = th.optim.SGD(model.hard_fc1.parameters(), lr=0.001)

    model.train()
    # discriminator.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    #------------------------train-----------------------------------------
    for epoch in range(n_epochs):
        traindata_list,valdata_list,testdata_list = loadUdData(dataname, x_train,x_val, x_test,method, droprate=0.3)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        val_loader = DataLoader(valdata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0

        tqdm_train_loader = tqdm(train_loader) # tqdm
        NUM=1
        for Batch_data in tqdm_train_loader:

            
            Batch_data.to(device)

            src_predict,cl_loss, src_feature,src_labels= model(Batch_data)
            src_label_loss=F.nll_loss(src_predict,src_labels)

            loss = src_label_loss + 0.001 * cl_loss
            avg_loss.append(loss.item())


            
            ##--------------------------------##

            ##------------- S2 ---------------##



            optimizer.zero_grad()
            loss.backward()

            fgm.attack()


            src_predict,cl_loss, src_feature,src_labels= model(Batch_data)
            src_label_loss=F.nll_loss(src_predict,src_labels)

            loss_adv =src_label_loss + 0.001*cl_loss

            loss_adv.backward()
            fgm.restore()
            optimizer.step()

            _, pred = src_predict.max(dim=-1) #
            correct = pred.eq(src_labels).sum().item()
            train_acc = correct / len(src_labels) 
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
            NUM += 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
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
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                val_pred, y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model, method, seeds)
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.Acc1
            acc2 = early_stopping.Acc2
            pre1 = early_stopping.Prec1
            pre2 = early_stopping.Prec2
            rec1 = early_stopping.Recll1
            rec2 = early_stopping.Recll2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            
            break

    
    train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2=test_GCN(x_test, x_train,x_val,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds)
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2



def test_GCN(x_test, x_train,x_val,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds):
    model = GCN_Net(768,64,64).to(device)
    if method=='c':
        state_dict=th.load('./model_all_domain/random_division/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')
    elif method=='b':
        state_dict=th.load('./model_all_domain/event_division/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')
    elif method=='d':
        state_dict=th.load('./model_all_domain/time_division/GCN'+seeds+'_2bDANNALL_CH_checkpoint.pth')
    model.load_state_dict(state_dict)
    fgm=FGM(model)
    temp_val_losses = []
    temp_val_accs = []
    temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
    temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
    temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
    temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    traindata_list,traindata_100_list, testdata_list = loadUdData(dataname, x_train, x_val,x_test,method, droprate=0) # traindata_list：类实例化对象   testdata_list：类实例化对象 
    
    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=10)
    tqdm_test_loader = tqdm(test_loader)
    for Batch_data in tqdm_test_loader:
        Batch_data.to(device)
            
        val_out, val_cl_loss, val_feature,y = model(Batch_data)
        val_loss = F.nll_loss(val_out, y)
        temp_val_losses.append(val_loss.item())
        _, val_pred = val_out.max(dim=1) 
        correct = val_pred.eq(y).sum().item()
        val_acc = correct / len(y) 
        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                val_pred, y)
        temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
        temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
        temp_val_accs.append(val_acc)


    res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
    print('results:', res)
    accs = np.mean(temp_val_Acc_all)
    acc1 = np.mean(temp_val_Acc1)
    acc2 = np.mean(temp_val_Acc2)
    pre1 = np.mean(temp_val_Prec1)
    pre2 = np.mean(temp_val_Prec2)
    rec1 = np.mean(temp_val_Recll1)
    rec2 = np.mean(temp_val_Recll2)
    F1 = np.mean(temp_val_F1)
    F2 = np.mean(temp_val_F2)
    train_losses, val_losses, train_accs, val_accs=[],[],[],[]
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2

class GCN1(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
    def forward(self, data):
        # ====================================add========================================
      
        init_x, edge_index1, edge_index2= data.x, data.edge_index, data.edge_index2
        x1 = self.conv1(init_x, edge_index1) # anchor
        # x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        return x1,data.y1
def test_distribution(x_test,x_100_train, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,method,seeds):

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    feature=None
    traindata_list,traindata_100_list, testdata_list = loadUdData(dataname, x_train, x_100_train,x_test,method, droprate=0)
    concat_dataset = ConcatDataset([ traindata_list, traindata_100_list, testdata_list])
    print(len(concat_dataset))
    test_loader = DataLoader(concat_dataset, batch_size=batchsize, shuffle=True, num_workers=10)
    tqdm_test_loader = tqdm(test_loader)
    for Batch_data in tqdm_test_loader:
        Batch_data.to(device)
            
        val_feature,y = model(Batch_data)
        
        if out_label_ids is None:
            
            out_label_ids = y.detach().cpu().numpy()
            feature=val_feature.detach().cpu().numpy()
        else:
            out_label_ids = np.append(out_label_ids, y.detach().cpu().numpy(), axis=0)
            feature=np.append(feature,val_feature.detach().cpu().numpy(), axis=0)
    print(out_label_ids)
    print(out_label_ids.shape)
    print(feature.shape)
    tsne = manifold.TSNE(n_components=2, init='pca')
    features_tsne = tsne.fit_transform(feature)
    # features_tsne_1 = tsne.fit_transform(data_1)
    # print(features_tsne.shape)
    
        # print(features_tsne.shape)

    domain_0=np.zeros(len(traindata_list),dtype = int)
    domain_1=np.ones((len(traindata_100_list)+len(testdata_list)),dtype = int)
    domain=np.concatenate((domain_0,domain_1),axis=0)

        # print(features_tsne)
        #
    plot_embedding(dataname,features_tsne, out_label_ids,domain, "")

    plt.show()
    plt.savefig('./data/t-SNE/GACL_class'+dataname+seeds)

##---------------------------------main-train---------------------------------------


def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True




lr=0.0005
weight_decay=1e-4
patience=35
n_epochs=200
batchsize=120 # twitter

datasetname='Pheme'
iterations=1
model="GCN"
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# device = th.device('cpu')
print(device)


# method='b'
method=sys.argv[2]
type=sys.argv[1]
if method=='c':
    path='./model_all_domain/random_division/'
elif method=='b':
    path='./model_all_domain/event_division/'
path_other=''
# path_other=''

# type='train'

if type=='train' and method=='b':
    print('TRAIN')
    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768,64,64).to(device)

        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
    
        
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(
                                                                                               fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')
        setup_seed(30)
        print('seed=30, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(
                                                                                               fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'30')
        setup_seed(40)
        print('seed=40, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(
                                                                                               fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'40')
        setup_seed(50)
        print('seed=50, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(
                                                                                               fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'50')
        setup_seed(60)
        print('seed=60, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(
                                                                                                fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'60')
    
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))



elif type=='train' and method=='c':
    print('TRAIN')
    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768,64,64).to(device)

        fgm=FGM(model)
        fold0_test,fold0_val,fold0_train,\
           fold1_test,fold1_val,fold1_train,\
           fold2_test,fold2_val,fold2_train,\
           fold3_test,fold3_val,fold3_train,\
           fold4_test, fold4_val,fold4_train= load5foldData(datasetname,4)
    
        
        print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
        print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold1_val))
        print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold2_val))
        print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold3_val))
        print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold4_val))
        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(
                                                                                               fold0_val,
                                                                                               fold0_train,
                                                                                               fold0_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')
       
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
       
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(
                                                                                              fold1_val,
                                                                                               fold1_train,
                                                                                               fold1_test,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'30')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)

        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(
                                                                                              fold2_val,
                                                                                               fold2_train,
                                                                                               fold2_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'40')
        
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
     
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(
                                                                                              fold3_val,
                                                                                               fold3_train,
                                                                                               fold3_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'50')
       
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(
                                                                                                fold4_val,
                                                                                               fold4_train,
                                                                                               fold4_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'60')
    
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))
elif type=='train' and method=='d':
    print('TRAIN')
    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768,64,64).to(device)

        fgm=FGM(model)
        fold0_x_test, fold0_x_val,fold0_x_train= load5foldData(datasetname,1)
    
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val),len(fold0_x_train))
        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(
                                                                                               fold0_x_val,
                                                                                               fold0_x_train,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')
        setup_seed(30)
        print('seed=30, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_val,fold0_x_train= load5foldData(datasetname,1)
    
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val),len(fold0_x_train))
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(
                                                                                               fold0_x_val,
                                                                                               fold0_x_train,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'30')
        setup_seed(40)
        print('seed=40, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_val,fold0_x_train= load5foldData(datasetname,1)
    
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val),len(fold0_x_train))
        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(
                                                                                               fold0_x_val,
                                                                                               fold0_x_train,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'40')
        setup_seed(50)
        print('seed=50, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_val,fold0_x_train= load5foldData(datasetname,1)
    
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val),len(fold0_x_train))
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(
                                                                                               fold0_x_val,
                                                                                               fold0_x_train,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'50')
        setup_seed(60)
        print('seed=60, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        fgm=FGM(model)
        fold0_x_test, fold0_x_val,fold0_x_train= load5foldData(datasetname,1)
    
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val),len(fold0_x_train))
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(
                                                                                               fold0_x_val,
                                                                                               fold0_x_train,
                                                                                               fold0_x_test,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'60')
    
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))





elif type=='test' and method=='b':
    print('TEST')
    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN20_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = test_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_100_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')
        setup_seed(30)
        print('seed=30, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN30_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = test_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_100_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'30')
        setup_seed(40)
        print('seed=40, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN40_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = test_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_100_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'40')
        setup_seed(50)
        print('seed=50, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN50_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = test_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_100_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'50')
        setup_seed(60)
        print('seed=60, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN60_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = test_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train_8,
                                                                                               fold0_x_100_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'60')
    
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))

elif type=='test' and method=='c':
    print('TEST')
    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]

    th.backends.cudnn.enabled = False
    for iter in range(iterations):

        setup_seed(20)
        print('seed=20, t=0.8')
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN20_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)

        fold0_test,fold0_val,fold0_train,\
           fold1_test,fold1_val,fold1_train,\
           fold2_test,fold2_val,fold2_train,\
           fold3_test,fold3_val,fold3_train,\
           fold4_test, fold4_val,fold4_train= load5foldData(datasetname,4)
    
        
        print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
        print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold0_val))
        print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold0_val))
        print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold0_val))
        print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold0_val))

        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = test_GCN(
                                                                                               fold0_test,
                                                                                               fold0_train,
                                                                                               fold0_val,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')
     
        
        
    
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN30_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = test_GCN(
                                                                                              fold1_test,
                                                                                               fold1_train,
                                                                                               fold1_val,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'30')
        
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN40_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)

        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = test_GCN(
                                                                                              fold2_test,
                                                                                               fold2_train,
                                                                                               fold2_val,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'40')
       
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN50_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = test_GCN(
                                                                                              fold3_test,
                                                                                               fold3_train,
                                                                                               fold3_val,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'50')
       
        model = GCN_Net(768,64,64).to(device)
        state_dict=th.load(path+'GCN60_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        model.load_state_dict(state_dict)
        fgm=FGM(model)
        
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = test_GCN(
                                                                                               fold4_test,
                                                                                               fold4_train,
                                                                                               fold4_val,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'60')
    
    
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))


elif type=='test_data':
    for iter in range(iterations):
        setup_seed(20)
        print('seed=20, t=0.8')
        # model = GCN_Net(768,64,64).to(device)
        # state_dict=th.load(path+'GCN20_2bDANNALL_CH'+path_other+'_checkpoint.pth')
        # model.load_state_dict(state_dict)
        # fgm=FGM(model)
        model=GCN1(768,64,64).to(device)
        fold0_x_test, fold0_x_train_8,fold0_x_train_2,fold0_x_100_train= load5foldData(datasetname,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8),len(fold0_x_train_2),len(fold0_x_100_train))
        test_distribution(
                                                                                               fold0_x_test,
                                                                                               fold0_x_100_train,
                                                                                               fold0_x_train_8,
                                        
                                                                                               
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter,method,'20')

