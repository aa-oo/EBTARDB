import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.Acc1 = 0
        self.Prec1 = 0
        self.Recll1 = 0

        self.Acc2 = 0
        self.Prec2 = 0
        self.Recll2 = 0

        self.Acc3 =0
        self.Prec3 = 0
        self.Recll3 = 0

        self.Acc4 = 0
        self.Prec4 = 0
        self.Recll4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4,model,modelname,str):


        score = (accs+F1+F2+F3+F4+Acc1+Acc2+Acc3+Acc4+Prec1+Prec2+Prec3+Prec4+Recll1+Recll2+Recll3+Recll4) /17
        

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.Acc1=Acc1
            self.Prec1=Prec1
            self.Recll1=Recll1
            
            self.Acc2=Acc2
            self.Prec2=Prec2
            self.Recll2=Recll2
            
            self.Acc3=Acc3
            self.Prec3=Prec3
            self.Recll3=Recll3
            
            self.Acc4=Acc4
            self.Prec4=Prec4
            self.Recll4=Recll4
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                #       .format(self.accs,self.F1,self.F2,self.F3,self.F4))
                print("BEST Accuracy: {:.4f}|UR Acc: {:.4f}|UR Prec: {:.4f}|UR Recll: {:.4f}|UR F1: {:.4f}|NR Acc: {:.4f}|NR Prec: {:.4f}|NR Recll: {:.4f}|NR F1: {:.4f}|TR Acc: {:.4f}|TR Prec: {:.4f}|TR Recll: {:.4f}|TR F1: {:.4f}|FR Acc: {:.4f}|FR Prec: {:.4f}|FR Recll: {:.4f}|FR F1: {:.4f}".format(self.accs,self.Acc1,self.Prec1,self.Recll1,self.F1,self.Acc2,self.Prec2,self.Recll2,self.F2,self.Acc3,self.Prec3,self.Recll3,self.F3,self.Acc4,self.Prec4,self.Recll4,self.F4))


        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.Acc1=Acc1
            self.Prec1=Prec1
            self.Recll1=Recll1
            
            self.Acc2=Acc2
            self.Prec2=Prec2
            self.Recll2=Recll2
            
            self.Acc3=Acc3
            self.Prec3=Prec3
            self.Recll3=Recll3
            
            self.Acc4=Acc4
            self.Prec4=Prec4
            self.Recll4=Recll4
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''
                Saves model when validation loss decrease.

                '''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        torch.save(model.state_dict(), './model_all_domain/'+modelname+'/random_division'+'/GCN'+str+'_2bDANNALL_CH_checkpoint.pth')
        # torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss
        
