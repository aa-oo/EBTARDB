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
        self.Acc1=0
        self.Prec1=0
        self.Recll1=0
        self.F2 = 0
        self.Acc2=0
        self.Prec2=0
        self.Recll2=0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,Acc1,Acc2,Prec1,Prec2,Recll1,Recll2,F1,F2,model,modelname,str):

        score = (accs+F1+Acc1+Prec1+Recll1+F2+Acc2+Prec2+Recll2) /9

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.Acc1=Acc1
            self.Prec1=Prec1
            self.Recll1=Recll1
            self.F2 = F2
            self.Acc2=Acc2
            self.Prec2=Prec2
            self.Recll2=Recll2
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.3f}|R F1: {:.3f}|R Acc: {:.3f}|R Prec: {:.3f}|R Recll: {:.3f}|N F1: {:.3f}|N Acc: {:.3f}|N Prec: {:.3f}|N Recll: {:.3f}"
                      .format(self.accs,self.F1,self.Acc1,self.Prec1,self.Recll1,self.F2,self.Acc2,self.Prec2,self.Recll2))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.Acc1=Acc1
            self.Prec1=Prec1
            self.Recll1=Recll1
            self.F2 = F2
            self.Acc2=Acc2
            self.Prec2=Prec2
            self.Recll2=Recll2
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
        # torch.save(model.state_dict(), './model3/'+modelname+str+'_2bDANNfeo3_checkpoint.pth')
        if modelname=='c':
            torch.save(model.state_dict(), './model_all_domain/random_division/GCN' + str + '_2bDANNALL_CH_checkpoint.pth')
        elif modelname=='b':
            torch.save(model.state_dict(), './model_all_domain/event_division/GCN' + str + '_2bDANNALL_CH_checkpoint.pth')
        elif modelname=='d':
            torch.save(model.state_dict(), './model_all_domain/time_division/GCN' + str + '_2bDANNALL_CH_checkpoint.pth')
        # torch.save(model, 'finish_model.pkl')
        # torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss
        
