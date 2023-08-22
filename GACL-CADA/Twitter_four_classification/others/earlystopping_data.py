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
        
        self.Acc1 = 0
        
        self.Acc2 = 0
        
        self.Acc3 =0
        
        self.Acc4 = 0

        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,Acc1, Acc2, Acc3, Acc4, model,modelname,str,data_division):


        score = (accs+Acc1+Acc2+Acc3+Acc4) /5
        

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            
            self.Acc1=Acc1
           
            self.Acc2=Acc2
           
            self.Acc3=Acc3
            
            self.Acc4=Acc4
        
            self.save_checkpoint(val_loss, model,modelname,str,data_division)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                #       .format(self.accs,self.F1,self.F2,self.F3,self.F4))
                print("BEST Accuracy: {:.4f}|UR Acc: {:.4f}|NR Acc: {:.4f}|TR Acc: {:.4f}|FR Acc: {:.4f}".format(self.accs,self.Acc1,self.Acc2,self.Acc3,self.Acc4))


        else:
            self.best_score = score
            self.accs = accs
           
            self.Acc1=Acc1
        
            
            self.Acc2=Acc2
           
            self.Acc3=Acc3
            
            self.Acc4=Acc4
            
            self.save_checkpoint(val_loss, model,modelname,str,data_division)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str,data_division):
        '''
                Saves model when validation loss decrease.

                '''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        if data_division=='c':
            torch.save(model.state_dict(), './model_all_domain/'+modelname+'/random_division'+'/GCN'+str+'_2bDANNALL_CH_checkpoint.pth')
        elif data_division=='b':
            torch.save(model.state_dict(), './model_all_domain/'+modelname+'/event_division'+'/GCN'+str+'_2bDANNALL_CH_checkpoint.pth')
        
        self.val_loss_min = val_loss
        
