import random

import numpy as np
import argparse
import time, os
import math

import word_embedding_twitter_data as process_data

import copy
import pickle
from random import sample

from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import  StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# import torch.nn.utils.rnn.pack_padded_sequence
from torch.nn import utils as nn_utils
from sklearn import manifold
from sklearn import metrics

import warnings

from evaluate import evaluationclass

warnings.filterwarnings("ignore")

import datetime

from rand5fold_pheme_early_domain import *


class Rumor_data(Dataset): 
    def __init__(self, dataset):
        self.text = torch.tensor(list(dataset['post_text'])) 
        self.mask = torch.tensor(list(dataset['mask'])) 
        self.label = torch.tensor(list(dataset['label']))   
        self.event_label = torch.tensor(list(dataset['event_label']))   
        print('Text: %d, label: %d, Event: %d'
              % (len(self.text), len(self.label), len(self.event_label)))


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return(self.text[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lamdb = args.lambd
        return x.view_as(x)  
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lamdb)   


def grad_reverse(x):
    return ReverseLayerF().apply(x)


class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()  
        self.args = args

        self.event_num = args.event_num    

        vocab_size = args.vocab_size   
        emb_dim = args.embed_dim   

        self.hidden_size = args.hidden_dim 
        self.lstm_size = args.embed_dim  
        self.social_size = 19 

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))

        # self.lstm = nn.LSTM(self.lstm_size, self.hidden_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)


        # Text CNN
        channel_in = 1   
        filter_num = 20   
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size]) 
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)  

        self.dropout = nn.Dropout(args.dropout)


        # false information classifier
        self.class_classifier = nn.Sequential()  
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))   
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))  

        # event/domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2',nn.Linear(self.hidden_size, self.event_num)) 
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1)) 


        # weight measurer，
        self.weight_measurer = nn.Sequential()
        self.weight_measurer.add_module('w_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.weight_measurer.add_module('w_relu1', nn.LeakyReLU(True))
        self.weight_measurer.add_module('w_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.weight_measurer.add_module('w_softmax', nn.Softmax(dim=1))


    def init_hidden(self, batch_size):
      
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),   # to_var
                to_var(torch.zeros(1, batch_size, self.lstm_size)))   # to_var


    def conv_and_pool(self, x, conv):
      
        x = F.relu(conv(x)).squeeze(3)  # (sample number, hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2) 
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  
        return x


    def forward(self, text, mask, mmd_loss):
        # CNN
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text) 
        text = text.unsqueeze(1)  
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs] 
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]

        text = torch.cat(text, 1) 
        text = self.dropout(F.relu(self.fc1(text)))
        # text = F.relu(self.fc1(text))

        # Class
        class_output = self.class_classifier(text)
        # Domain/Event
        reverse_feature = grad_reverse(text)
        domain_output = self.domain_classifier(reverse_feature)  
        # weight
        if mmd_loss >= args.mmd_threshold:
            weight_output = self.weight_measurer(text)
        else:
            weight_output = torch.zeros_like(domain_output)
        return class_output, domain_output, weight_output 


def to_var(x): 
    if torch.cuda.is_available():

        x = x.cuda()   
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()   




def main(args,seed):
    print('loading data')
    # print(args)
    source_train, target_train, validation, test, W, mmd_loss = load_data(args) 
    # print(test['post_text'])  
    #
    # test_list = list(test['post_text'])
    # print(len(test_list))
    # # print(train_list[0])
    # print(test_list[0])

    # [post_id, original_post, post_text, label, type, event_label, mask]
    # print(source_train)
    source_train_dataset = Rumor_data(source_train)
    target_train_dataset = Rumor_data(target_train)
    validation_dataset = Rumor_data(validation)
    test_dataset = Rumor_data(test)


    # Data Loader (Input Pipeline)
    source_train_loader = DataLoader(dataset=source_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    target_train_loader = DataLoader(dataset=target_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print("MMD distance between source and target is: ", mmd_loss)

    print('building model')
    model = CNN_Fusion(args, W)
    print('model settled')

    if torch.cuda.is_available():
        print("CUDA available")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 

    if mmd_loss < args.mmd_threshold:
        model.weight_measurer.w_fc1.weight.requires_grad = False
        model.weight_measurer.w_fc1.bias.requires_grad = False

        model.weight_measurer.w_fc2.weight.requires_grad = False
        model.weight_measurer.w_fc2.bias.requires_grad = False


    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 # lr=args.learning_rate)

    scheduler = StepLR(optimizer, step_size= 10, gamma= 1) 

    # iter_per_epoch = len(train_loader)  
    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    print("loader size: " + str(len_dataloader))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''

    print('training model')
    adversarial = True

    # Train the Model
    for epoch in range(args.num_epochs): 
        p = float(epoch) / 100
        lr = 0.001  
        optimizer.lr = lr  

        start_time = time.time()  
        cost_vector = [] 
        class_cost_vector = [] 
        domain_cost_vector = [] 
        weight_cost_vector = [] 

        acc_vector = []  
        valid_acc_vector = []  
        # test_acc_vector = []

        vali_cost_vector = [] 
        # test_cost_vector = []
        source_train_loader_iter = iter(source_train_loader)
        target_train_loader_iter = iter(target_train_loader)
        validation_loader_iter = iter(validation_loader)

        i = 0
        while i < len_dataloader:

         
            data_source = next(source_train_loader_iter)

            source_data, source_labels, source_event_labels = data_source

            source_text = to_var(source_data[0])
            source_mask = to_var(source_data[1])
            source_labels = to_var(source_labels)
            source_event_labels = to_var(source_event_labels)

            # print(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  #

            class_outputs, source_domain_outputs, weight_outputs = model(source_text, source_mask, mmd_loss)

            # print(weight_outputs)

            if mmd_loss >= args.mmd_threshold:
                weight = [x[0] for x in weight_outputs]
                class_loss = weighted_CrossEntropyLoss_detect(source_labels, class_outputs, weight)
                class_loss *= 10
                source_domain_loss = weighted_CrossEntropyLoss_domain(source_event_labels, source_domain_outputs, weight)

                weight_loss = criterion(weight_outputs, source_event_labels)

            else:
                class_loss = criterion(class_outputs, source_labels)
                source_domain_loss = criterion(source_domain_outputs, source_event_labels)
                weight_loss = 0

        
            data_target = next(target_train_loader_iter)

            target_data, _, target_event_labels = data_target

            target_text = to_var(target_data[0])
            target_mask = to_var(target_data[1])
            target_event_labels = to_var(target_event_labels)

            _, target_domain_outputs, _ = model(target_text, target_mask, mmd_loss)

            target_domain_loss = criterion(target_domain_outputs, target_event_labels)

            domain_loss = source_domain_loss + target_domain_loss

            loss = class_loss + domain_loss + weight_loss 

            loss.backward() 
            optimizer.step()

            _, argmax = torch.max(class_outputs, 1) 

            cross_entropy = True

           
            if True:
                accuracy = (source_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(source_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item()) 
            with torch.no_grad():
                model.eval() 
                valid_acc_vector_temp = []  

                try:
                    data_validation = next(validation_loader_iter)
                except StopIteration:
                    validation_loader_iter = iter(validation_loader)
                    data_validation = next(validation_loader_iter)

                validate_data, validate_labels, _ = data_validation
                validate_text = to_var(validate_data[0])
                validate_mask = to_var(validate_data[1])
                validate_labels = to_var(validate_labels)

                validate_outputs, _, _ = model(validate_text, validate_mask, mmd_loss)
                _, validate_argmax = torch.max(validate_outputs, 1)  
                vali_loss = criterion(validate_outputs, validate_labels)

                validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
                vali_cost_vector.append(vali_loss.item())

                valid_acc_vector_temp.append(validate_accuracy.item())

                validate_acc = np.mean(valid_acc_vector_temp)
                valid_acc_vector.append(validate_acc)

            model.train() 
            print('Epoch [%d/%d], Loss: %.4f, Class Loss: %.4f, Validate loss: %.4f,'
                  ' Train_Acc: %.4f, Validate_Acc: %.4f'
                  % (epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                     np.mean(vali_cost_vector), np.mean(acc_vector), validate_acc))


            

            i += 1
    if not os.path.exists(args.output_file+args.data_division +'/'):  
        os.mkdir(args.output_file+args.data_division +'/')

    best_validate_dir =args.output_file+args.data_division +'/seed'+seed+ '_text.pkl'
    print("the best validate model is: " + str(best_validate_dir))
    torch.save(model.state_dict(), best_validate_dir)

    duration = time.time() - start_time
    print('Epoch: %d, Duration: %.4f ' % (epoch + 1, duration))


    # Test the Model   
    print('testing model')
    model = CNN_Fusion(args, W)
    model.load_state_dict(torch.load(best_validate_dir))
    print("the selected model is: " + str(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        model.eval()  
        test_score = []  
        test_pred = [] 
        test_true = []   

        for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
            test_text = to_var(test_data[0])  # to_var
            test_mask = to_var(test_data[1])  # to_var
            test_labels = to_var(test_labels)  # to_var
            test_outputs, _, _= model(test_text, test_mask, mmd_loss)   
            _, test_argmax = torch.max(test_outputs, 1)
            # print(test_argmax)  
            if i == 0:
                test_score = to_np(test_outputs.squeeze())  
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())

            else:
                test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0) 
                test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
            test_pred, test_true)
        test_accuracy = metrics.accuracy_score(test_true, test_pred)   


        test_score_convert = [x[1] for x in test_score] 
        test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro'

        test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)  

        print("Classification Acc: %.4f, AUC-ROC: %.4f"
              % (test_accuracy, test_aucroc))
        print("Classification report: \n%s\n"
              % (metrics.classification_report(test_true, test_pred, digits=4)))



        print("Classification confusion matrix: \n%s\n"
              % (test_confusion_matrix))


        print("Saving results")
        return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2

def main_test(args,seed,best_validate_dir):
    print('loading data')
    print(args)
    source_train, target_train, validation, test, W, mmd_loss = load_data(args) 

    test_dataset = Rumor_data(test)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print("MMD distance between source and target is: ", mmd_loss)

    print('building model')
    model = CNN_Fusion(args, W)
    print('model settled')

    if torch.cuda.is_available():
        print("CUDA available")
        model.cuda()
    model.load_state_dict(torch.load(best_validate_dir))  
    print("the selected model is: " + str(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        model.eval()  
        test_score = []  
        test_pred = []  
        test_true = []  

        for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
            test_text = to_var(test_data[0])  # to_var
            test_mask = to_var(test_data[1])  # to_var
            test_labels = to_var(test_labels)  # to_var
            test_outputs, _, _ = model(test_text, test_mask, mmd_loss) 
            _, test_argmax = torch.max(test_outputs, 1)
            # print(test_argmax)  # 580*1的tensor
            if i == 0:
                test_score = to_np(test_outputs.squeeze()) 
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())

            else:
                test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0) 
                test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
            test_pred, test_true)
        test_accuracy = metrics.accuracy_score(test_true, test_pred) 

        test_score_convert = [x[1] for x in test_score] 
        test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro') 

        test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)  

        print("Classification Acc: %.4f, AUC-ROC: %.4f"
              % (test_accuracy, test_aucroc))
        print("Classification report: \n%s\n"
              % (metrics.classification_report(test_true, test_pred, digits=4)))


        print("Classification confusion matrix: \n%s\n"
              % (test_confusion_matrix))

        print("Saving results")
        return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2

def parse_arguments(parser):  
    parser = argparse.ArgumentParser()
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=50, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='')
    parser.add_argument('--event_num', type=int, default=2, help='')
    parser.add_argument('--mmd_threshold', type=float, default=1.0, help='')
    parser.add_argument('--datasetname', type=str, default='Pheme', help='')
    parser.add_argument('--data_division', type=str, default='event', help='')
    parser.add_argument('--datalist', type=list, default=[], help='')
    return parser


def word2vec(post, word_id_map, W): 

    word_embedding = []
    mask = []

    for sentence in post:
        sen_embedding = []
        sentence = sentence.split()  
        # seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])  

        while len(sen_embedding) < args.sequence_len:  
            sen_embedding.append(0)

 

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq)) 

    return word_embedding, mask


def load_data(args):
    source_train, target_train, validate, test, MMD_loss = process_data.get_data(args)

   
    source_train_label_list = []
    target_train_label_list = []
    validate_label_list = []
    test_label_list = []

    for i in source_train['label']:
        source_train_label_list.append(int(i))
    for j in target_train['label']:
        target_train_label_list.append(int(j))
    for k in validate['label']:
        validate_label_list.append(int(k))
    for l in test['label']:
        test_label_list.append(int(l))

    source_train['label'] = source_train_label_list
    target_train['label'] = target_train_label_list
    validate['label'] = validate_label_list
    test['label'] = test_label_list
    word_vector_path = "../data/"+args.datasetname+"_"+args.data_division+"_word_embedding.pickle"
    
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)   # W, W2, word_idx_map, vocab，max_sentence_length
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)     
    args.sequence_len = max_len     

    print("translate source train data to embedding")
    word_embedding, mask = word2vec(source_train['post_text'], word_idx_map, W)
    source_train['post_text'] = word_embedding
    source_train['mask'] = mask

    print("translate target train data to embedding")
    word_embedding, mask = word2vec(target_train['post_text'], word_idx_map, W)
    target_train['post_text'] = word_embedding
    target_train['mask'] = mask

    print("translate validation data to embedding")
    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, W)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, W)
    test['post_text'] = word_embedding
    test['mask'] = mask

    print("sequence length " + str(args.sequence_length))

    train_data_size = len(source_train['post_text']) + len(target_train['post_text'])
    print("train data size is " + str(train_data_size))
    print("Finished loading data")
    return source_train, target_train, validate, test, W, MMD_loss


def weighted_CrossEntropyLoss_domain(target, input, weight):
    loss = .0
    num = len(target)
    for label, prob, sim in zip(target, input, weight):
        if label == 0:
            loss -= prob[0] - math.log((math.exp(prob[0]) + math.exp(prob[1])))
            loss *= sim
        else:
            loss -= prob[1] - math.log((math.exp(prob[0]) + math.exp(prob[1])))
    loss = loss / num
    return loss


def weighted_CrossEntropyLoss_detect(target, input, weight):
    loss = .0
    num = len(target)
    for label, prob, sim in zip(target, input, weight):
        if label == 0:
            loss -= prob[0] - math.log((math.exp(prob[0]) + math.exp(prob[1])))
        else:
            loss -= prob[1] - math.log((math.exp(prob[0]) + math.exp(prob[1])))
        loss *= sim
    loss = loss / num
    return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    iterations=1
    parse = argparse.ArgumentParser()
    parse = parse_arguments(parse)
    train = ''
    test = ''
    output ='../data/output/Pheme/'
    args = parse.parse_args([train, test, output])
    
    # args.data_division='event'

    # type = 'train'
    args.data_division=sys.argv[2]
    type=sys.argv[1]


    if type == 'train' and args.data_division=='event':
        print('TRAIN')
        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

        for iters in range(iterations):
            setup_seed(20)
            print('seed=20, t=0.8')
            fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataP(args.datasetname, 3)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
            fold0_x_train_8.extend(fold0_x_train_2)
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
            accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = main(args,'20')

            setup_seed(30)
            print('seed=30, t=0.8')
            fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataP(args.datasetname, 3)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
            fold0_x_train_8.extend(fold0_x_train_2)
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
            accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = main(args,'30')

            setup_seed(40)
            print('seed=40, t=0.8')
            fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataP(args.datasetname, 3)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
            fold0_x_train_8.extend(fold0_x_train_2)
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
            accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = main(args,'40')

            setup_seed(50)
            print('seed=50, t=0.8')
            fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataP(args.datasetname, 3)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
            fold0_x_train_8.extend(fold0_x_train_2)
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
            accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = main(args,'50')

            setup_seed(60)
            print('seed=60, t=0.8')
            fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataP(args.datasetname, 3)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
            fold0_x_train_8.extend(fold0_x_train_2)
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
            accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = main(args,'60')

            test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
            ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
            ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
            PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
            PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
            REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
            REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
            F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
            F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
              "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations,
                                                                        sum(ACC1) / iterations,
                                                                        sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                        sum(PRE2) / iterations,
                                                                        sum(REC1) / iterations, sum(REC2) / iterations,
                                                                        sum(F1) / iterations, sum(F2) / iterations))
    elif type == 'train' and args.data_division=='random':
        print('TRAIN')
        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

        for iters in range(iterations):
            setup_seed(20)
            print('seed=20, t=0.8')
            fold0_test,fold0_val,fold0_train,\
            fold1_test,fold1_val,fold1_train,\
            fold2_test,fold2_val,fold2_train,\
            fold3_test,fold3_val,fold3_train,\
            fold4_test, fold4_val,fold4_train= load5foldDataP(args.datasetname, 4)
    
        
            print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
            print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold1_val))
            print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold2_val))
            print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold3_val))
            print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold4_val))
            args.datalist = [args.datasetname, fold0_test,fold0_train,fold0_val,fold0_val]
            accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = main(args,'20')

            args.datalist = [args.datasetname, fold1_test,fold1_train,fold1_val,fold1_val]
            accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = main(args,)

            args.datalist = [args.datasetname, fold2_test,fold2_train,fold2_val,fold2_val]
            accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = main(args,'40')

            args.datalist = [args.datasetname, fold3_test,fold3_train,fold3_val,fold3_val]
            accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = main(args,'50')

            args.datalist = [args.datasetname, fold4_test,fold4_train,fold4_val,fold4_val]
            accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = main(args,'60')
            test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
            ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
            ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
            PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
            PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
            REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
            REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
            F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
            F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
              "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations,
                                                                        sum(ACC1) / iterations,
                                                                        sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                        sum(PRE2) / iterations,
                                                                        sum(REC1) / iterations, sum(REC2) / iterations,
                                                                        sum(F1) / iterations, sum(F2) / iterations))

    elif type == 'train' and args.data_division=='time':
        print('TRAIN')
        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

        for iters in range(iterations):
            setup_seed(20)
            print('seed=20, t=0.8')
            fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(args.datasetname, 1)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]
            accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = main(args,'20')

            setup_seed(30)
            print('seed=30, t=0.8')
            fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(args.datasetname, 1)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]
            accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = main(args,'30')

            setup_seed(40)
            print('seed=40, t=0.8')
            fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(args.datasetname, 1)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]
            accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = main(args,'40')

            setup_seed(50)
            print('seed=50, t=0.8')
            fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(args.datasetname, 1)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]
            accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = main(args,'50')

            setup_seed(60)
            print('seed=60, t=0.8')
            fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(args.datasetname, 1)
            print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
            args.datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]
            accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = main(args,'60')

            test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
            ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
            ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
            PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
            PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
            REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
            REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
            F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
            F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
              "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations,
                                                                        sum(ACC1) / iterations,
                                                                        sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                        sum(PRE2) / iterations,
                                                                        sum(REC1) / iterations, sum(REC2) / iterations,
                                                                        sum(F1) / iterations, sum(F2) / iterations))
    elif type=='test':
        print('TEST')
        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

        for iter in range(iterations):
            setup_seed(20)
            print('seed=20, t=0.8')
            model_path='../data/output/'+args.datasetname+'87_seed20_text.pkl'
            accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = main_test(args, '20',model_path)
            setup_seed(30)
            print('seed=30, t=0.8')
            accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = main_test(args, '30',model_path)
            setup_seed(40)
            print('seed=40, t=0.8')
            accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = main_test(args, '40',model_path)
            setup_seed(50)
            print('seed=50, t=0.8')
            accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = main_test(args, '50',model_path)
            setup_seed(60)
            print('seed=60, t=0.8')
            accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = main_test(args, '60',model_path)

            test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
            ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
            ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
            PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
            PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
            REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
            REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
            F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
            F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
              "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations,
                                                                        sum(ACC1) / iterations,
                                                                        sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                        sum(PRE2) / iterations,
                                                                        sum(REC1) / iterations, sum(REC2) / iterations,
                                                                        sum(F1) / iterations, sum(F2) / iterations))

    # if type == 'train':
    #     print('TRAIN')
    #     test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

    #     for iters in range(iterations):
    #         setup_seed(20)
    #         print('seed=20, t=0.8')

    #         accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = main(args, '20')
    #         setup_seed(30)
    #         print('seed=30, t=0.8')
    #         accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = main(args,'30' )
    #         setup_seed(40)
    #         print('seed=40, t=0.8')
    #         accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = main(args, '40')
    #         setup_seed(50)
    #         print('seed=50, t=0.8')
    #         accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = main(args, '50')
    #         setup_seed(60)
    #         print('seed=60, t=0.8')
    #         accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = main(args, '60')

    #         test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
    #         ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    #         ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
    #         PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    #         PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    #         REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    #         REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    #         F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    #         F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

    #     print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
    #           "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations,
    #                                                                     sum(ACC1) / iterations,
    #                                                                     sum(ACC2) / iterations, sum(PRE1) / iterations,
    #                                                                     sum(PRE2) / iterations,
    #                                                                     sum(REC1) / iterations, sum(REC2) / iterations,
    #                                                                     sum(F1) / iterations, sum(F2) / iterations))


























