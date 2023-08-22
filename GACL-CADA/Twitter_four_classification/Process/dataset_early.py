import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
#from transformers import *
import json
from torch.utils.data import DataLoader


# global
label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }
def random_pick(list, probabilities): 
    x = random.uniform(0,1)
    cumulative_probability = 0.0 
    for item, item_probability in zip(list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])#torch.LongTensor(batch_label).to(device1)
        #item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GraphDataset(Dataset):
    def __init__(self, fold_x,datasetname, droprate): 
        
        self.fold_x = fold_x
        self.droprate = droprate
        self.datasetname=datasetname

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index]
        datasetname=self.datasetname
    
        
        if datasetname=='Twitter15':
            with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
                tweets = pickle.load(t)
            with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
                inf = pickle.load(f)
            with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:      
                json_inf0 = json.load(j_f0)
            with open('./bert_w2c/T15/t15_mask_015/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
            with open('./data/label_15.json', 'r') as j_tags:
                tags = json.load(j_tags)
        elif datasetname=='Twitter16':
            with open('./data/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
                tweets = pickle.load(t)
            with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
                inf = pickle.load(f)
            with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
                json_inf0 = json.load(j_f0)
            with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
            with open('./data/label_16.json', 'r') as j_tags:
                tags = json.load(j_tags)
                
        # ====================================edgeindex========================================
        
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        
        
        init_row = list(edgeindex[0]) 
        init_col = list(edgeindex[1]) 
        burow = list(edgeindex[1]) 
        bucol = list(edgeindex[0]) 
        row = init_row + burow 
        col = init_col + bucol
      
        new_edgeindex = [row, col]

        #==================================- dropping + adding + misplacing -===================================#

        choose_list = [1,2,3] # 1-drop 2-add 3-misplace
        if datasetname=='Twitter15':
            probabilities = [0.5,0.3,0.2] 
        elif datasetname=='Twitter16':
            probabilities = [0.7,0.2,0.1] # T15: probabilities = [0.5,0.3,0.2] 
        
        choose_num = random_pick(choose_list, probabilities)

        if self.droprate > 0:
            if choose_num == 1:
            
                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                poslist = sorted(poslist)
                row2 = list(np.array(row)[poslist])
                col2 = list(np.array(col)[poslist])
                new_edgeindex2 = [row2, col2]
                #new_edgeindex = [row2, col2]
                '''
                length = len(list(set(sorted(row))))
                print('length:', length)
                poslist = random.sample(range(1,length), int(length * self.prerate))
                print('len of poslist: ', len(poslist))
                new_row = []
                new_col = []
                #print('row:',row)
                #print('poslist', poslist)
                for i_r, e_r in enumerate(row):
                    for i_c, e_c in enumerate(col):
                        if i_r == i_c:
                            if e_r not in poslist and e_c not in poslist:
                                new_row.append(e_r)
                                new_col.append(e_c)
                                #print('new_row:', new_row)
                                #print('new_col:', new_col)
                    
                print('len of new_row:', len(new_row))
                if len(new_row) != len(new_col):
                    print('setting error')
                Dict = {}
                for index, tweet in enumerate(sorted(list(set(new_row+new_col)))):
                    Dict[tweet] = index
                
                row2 = []
                col2 = []
                for i_nr in new_row:
                    row2.append(Dict[i_nr])
                for i_nc in new_col:
                    col2.append(Dict[i_nc])
                #print('row2:',row2)
                '''
                
                
            elif choose_num == 2:
                '''
                length = len(row)
                last_num = list(set(sorted(row)))[-1]
                add_list = list(range(last_num+1, int(length * self.prerate)))
                add_row = []
                add_col = []
                for add_item in add_list:
                    add_row.append(add_item)
                    add_col.append(random.randint(0, add_item-1))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row
                '''
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate)) 
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                new_edgeindex2 = [row2, col2]


                         
            elif choose_num == 3: 
                length = len(init_row)
                mis_index_list = random.sample(range(length), int(length * self.droprate))
                #print('mis_index_list:', mis_index_list)
                Sort_len = len(list(set(sorted(row))))
                if Sort_len > int(length * self.droprate):
                    mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                    #print('mis_valu_list:', mis_value_list)
                    #val_i = 0
                    for i, item in enumerate(init_row):
                        for mis_i,mis_item in enumerate(mis_index_list):
                            if i == mis_item and mis_value_list[mis_i] != item:
                                init_row[i] = mis_value_list[mis_i]
                    row2 = init_row + init_col
                    col2 = init_col + init_row
                    new_edgeindex2 = [row2, col2]


                else:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
        else:
             new_edgeindex = [row, col]
             new_edgeindex2 = [row, col]
        
        
        
        # =========================================X===============================================
        
        
        x0 = json_inf0[id]
        x0 = np.array(x0)
        
        
        x_list = json_inf[id]
        x = np.array(x_list)
        

        
            
        #twitter
        y = label2id[tags[id]]
                
        #y = np.array(y)
        if self.droprate > 0:
            if choose_num == 1:
                zero_list = [0]*768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list
                
                x2 = np.array(x_list)
                x = x2

        # print(x0)
        # print(new_edgeindex)
        # print(x0.shape, np.array(new_edgeindex).shape)
        return Data(x0=torch.tensor(x0,dtype=torch.float32),
                    x=torch.tensor(x,dtype=torch.float32), 
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y]))


class GraphDataset_100(Dataset):
    def __init__(self, fold_x, datasetname,method, droprate):

        self.fold_x = fold_x
        self.droprate = droprate
        self.datasetname = datasetname
        self.method = method

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        datasetname = self.datasetname
        method = self.method
        if datasetname == 'Twitter15':
            with open('./data/twitter15/' + id + '/early_tweets.pkl', 'rb') as t:
                tweets = pickle.load(t)
            with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
                tweets1 = pickle.load(t)
            with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
                inf = pickle.load(f)
            with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:
                json_inf0 = json.load(j_f0)
            with open('./bert_w2c/T15/t15_mask_015/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
            with open('./data/label_15.json', 'r') as j_tags:
                tags = json.load(j_tags)
        elif datasetname == 'Twitter16':
            with open('./data/twitter16/' + id + '/early_tweets.pkl', 'rb') as t:
                tweets = pickle.load(t)
            with open('./data/twitter16/' + id + '/after_tweets.pkl', 'rb') as t:
                tweets1 = pickle.load(t)
            with open('./data/twitter16/' + id + '/after_structure.pkl', 'rb') as f:
                inf = pickle.load(f)
            with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
                json_inf0 = json.load(j_f0)
            with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
            with open('./data/label_16.json', 'r') as j_tags:
                tags = json.load(j_tags)

        # ====================================edgeindex========================================

        if method == 'a':
            # original

            # ====================================edgeindex==============================================
            tweets = tweets1[0:1]
            new_edgeindex = [[0], [0]]
            new_edgeindex2 = [[0], [0]]

            # =========================================X=========================================================

            x = json_inf[id]
            x = x[0:1]
            x = np.array(x)

            y = label2id[tags[id]]

        elif method == 'b':
            # 包one-reply

            tweets = tweets[0:2]

            dict = {}
            for index, tweet in enumerate(tweets):
                dict[tweet] = index
            # print('dict: ', dict)

            inf = inf[1:]
            new_inf = []
            for pair in inf:
                new_pair = []
                for E in pair:
                    if E == 'ROOT' or E not in tweets:
                        break
                    E1 = dict[E]
                    new_pair.append(E1)
                if E != 'ROOT' and E in tweets:
                    new_inf.append(new_pair)
            new_inf = np.array(new_inf).T

            edgeindex = new_inf

            init_row = list(edgeindex[0])
            init_col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            row = init_row + burow
            col = init_col + bucol


            new_edgeindex = [row, col]
            choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
            if datasetname == 'Twitter15':
                probabilities = [0.5, 0.3, 0.2]
            elif datasetname == 'Twitter16':
                probabilities = [0.7, 0.2, 0.1]  # T15: probabilities = [0.5,0.3,0.2]
            choose_num = random_pick(choose_list, probabilities)
            # print('new_edgeindex； ', np.array([row, col]).shape)
            if self.droprate > 0:
                if choose_num == 1:

                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]


                elif choose_num == 2:
                    '''
                    length = len(row)
                    last_num = list(set(sorted(row)))[-1]
                    add_list = list(range(last_num+1, int(length * self.prerate)))
                    add_row = []
                    add_col = []
                    for add_item in add_list:
                        add_row.append(add_item)
                        add_col.append(random.randint(0, add_item-1))
                    row2 = row + add_row + add_col
                    col2 = col + add_col + add_row
                    '''
                    length = len(list(set(sorted(row))))
                    add_row = random.sample(range(length), int(length * self.droprate))
                    add_col = random.sample(range(length), int(length * self.droprate))
                    row2 = row + add_row + add_col
                    col2 = col + add_col + add_row

                    new_edgeindex2 = [row2, col2]



                elif choose_num == 3:
                    length = len(init_row)
                    mis_index_list = random.sample(range(length), int(length * self.droprate))
                    # print('mis_index_list:', mis_index_list)
                    Sort_len = len(list(set(sorted(row))))
                    if Sort_len > int(length * self.droprate):
                        mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                        # print('mis_valu_list:', mis_value_list)
                        # val_i = 0
                        for i, item in enumerate(init_row):
                            for mis_i, mis_item in enumerate(mis_index_list):
                                if i == mis_item and mis_value_list[mis_i] != item:
                                    init_row[i] = mis_value_list[mis_i]
                        row2 = init_row + init_col
                        col2 = init_col + init_row
                        new_edgeindex2 = [row2, col2]


                    else:
                        length = len(row)
                        poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                        poslist = sorted(poslist)
                        row2 = list(np.array(row)[poslist])
                        col2 = list(np.array(col)[poslist])
                        new_edgeindex2 = [row2, col2]
            else:
                new_edgeindex = [row, col]
                new_edgeindex2 = [row, col]

            # =========================================X=========================================================

            x_0 = json_inf0[id]
            s_index = tweets1.index(tweets[0])
            f_index = tweets1.index(tweets[1])
            x0 = [x_0[s_index]]
            x0.append(x_0[f_index])
            x0 = np.array(x0)

            x_list = json_inf[id]
            x = [x_list[s_index]]
            x.append(x_list[f_index])
            # x = np.array(x)

            y = label2id[tags[id]]
            if self.droprate > 0:
                if choose_num == 1:
                    zero_list = [0] * 768
                    x_length = len(x)
                    r_list = random.sample(range(x_length), int(x_length * self.droprate))
                    r_list = sorted(r_list)
                    for idex, line in enumerate(x):
                        for r in r_list:
                            if idex == r:
                                x[idex] = zero_list

                    x2 = np.array(x)
                    x = x2
        elif method == 'c':
            # all-reply
            # ====================================edgeindex==============================================

            dict = {}
            for index, tweet in enumerate(tweets1):
                dict[tweet] = index
            # print('dict: ', dict)

            inf = inf[1:]
            new_inf = []
            for pair in inf:
                new_pair = []
                for E in pair:
                    if E == 'ROOT':
                        break
                    E = dict[E]
                    new_pair.append(E)
                if E != 'ROOT':
                    new_inf.append(new_pair)
            new_inf = np.array(new_inf).T
            edgeindex = new_inf

            init_row = list(edgeindex[0])
            init_col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            row = init_row + burow
            col = init_col + bucol

            new_edgeindex = [row, col]

            # ==================================- dropping + adding + misplacing -===================================#

            choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
            if datasetname == 'Twitter15':
                probabilities = [0.5, 0.3, 0.2]
            elif datasetname == 'Twitter16':
                probabilities = [0.7, 0.2, 0.1]  # T15: probabilities = [0.5,0.3,0.2]

            choose_num = random_pick(choose_list, probabilities)

            if self.droprate > 0:
                if choose_num == 1:

                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
                    # new_edgeindex = [row2, col2]
                    '''
                    length = len(list(set(sorted(row))))
                    print('length:', length)
                    poslist = random.sample(range(1,length), int(length * self.prerate))
                    print('len of poslist: ', len(poslist))
                    new_row = []
                    new_col = []
                    #print('row:',row)
                    #print('poslist', poslist)
                    for i_r, e_r in enumerate(row):
                        for i_c, e_c in enumerate(col):
                            if i_r == i_c:
                                if e_r not in poslist and e_c not in poslist:
                                    new_row.append(e_r)
                                    new_col.append(e_c)
                                    #print('new_row:', new_row)
                                    #print('new_col:', new_col)

                    print('len of new_row:', len(new_row))
                    if len(new_row) != len(new_col):
                        print('setting error')
                    Dict = {}
                    for index, tweet in enumerate(sorted(list(set(new_row+new_col)))):
                        Dict[tweet] = index

                    row2 = []
                    col2 = []
                    for i_nr in new_row:
                        row2.append(Dict[i_nr])
                    for i_nc in new_col:
                        col2.append(Dict[i_nc])
                    #print('row2:',row2)
                    '''


                elif choose_num == 2:
                    '''
                    length = len(row)
                    last_num = list(set(sorted(row)))[-1]
                    add_list = list(range(last_num+1, int(length * self.prerate)))
                    add_row = []
                    add_col = []
                    for add_item in add_list:
                        add_row.append(add_item)
                        add_col.append(random.randint(0, add_item-1))
                    row2 = row + add_row + add_col
                    col2 = col + add_col + add_row
                    '''
                    length = len(list(set(sorted(row))))
                    add_row = random.sample(range(length), int(length * self.droprate))
                    add_col = random.sample(range(length), int(length * self.droprate))
                    row2 = row + add_row + add_col
                    col2 = col + add_col + add_row

                    new_edgeindex2 = [row2, col2]



                elif choose_num == 3:
                    length = len(init_row)
                    mis_index_list = random.sample(range(length), int(length * self.droprate))
                    # print('mis_index_list:', mis_index_list)
                    Sort_len = len(list(set(sorted(row))))
                    if Sort_len > int(length * self.droprate):
                        mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                        # print('mis_valu_list:', mis_value_list)
                        # val_i = 0
                        for i, item in enumerate(init_row):
                            for mis_i, mis_item in enumerate(mis_index_list):
                                if i == mis_item and mis_value_list[mis_i] != item:
                                    init_row[i] = mis_value_list[mis_i]
                        row2 = init_row + init_col
                        col2 = init_col + init_row
                        new_edgeindex2 = [row2, col2]


                    else:
                        length = len(row)
                        poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                        poslist = sorted(poslist)
                        row2 = list(np.array(row)[poslist])
                        col2 = list(np.array(col)[poslist])
                        new_edgeindex2 = [row2, col2]
            else:
                new_edgeindex = [row, col]
                new_edgeindex2 = [row, col]

            # =========================================X===============================================

            x0 = json_inf0[id]
            x0 = np.array(x0)

            x_list = json_inf[id]
            x = np.array(x_list)

            # twitter
            y = label2id[tags[id]]

            # y = np.array(y)
            if self.droprate > 0:
                if choose_num == 1:
                    zero_list = [0] * 768
                    x_length = len(x_list)
                    r_list = random.sample(range(x_length), int(x_length * self.droprate))
                    r_list = sorted(r_list)
                    for idex, line in enumerate(x_list):
                        for r in r_list:
                            if idex == r:
                                x_list[idex] = zero_list

                    x2 = np.array(x_list)
                    x = x2

        return Data(x0=torch.tensor(x0, dtype=torch.float32),
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y]))


class test_GraphDataset(Dataset):
    def __init__(self, fold_x,datasetname,method, droprate): 
        
        self.fold_x = fold_x
        self.droprate = droprate
        self.datasetname=datasetname
        self.method=method

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        datasetname=self.datasetname
        method=self.method
        if method=='a':

            # print("aa")
            if datasetname=='Twitter15':
                with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:      
                    json_inf = json.load(j_f0)
                
                with open('./data/label_15.json', 'r') as j_tags:
                    tags = json.load(j_tags)
            elif datasetname=='Twitter16':
                with open('./data/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
                    json_inf = json.load(j_f0)
                
                with open('./data/label_16.json', 'r') as j_tags:
                    tags = json.load(j_tags)
           
            #print(tweets)
            # ====================================edgeindex==============================================
            tweets=tweets[0:1]
            new_edgeindex = [[0],[0]] 
            new_edgeindex2 = [[0],[0]]
            # new_edgeindex = [[],[]] #结果一样
            # new_edgeindex2 = [[],[]]
            # =========================================X=========================================================
      
            x = json_inf[id]
            x=x[0:1]
            x = np.array(x) 
            
            

            y = label2id[tags[id]]
        
        elif method=='b':

     
            if datasetname=='Twitter15':
                with open('./data/twitter15/' + id + '/early_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
                    tweets1 = pickle.load(t)
                with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:      
                    json_inf = json.load(j_f0)
                
                with open('./data/label_15.json', 'r') as j_tags:
                    tags = json.load(j_tags)
            elif datasetname=='Twitter16':
                with open('./data/twitter16/'+ id + '/early_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter16/' + id + '/after_tweets.pkl', 'rb') as t:
                    tweets1 = pickle.load(t)
                with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
                    json_inf = json.load(j_f0)
                
                with open('./data/label_16.json', 'r') as j_tags:
                    tags = json.load(j_tags)
           
            # ====================================edgeindex==============================================
            
            tweets=tweets[0:2]

            dict = {}
            for index, tweet in enumerate(tweets):
                dict[tweet] = index
            # print('dict: ', dict)

            inf = inf[1:]

            # print(inf)
            # id to num
            new_inf = []
            for pair in inf:
                new_pair = []
                for E in pair:
                    if E == 'ROOT' or E not in tweets:
                        break
                    E1 = dict[E]
                    new_pair.append(E1)
                if E != 'ROOT' and E in tweets:
                    new_inf.append(new_pair)
            new_inf = np.array(new_inf).T


            edgeindex = new_inf
            # print('edgeindex: ', edgeindex.shape)
            # print(id)

            row = list(edgeindex[0])
            col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            row.extend(burow)
            col.extend(bucol)
            # print('new_edgeindex； ', np.array([row, col]).shape)

            if self.droprate > 0:
                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate)))  #
                poslist = sorted(poslist)
                row1 = list(np.array(row)[poslist])
                col1 = list(np.array(col)[poslist])

                poslist2 = random.sample(range(length), int(length * (1 - self.droprate)))  #
                poslist2 = sorted(poslist2)
                row2 = list(np.array(row)[poslist2])
                col2 = list(np.array(col)[poslist2])

                new_edgeindex = [row1, col1]
                new_edgeindex2 = [row2, col2]
            else:
                new_edgeindex = [row, col]
                new_edgeindex2 = [row, col]

            # =========================================X=========================================================      
            x = json_inf[id]
            s_index=tweets1.index(tweets[0])
            f_index=tweets1.index(tweets[1])
            x1=[x[s_index]]
            # print(np.array(x1).shape)
            # print(x)
            x1.append(x[f_index])
            
            # print(np.array(x1).shape)
            # x = json_inf[id]
            # x=x[0:2]
            x = np.array(x1)  
            # print(x)
            y = label2id[tags[id]]
        elif method=='c':
            if datasetname=='Twitter15':
                with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:      
                    json_inf = json.load(j_f0)
                
                with open('./data/label_15.json', 'r') as j_tags:
                    tags = json.load(j_tags)
            elif datasetname=='Twitter16':
                with open('./data/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
                    tweets = pickle.load(t)
                with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
                    inf = pickle.load(f)
                with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
                    json_inf = json.load(j_f0)                
                with open('./data/label_16.json', 'r') as j_tags:
                    tags = json.load(j_tags)
            # ====================================edgeindex==============================================
    

        
            dict = {}
            for index, tweet in enumerate(tweets):
                dict[tweet] = index
            #print('dict: ', dict)


            inf = inf[1:]

            #print(inf)
            # id to num
            new_inf = []
            for pair in inf:
                new_pair = []
                for E in pair:
                    if E == 'ROOT':
                        break
                    E = dict[E]
                    new_pair.append(E)
                if E != 'ROOT':
                    new_inf.append(new_pair)
            new_inf = np.array(new_inf).T
      
            edgeindex = new_inf

       
        
            row = list(edgeindex[0])
            col = list(edgeindex[1]) 
            burow = list(edgeindex[1]) 
            bucol = list(edgeindex[0]) 
            row.extend(burow) 
            col.extend(bucol) 
            #print('new_edgeindex； ', np.array([row, col]).shape)

            if self.droprate > 0: 
                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate))) # 
                poslist = sorted(poslist)
                row1 = list(np.array(row)[poslist])
                col1 = list(np.array(col)[poslist])

                poslist2 = random.sample(range(length), int(length * (1 - self.droprate))) # 
                poslist2 = sorted(poslist2)
                row2 = list(np.array(row)[poslist2])
                col2 = list(np.array(col)[poslist2])

                new_edgeindex = [row1, col1] 
                new_edgeindex2 = [row2, col2]
            else:
                new_edgeindex = [row, col] 
                new_edgeindex2 = [row, col]
                
            # print(np.array(new_edgeindex).shape)
            # =========================================X=========================================================     
            x = json_inf[id]
            # x=x[0:len(x)-1]
            x = np.array(x) 
            # print(x.shape)

            y = label2id[tags[id]]     
          


        return Data(x0=torch.tensor(x,dtype=torch.float32),
                    x=torch.tensor(x,dtype=torch.float32), 
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2), 
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y])) 
