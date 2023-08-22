import json
import random
from random import shuffle
import os

from datetime import datetime
# from data_pprocess.data_pkl import event_class
import pandas as pd
import pickle

def get_label(obj):
    if obj == 'Twitter15':
        label_path = '../data/Twitter15_label_All.txt'
    elif obj == 'Twitter16':
        label_path = '../data/Twitter16_label_All.txt'
    label_path = label_path
    if 'Twitter' in obj:
        labelPath = os.path.join(label_path)
        labelDic = {'unverified': {},'non-rumor': {},'true': {},'false': {}}  # {'615689290706595840': 'true',...}

        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[
                2]  # eid '656955120626880512' label 'false'

   
            if event in labelDic[label]:
                labelDic[label][event].append(eid)
            else:
                labelDic[label][event]=[eid]
        count=0
        for x in labelDic:
            for y in labelDic[x]:
                
                count+=len(labelDic[x][y])
        print(count)
        return labelDic

def load5foldDataT(obj,data_path,label_path,method):
    path = data_path
    label_path = label_path
    # eventallDic,df=event_class(obj)
    if 'Twitter' in obj and method==0:
        labelPath = os.path.join(label_path)
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['non-rumor'], ['false'], ['true'], ['unverified']
        t_path = path
        file_list = os.listdir(t_path)
        print('The len of file_list: ', len(file_list))

        NR,F,T,U = [],[],[],[]
        l1=l2=l3=l4=0
        labelDic = {} 
        for line in open(labelPath): 
            line = line.rstrip() 
            label, eid = line.split('\t')[0], line.split('\t')[2] 

            if eid in file_list:
                labelDic[eid] = label.lower() 
                    
                if label in labelset_nonR: 
                    NR.append(eid)
                    l1 += 1
                if labelDic[eid] in labelset_f: # F
                    F.append(eid)
                    l2 += 1
                if labelDic[eid] in labelset_t: # T
                    T.append(eid)
                    l3 += 1
                if labelDic[eid] in labelset_u: # U
                    U.append(eid)
                    l4 += 1
        print(len(labelDic)) 
        print(l1,l2,l3,l4) 
        random.shuffle(NR) 
        random.shuffle(F) 
        random.shuffle(T) 
        random.shuffle(U) 

        
        fold0_x_test,fold1_x_test,fold2_x_test,fold3_x_test,fold4_x_test=[],[],[],[],[] 
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2) 
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        
        fold0_x_test.extend(NR[0:leng1]) 
        fold0_x_test.extend(F[0:leng2]) 
        fold0_x_test.extend(T[0:leng3]) 
        fold0_x_test.extend(U[0:leng4]) 
        fold0_x_train.extend(NR[leng1:]) 
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])

        fold1_x_train.extend(NR[0:leng1]) 
        fold1_x_train.extend(NR[leng1 * 2:]) 
        fold1_x_train.extend(F[0:leng2]) 
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2]) 
        fold1_x_test.extend(F[leng2:leng2*2])
        fold1_x_test.extend(T[leng3:leng3*2])
        fold1_x_test.extend(U[leng4:leng4*2])

        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        fold2_x_train.extend(T[0:leng3*2])
        fold2_x_train.extend(T[leng3*3:])
        fold2_x_train.extend(U[0:leng4*2])
        fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        fold2_x_test.extend(T[leng3*2:leng3*3])
        fold2_x_test.extend(U[leng4*2:leng4*3])

        fold3_x_train.extend(NR[0:leng1*3]) 
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        fold3_x_train.extend(T[0:leng3*3])
        fold3_x_train.extend(T[leng3*4:])
        fold3_x_train.extend(U[0:leng4*3])
        fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        fold3_x_test.extend(T[leng3*3:leng3*4])
        fold3_x_test.extend(U[leng4*3:leng4*4])

        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        fold4_x_train.extend(T[0:leng3*4])
        fold4_x_train.extend(T[leng3*5:])
        fold4_x_train.extend(U[0:leng4*4])
        fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        fold4_x_test.extend(T[leng3*4:leng3*5])
        fold4_x_test.extend(U[leng4*4:leng4*5])
    elif 'Twitter' in obj and method==1:
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)
        print('The len of file_list: ', len(file_list))
        if obj=='Twitter15':
            test_names=['CIKM_1000737','parisreview','CIKM_150','BBCBreaking','ferguson']
        elif obj=='Twitter16':
            test_names=['E92','E2016-100777','whitehouse','charliehebdo','sydneysiege']
            
        fold0_x_train=[]
        fold0_x_100_train=[]
        fold0_x_test=[]
        fold0_x_list=[]
        for line in open(labelPath): 
            line = line.rstrip() 
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]         
            if eid in file_list:
                if event in test_names:
                    fold0_x_test.append(eid)
                else:
                    fold0_x_train.append(eid) 
                    
        fold0_x_100_train=fold0_x_test
    elif 'Twitter' in obj and method==2:
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)
        print('The len of file_list: ', len(file_list))
        if obj=='Twitter15':
            test_names=['CIKM_1000737','parisreview','CIKM_150','BBCBreaking','ferguson']
            target_name=['BBCBreaking']
        elif obj=='Twitter16':
            test_names=['E92','E2016-100777','whitehouse','charliehebdo','sydneysiege']
            target_name=['sydneysiege']
        target_list=[]
        fold0_x_train=[]
        fold0_x_100_train=[]
        fold0_x_test=[]
        fold0_x_list=[]
        for line in open(labelPath): 
            line = line.rstrip() 
           
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]         

            if eid in file_list:
                if event in test_names:
                    fold0_x_list.append(eid)
                elif event in target_name:
                    fold0_x_test.append(eid)
                else:
                    fold0_x_train.append(eid) 
        fold0_x_100_train=fold0_x_test[:10]
        fold0_x_test=fold0_x_test[10:]

    elif 'Twitter' in obj and method == 3:
        labelPath = os.path.join(label_path)
        
        labelDic=get_label(obj)
        test_names = []
        for label, labelevent in labelDic.items():
            # print(labelevent)
            a = random.sample(list(labelevent.keys()), 5)
            test_names.extend(a)
        # print(test_names)
        fold0_x_train = []
        fold0_x_100_train = []
        fold0_x_test = []
        fold0_x_list = []
        for line in open(labelPath):
            line = line.rstrip()
    
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]
            
            if event in test_names:
                fold0_x_list.append(eid)
            else:
                fold0_x_train.append(eid)
        shuffle(fold0_x_list)
        fold0_x_100_train = fold0_x_list[0:int(len(fold0_x_list)*0.2)]
        fold0_x_test = fold0_x_list[int(len(fold0_x_list)*0.2):]
        # print(fold0_x_test)

        shuffle(fold0_x_train)
        fold0_x_train_8 = fold0_x_train[0:int(len(fold0_x_train) * 0.8)]
        fold0_x_train_2 = fold0_x_train[int(len(fold0_x_train) * 0.8):]
    elif 'Twitter' in obj and method==4:
        if obj=='Twitter15':
            label_path = '../data/label_15.json'
        elif obj=='Twitter16':
            label_path = '../data/label_16.json'
        with open(label_path, encoding='utf-8') as f:
            json_inf = json.load(f)
        print('The len of file_list: ', len(json_inf))

        F=list(json_inf.keys())
        random.shuffle(F) 
        l1=len(F)
        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        fold0_x_val, fold1_x_val, fold2_x_val, fold3_x_val, fold4_x_val = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l1 * 0.1)

        fold0_x_test.extend(F[0:leng1])
        fold0_x_val.extend(F[leng1:leng1+leng2])
        fold0_x_train.extend(F[leng1+leng2:])
        #print(len(F),len(T))

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2+leng2:])
      
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_val.extend(F[leng1 * 2:leng1 * 2+leng2])
        #print(len(F),len(T))

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3+leng2:])
       
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        #print(len(fold2_x_test))
        fold2_x_val.extend(F[leng1 * 3:leng1 * 3+leng2])
        #print(len(fold2_x_test))

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4+leng2:])
 
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_val.extend(F[leng1 * 4:leng1 * 4+leng2])


        #print(len(F),len(T))

        fold4_x_train.extend(F[leng2:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_val.extend(F[0 :leng2])

 
   
    elif 'Twitter' in obj and method==5:
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)
        print('The len of file_list: ', len(file_list))
        if obj=='Twitter15':
            test_names=['CIKM_1000737','parisreview','CIKM_150','BBCBreaking','ferguson']
        elif obj=='Twitter16':
            test_names=['E92','E2016-100777','whitehouse','charliehebdo','sydneysiege']
            
        fold0_x_train=[]
        fold0_x_100_train=[]
        fold0_x_test=[]
        fold0_x_list=[]
        for line in open(labelPath): 
            line = line.rstrip() 
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]         
            if eid in file_list:
                if event in test_names:
                    fold0_x_list.append(eid)
                else:
                    fold0_x_train.append(eid) 
        shuffle(fold0_x_list) 
                   
        fold0_x_100_train=fold0_x_list[0:20] 
        fold0_x_test=fold0_x_list[20:]   
        with open( './t16_test_tweets.pkl', 'wb') as t:
            pickle.dump(fold0_x_test,t)
        print(fold0_x_test)  
    if method==0:
        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_train = list(fold0_x_train)
        shuffle(fold0_train)
        fold1_test = list(fold1_x_test)
        shuffle(fold1_test)
        fold1_train = list(fold1_x_train)
        shuffle(fold1_train)
        fold2_test = list(fold2_x_test)
        shuffle(fold2_test)
        fold2_train = list(fold2_x_train)
        shuffle(fold2_train)
        fold3_test = list(fold3_x_test)
        shuffle(fold3_test)
        fold3_train = list(fold3_x_train)
        shuffle(fold3_train)
        fold4_test = list(fold4_x_test)
        shuffle(fold4_test)
        fold4_train = list(fold4_x_train)
        shuffle(fold4_train)
        return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)
    elif method==3:

        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_train_8 = list(fold0_x_train_8)
        shuffle(fold0_train_8)
        fold0_train_2 = list(fold0_x_train_2)
        shuffle(fold0_train_2)
        fold0_100_train = list(fold0_x_100_train)
        shuffle(fold0_100_train)
        return list(fold0_test), list(fold0_train_8),list(fold0_train_2),list(fold0_100_train)
    elif method==4 :
        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_val = list(fold0_x_val)
        shuffle(fold0_val)
        fold0_train = list(fold0_x_train)
        shuffle(fold0_train)
        fold1_test = list(fold1_x_test)
        shuffle(fold1_test)
        fold1_val = list(fold1_x_val)
        shuffle(fold1_val)
        fold1_train = list(fold1_x_train)
        shuffle(fold1_train)
        fold2_test = list(fold2_x_test)
        shuffle(fold2_test)
        fold2_val = list(fold2_x_val)
        shuffle(fold2_val)
        fold2_train = list(fold2_x_train)
        shuffle(fold2_train)
        fold3_test = list(fold3_x_test)
        shuffle(fold3_test)
        fold3_val = list(fold3_x_val)
        shuffle(fold3_val)
        fold3_train = list(fold3_x_train)
        shuffle(fold3_train)
        fold4_test = list(fold4_x_test)
        shuffle(fold4_test)
        fold4_val = list(fold4_x_val)
        shuffle(fold4_val)
        fold4_train = list(fold4_x_train)
        shuffle(fold4_train)
        return list(fold0_test),list(fold0_val),list(fold0_train),\
           list(fold1_test),list(fold1_val),list(fold1_train),\
           list(fold2_test),list(fold2_val),list(fold2_train),\
           list(fold3_test),list(fold3_val),list(fold3_train),\
           list(fold4_test), list(fold4_val),list(fold4_train)
    else:
        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_train = list(fold0_x_train)
        shuffle(fold0_train)
        fold0_100_train = list(fold0_x_100_train)
        shuffle(fold0_100_train)
        return list(fold0_test), list(fold0_train),list(fold0_100_train)

   
