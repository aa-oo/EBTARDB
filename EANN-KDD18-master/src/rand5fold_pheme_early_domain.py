import random
from random import shuffle
import os
from datetime import datetime

import pandas as pd
cwd=os.getcwd()




def load5foldDataP(obj,method):
    
    if obj == "Pheme" and method==0:

        rumor_path = './data/pheme/rumor/'
        non_rumor_path = './data/pheme/non-rumor/'
        
        rumor_dirs = os.listdir(rumor_path)
        non_rumor_dirs = os.listdir(non_rumor_path)


        #l1,l2 = len(rumor_dirs), len(non_rumor_dirs)

        F = rumor_dirs
        T = non_rumor_dirs

        #T = T[0:1789]

        print('rumor : non-romor', len(F), len(T))
        random.shuffle(F) 
        random.shuffle(T) 

        l1,l2 = len(F), len(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)

        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        #print(len(F),len(T))

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        #print(len(F),len(T))

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        #print(len(fold2_x_test))
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        #print(len(fold2_x_test))

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        #print(len(F),len(T))

        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    elif obj == "Pheme" and method==1:
        label_path = '../data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)

        eventlist = []
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]

            time = str(datetime.strptime(time, '%a %b %d %H:%M:%S %z %Y'))

            # time = time[0:13]
            eventlist.append([label, event, eid, time])

        random.shuffle(eventlist)
        df = pd.DataFrame(eventlist, columns=['label', 'event', 'eid', 'time'])
        


        df = df.sort_values(by="time")
        print(df)

        fold_list = df.iloc[:, 2].to_list()
        fold0_x_train = fold_list[0:int(len(fold_list) * 0.7)]
        fold0_x_val=fold_list[int(len(fold_list)* 0.7) :int(len(fold_list)* 0.8)]
        fold0_x_test = fold_list[int(len(fold_list) * 0.8):]
    elif obj == "Pheme" and method==2:
        label_path = './data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)
        train_names=['ferguson','ebolaessien','ottawashooting','princetoronto','gurlitt','sydneysiege']
        test_names=['charliehebdo','putinmissing']

        
        fold0_x_train=[]
        fold0_x_test=[]
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]
            
            if event in train_names:
                fold0_x_train.append(eid)
            elif event in test_names:
                fold0_x_test.append(eid)
    elif obj == "Pheme" and method==3:
        label_path = '../data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)
        # train_names=['ferguson','ebolaessien','ottawashooting','princetoronto','gurlitt','sydneysiege']
        # test_names=['charliehebdo']
        train_names=['ferguson','ebolaessien','ottawashooting','princetoronto','gurlitt','sydneysiege','putinmissing','charliehebdo'] 
        test_names=['germanwingscrash']

        fold0_x_train=[]
        fold0_x_100_train=[]
        fold0_x_test=[]
        fold0_x_list=[]
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]         
            time = str(datetime.strptime(time, '%a %b %d %H:%M:%S %z %Y'))
            if event in train_names:
                fold0_x_train.append(eid)
            elif event in test_names:
                fold0_x_list.append([label, event, eid, time])
            
        df = pd.DataFrame(fold0_x_list, columns=['label', 'event', 'eid', 'time'])
        print(df)


        df = df.sort_values(by="time")
        fold_list = df.iloc[:, 2].to_list()
        fold0_x_100_train=fold_list[0:100]
        print(len(fold0_x_100_train))
        fold0_x_test = fold_list[100:]
        print(len(fold0_x_test))

        shuffle(fold0_x_train)
        fold0_x_train_8=fold0_x_train[0:int(len(fold0_x_train)*0.8)]
        fold0_x_train_2=fold0_x_train[int(len(fold0_x_train)*0.8):]
    elif obj == "Pheme" and method==4:

        label_path = '../data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)
        F=[]
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]         
            F.append(eid)

        print('rumor : non-romor', len(F))
        random.shuffle(F) 
        

        l1= len(F)

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
    if method==0 :
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
    elif method==1:
        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_val = list(fold0_x_val)
        shuffle(fold0_val)
        fold0_train = list(fold0_x_train)
        shuffle(fold0_train)
     
        return list(fold0_test), list(fold0_val),list(fold0_train)
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
    elif method==4:
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
     
        return list(fold0_test), list(fold0_train)
   
