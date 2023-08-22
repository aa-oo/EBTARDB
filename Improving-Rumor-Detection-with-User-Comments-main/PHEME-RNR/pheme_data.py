import pickle
import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers.data.processors.utils import InputExample

### Step 1.1 : Data Viewing and Simple Preprocessing
# raw_data = pd.read_csv('./data/raw_data.csv')
# print(raw_data)
# raw_data.head() 
# print(raw_data.head())
# raw_data.sort_values(by='count', inplace=True)
# raw_data.head(500)
# raw_data.shape[0]


import pandas as pd
import os
import json
import numpy as np
#%%
# # Filter irrelevant characters and links
def sentence_process(string):
    word_list = string.split(' ')
    new_list = []
    for word in word_list:
        if word.startswith('@') or ('http' in word):
            continue
        else:
            new_list.append(word)
    final_string = ' '.join(new_list)
    if not final_string.endswith('\n'):
        final_string += '\n'

    return final_string

#%%

title = ['id','text_comments','text_only','comments_only','label','count']  
csv_list = []
base_path ='../../Data/data/pheme/all'
label_path='../../Data/data/pheme/pheme_label.json'
#%%
rumour_path=base_path
with open(label_path, 'r', encoding='utf-8') as file:
    tags = json.load(file)
for rumour_tweet in os.listdir((rumour_path)):
    text_comments = ""
    text_only = ""
    comments_only = ""
    count = 0
    label = tags[rumour_tweet]
    rumour_text = os.path.join(rumour_path, rumour_tweet, 'source-tweets')
    for tweet_text in os.listdir(rumour_text):
        json_path = os.path.join(rumour_text,tweet_text)
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            text_comments += sentence_process(json_data['text'])
            text_comments += '[SEP]'
            text_only += sentence_process(json_data['text'])
            text_only += '[SEP]'

    rumour_comment = os.path.join(rumour_path, rumour_tweet, 'reactions')
    reaclist=[]
    for comment in os.listdir(rumour_comment):
        json_path = os.path.join(rumour_comment, comment)
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            if sentence_process(json_data['text']) != '\n':
                # text_comments += sentence_process(json_data['text'])
                # text_comments += '[SEP]'
                # comments_only += sentence_process(json_data['text'])
                # comments_only += '[SEP]'
                count += 1
                time=json_data['created_at']
                reaclist.append([sentence_process(json_data['text']),time])

    df = pd.DataFrame(reaclist, columns=['text', 'time'])
    df = df.sort_values(by="time")
    early_tweets = df.iloc[:, 0].to_list()
    for i in range(len(early_tweets)):
        text_comments += early_tweets[i]
        text_comments += '[SEP]'
        comments_only += early_tweets[i]
        comments_only += '[SEP]'
    single_list = [str(rumour_tweet),text_comments, text_only, comments_only, label, count]
    csv_list.append(single_list)

csv_file = pd.DataFrame(columns=title, data=csv_list)
csv_file.to_csv('./data/raw_data1.csv', index=0)

raw_data = pd.read_csv('./data/raw_data_Pheme.csv')
early_tweets = raw_data .iloc[:, 0].to_list()

label_path = '../../Data/data/pheme/Pheme_label_All.txt'
labelPath = os.path.join(label_path)

eventlist = []
for line in open(labelPath):
    line = line.rstrip()
    label, event, eid, time = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3]
    eventlist.append([label, event, eid, time])

df = pd.DataFrame(eventlist, columns=['label', 'event', 'eid', 'time'])


fold_list = df.iloc[:, 2].to_list()
print(len(early_tweets))
print(early_tweets)
print(len(fold_list))
print(fold_list)
for i in early_tweets:
    if i in fold_list:
        continue
    else:
        print(i)



import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers.data.processors.utils import InputExample

# ## Step 1.1 : Data Viewing and Simple Preprocessing
# raw_data = pd.read_csv('./data/raw_data.csv')
# print(raw_data)
# raw_data.head() 
# print(raw_data.head())
# raw_data.sort_values(by='count', inplace=True)
# raw_data.head(500)
# raw_data.shape[0]

def get_labeljson(datasetname):
    label={}
    if datasetname=='Pheme':
        data_path ='../../Data/PHEME_veracity/all-rnr-annotated-threads'
    # dataPath = os.path.join(data_path)
    file_list = os.listdir(data_path)
    for event in os.listdir(data_path):
        non_rumour = os.path.join(data_path, event, 'non-rumours')
        for eid in os.listdir(non_rumour):
            label[eid]='non_rumor'
        rumour = os.path.join(data_path, event, 'rumours')
        for eid in os.listdir(rumour):
            label[eid]='rumor'
    with open(data_path + '/pheme_label.json', 'wb') as t:
        pickle.dump(label,t)
def get_labeltxt(datasetname):
    label={}
    if datasetname=='Pheme':
        data_path ='../../Data/PHEME_veracity/all-rnr-annotated-threads'
    # dataPath = os.path.join(data_path)
    file_list = os.listdir(data_path)
    for event in os.listdir(data_path):
        non_rumour = os.path.join(data_path, event, 'non-rumours')
        for eid in os.listdir(non_rumour):
            label[eid]='non_rumor'
        rumour = os.path.join(data_path, event, 'rumours')
        for eid in os.listdir(rumour):
            label[eid]='rumor'
    with open(data_path + '/pheme_label.json', 'wb') as t:
        pickle.dump(label,t)

def get_labeleid(datasetname):
    if datasetname == 'Twitter15':
        data_path = '../data/twitter15/'
        label_path = '../data/Twitter15_label_All.txt'
    elif datasetname == 'Twitter16':
        data_path = '../data/twitter16/'
        label_path = '../data/Twitter16_label_All.txt'
    elif datasetname == 'Pheme':
        data_path = '../../data/pheme/all'
        label_path = '../../data/pheme/pheme_label.json'
    path = data_path
    label_path = label_path
    if 'Twitter' in datasetname:
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)  # 662 ['.DS_Store', '498430783699554305', '500378223977721856'...]
        print('The len of file_list: ', len(file_list))

        labelDic = {'unverified': {}, 'non-rumor': {}, 'true': {}, 'false': {}}  # 字典 {'615689290706595840': 'true',...}

        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[
                2]  # eid '656955120626880512' label 'false'

            if eid in labelDic[label]:
                labelDic[label].append(eid)
            else:
                labelDic[label] = [eid]
        print(labelDic)
        return labelDic
    if 'Pheme' in datasetname:
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)  # 662 ['.DS_Store', '498430783699554305', '500378223977721856'...]
        print('The len of file_list: ', len(file_list))

        labelDic = {'rumor': [], 'non-rumor': []}  #  {'615689290706595840': 'true',...}

        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[
                2]  # eid '656955120626880512' label 'false'

            if eid in labelDic[label]:
                labelDic[label].append(eid)
            else:
                labelDic[label] = [eid]
        print(labelDic)
        return labelDic
