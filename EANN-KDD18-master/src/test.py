import os
import pickle

id = pickle.load(open("../data/weibo/train_id.pickle", 'rb'))
print(id)
func = lambda z: dict([(x, y) for y, x in z.items()])
# print(id)
# print(func(id))
id1=func(func(id))

print(id1.values())
id = pickle.load(open("../data/weibo/test_id.pickle", 'rb'))
id1=func(func(id))
print(id1.values())

id = pickle.load(open("../data/weibo/validate_id.pickle", 'rb'))
id1=func(func(id))
print(id1.values())


def get_eventnum(datasetname):
    if datasetname == 'Twitter15':
        data_path = '../data/twitter15/'
        label_path = '../data/Twitter15_label_All.txt'
    elif datasetname == 'Twitter16':
        data_path = '../data/twitter16/'
        label_path = '../data/Twitter16_label_All.txt'

    if 'Twitter' in datasetname:
        path = data_path
        label_path = label_path
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)  # 662 ['.DS_Store', '498430783699554305', '500378223977721856'...]
        print('The len of file_list: ', len(file_list))
        eventDic = {}  # 字典 {'615689290706595840': 'true',...}
        eventnumDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[
                2]  # eid '656955120626880512' label 'false'

            if eid in file_list:
                if event in eventDic:
                    eventnumDic[eid] = eventDic[event]
                elif event not in eventDic:
                    eventDic[event] = len(eventDic)
                    eventnumDic[eid] = eventDic[event]

        print(eventnumDic)
        eventnumDic1 = func(func(eventnumDic))
        print(eventnumDic1.values())
        print(len(eventnumDic1.values()))
        pickle.dump(eventnumDic, open("../data/" + datasetname + "_id_event.pickle", 'wb+'))
    elif 'Pheme' in datasetname:
        label_path = '../data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)
        eventDic = {}
        eventnumDic = {}
        train_names = ['ferguson', 'ebolaessien', 'ottawashooting', 'princetoronto', 'gurlitt', 'sydneysiege','charliehebdo']
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]

            if event in eventDic:
                eventnumDic[eid] = eventDic[event]
            elif event not in eventDic and event in train_names:
                eventDic[event] = len(eventDic)
                eventnumDic[eid] = eventDic[event]

        print(eventnumDic)
        print(len(eventnumDic.values()))
        # with open(path + eventid + '/early_tweets.pkl', 'wb') as t:
        #     pickle.dump(early_tweets, t)
        pickle.dump(eventnumDic, open("../data/"+datasetname+"_id_event_data.pickle", 'wb+'))

def get_eventnum_data(datasetname):
    if datasetname == 'Twitter15':
        label_path = '../data/Twitter15_label_All.txt'
    elif datasetname == 'Twitter16':
        label_path = '../data/Twitter16_label_All.txt'

    if 'Twitter' in datasetname:
        label_path = label_path
        labelPath = os.path.join(label_path)
    
        eventDic = {}
        eventnumDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[
                2]  # eid '656955120626880512' label 'false'

            if event in eventDic:
                eventnumDic[eid] = eventDic[event]
            elif event not in eventDic:
                eventDic[event] = len(eventDic)
                eventnumDic[eid] = eventDic[event]

        print(eventnumDic)
        eventnumDic1 = func(func(eventnumDic))
        print(eventnumDic1.values())
        print(len(eventnumDic1.values()))
        pickle.dump(eventnumDic, open("../data/" + datasetname + "_id_event_data.pickle", 'wb+'))
    elif 'Pheme' in datasetname:
        label_path = '../data/pheme/Pheme_label_All.txt'
        labelPath = os.path.join(label_path)
        eventDic = {}
        eventnumDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, event, eid, time = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2], line.split('\t')[3]

            if event in eventDic:
                eventnumDic[eid] = eventDic[event]
            elif event not in eventDic:
                eventDic[event] = len(eventDic)
                eventnumDic[eid] = eventDic[event]

        print(eventnumDic)
        print(len(eventnumDic.values()))
        # with open(path + eventid + '/early_tweets.pkl', 'wb') as t:
        #     pickle.dump(early_tweets, t)
        pickle.dump(eventnumDic, open("../data/"+datasetname+"_id_event_data.pickle", 'wb+'))
datasetname='Pheme'
# get_eventnum_data(datasetname)
