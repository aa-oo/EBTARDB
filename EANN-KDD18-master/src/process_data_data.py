# encoding=utf-8
import argparse
import pickle
# import cPickle as pickle
import random
from importlib import reload
from random import *
import numpy as np
import torchvision
# from numpy import unicode
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
from gensim.models import Word2Vec
import jieba
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
from gensim.models import Word2Vec
#-*- encoding:utf-8 -*-
import importlib, sys

from rand5fold_early import load5foldDataT
from rand5fold_pheme_early_domain import load5foldDataP

importlib.reload(sys)
cwd=os.getcwd()
# sys.setdefaultencoding('utf-8')

def get_split(text):
	method='c'
	if method=='a':
		# Delete '[SEP]'
		text = text.replace('[SEP]','')
		# print(len(text.split()))
		l_total = []
		l_parcial = []
		if len(text.split())//150 >0:
			n = len(text.split())//150
		else:
			n = 1
		for w in range(n):
			if w == 0:
				l_parcial = text.split()[:200]
				l_total.append(" ".join(l_parcial))
			else:
				l_parcial = text.split()[w*150:w*150 + 200]
				l_total.append(" ".join(l_parcial))

		return l_total
	elif method=='b':
		# Delete '[SEP]'
		# text = text.replace('[SEP]','')
		text_list=[]
		text1=''
		text_list = text.split('[SEP]')[0:3]
		for x in text_list:
			text1+=x
		text=text1
		# print(len(text.split()))
		l_total = []
		l_parcial = []
		if len(text.split())//150 >0:
			n = len(text.split())//150
		else:
			n = 1
		for w in range(n):
			if w == 0:
				l_parcial = text.split()[:200]
				l_total.append(" ".join(l_parcial))
			else:
				l_parcial = text.split()[w*150:w*150 + 200]
				l_total.append(" ".join(l_parcial))

		return l_total
	elif method=='c':
		# Delete '[SEP]'
		text = text.replace('[SEP]','')
		return text

def get_split_test(text):
	method='b'
	if method=='a':
		# Delete '[SEP]'
		text = text.replace('[SEP]','')
		return text
	elif method=='b':
		# Delete '[SEP]'
		# text = text.replace('[SEP]','')
		text_list=[]
		text1=''
		text_list = text.split('[SEP]')[0:3]
		for x in text_list:
			text1+=x
		text=text1
		return text
	elif method=='c':
		# Delete '[SEP]'
		text = text.replace('[SEP]','')
		return text

def stopwordslist(filepath = '../data/weibo/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r',encoding="utf-8").readlines():
        # line = unicode(line, "utf-8").strip()
        line = line.strip()
        stopwords[line] = 1
    #stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
#
def read_image():
    image_list = {}
    file_list = ['../Data/weibo/nonrumor_images/', '../Data/weibo/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def write_txt(data):
    f = open("../data/weibo/top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l+"\n")
        f.write("\n")
        f.write("\n")
    f.close()
text_dict = {}

def write_data(flag, image, text_only,datalist,args):

    def read_post(flag,datalist,args):
        stop_words = stopwordslist() #{'$': 1, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,

        datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = datalist[0], datalist[1], \
                                                                                         datalist[2], datalist[3], \
                                                                                         datalist[4]

        raw_data = pd.read_csv('../data/raw_data_' + datasetname + '.csv')  # intid 源+评论 源 评论 标签 评论数

        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        raw_data['label'] = raw_data['label'].apply(lambda x: label2id[x])



        raw_data_test = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_test)]
        raw_data_test_100 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_100_train)]
        raw_data_train_8 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_8)]
        raw_data_train_2 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_2)]
        # print(len(raw_data_test), len(raw_data_test_100), len(raw_data_train_2), len(raw_data_train_8))

        raw_data_test = raw_data_test[['id', 'text_comments', 'label']]
        raw_data_test = raw_data_test.rename(columns={'text_comments': 'text'})
        raw_data_test_100 = raw_data_test_100[['id', 'text_comments', 'label']]
        raw_data_test_100 = raw_data_test_100.rename(columns={'text_comments': 'text'})
        raw_data_train_2 = raw_data_train_2[['id', 'text_comments', 'label']]
        raw_data_train_2 = raw_data_train_2.rename(columns={'text_comments': 'text'})
        raw_data_train_8 = raw_data_train_8[['id', 'text_comments', 'label']]
        raw_data_train_8 = raw_data_train_8.rename(columns={'text_comments': 'text'})

        raw_data_test = raw_data_test.dropna(axis=0)
        raw_data_test_100 = raw_data_test_100.dropna(axis=0)
        raw_data_train_8 = raw_data_train_8.dropna(axis=0)
        raw_data_train_2 = raw_data_train_2.dropna(axis=0)

        data_test = raw_data_test.copy()
        data_test_100 = raw_data_test_100.copy()
        data_train_8 = raw_data_train_8.copy()
        data_train_2 = raw_data_train_2.copy()
        data_test = data_test.reindex(np.random.permutation(data_test.index))
        data_test_100 = data_test_100.reindex(np.random.permutation(data_test_100.index))
        data_train_8 = data_train_8.reindex(np.random.permutation(data_train_8.index))
        data_train_2 = data_train_2.reindex(np.random.permutation(data_train_2.index))
        data_test_100.head(10)

        data_test.reset_index(drop=True, inplace=True)
        data_test_100.reset_index(drop=True, inplace=True)
        data_train_8.reset_index(drop=True, inplace=True)
        data_train_2.reset_index(drop=True, inplace=True)

        train_tmp_8 = data_train_8.copy()
        train_tmp_8['text_split'] = data_train_8['text'].apply(get_split)
        train_8 = train_tmp_8
        train_tmp_2 = data_train_2.copy()
        train_tmp_2['text_split'] = data_train_2['text'].apply(get_split)
        train_2 = train_tmp_2
        val_tmp = data_test.copy()
        val_tmp_100 = data_test_100.copy()
        if args.data_division=='random' or args.data_division=='time':
            val_tmp['text_split'] = data_test['text'].apply(get_split)
            val_tmp_100['text_split'] = data_test_100['text'].apply(get_split)
        elif args.data_division=='event':
            val_tmp['text_split'] = data_test['text'].apply(get_split_test)
            val_tmp_100['text_split'] = data_test_100['text'].apply(get_split_test)
        val = val_tmp
        val_100 = val_tmp_100

        if flag == "train":
            id = train_8
        elif flag == "validate":
            id = val_100
        elif flag == "test":
            id = val


        data_frame1=train_8.append(val_100)
        data_frame=data_frame1.append(val)

        post_content = []
        labels = []
        image_ids = []
        twitter_ids = []
        data = []
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        key = -1
        map_id = {}
        top_data = []

        eventid = pickle.load(open("../data/"+datasetname+"_id_event_data.pickle", 'rb'))

        for i in range(len(data_frame)):
            line_data = [str(data_frame.iloc[i]['id'])]
            label = data_frame.iloc[i]['label']

            l = data_frame.iloc[i]['text_split']
            l = clean_str_sst(l)

            # sent = str(l)
            # seg_list = sent.split()
            # new_seg_list = " ".join(seg_list)
            #
            # clean_l = new_seg_list




            seg_list = jieba.cut_for_search(l)

            new_seg_list = []
            for word in seg_list:
                if word not in stop_words:
                    new_seg_list.append(word)

            clean_l = " ".join(new_seg_list)

            if len(clean_l) > 10 and (int(line_data[0]) in id.iloc[:, 0].to_list()):
                post_content.append(l)
                line_data.append('URL')
                line_data.append(l)
                line_data.append(clean_l)
                line_data.append(label)
                event = int(eventid[line_data[0]])
                if event not in map_id:
                    map_id[event] = len(map_id)
                    event = map_id[event]
                else:
                    event = map_id[event]

                line_data.append(event)

                data.append(line_data)



        data_df = pd.DataFrame(np.array(data), columns=column)
        write_txt(top_data) #top_data=[]

        return post_content, data_df

    post_content, post = read_post(flag,datalist,args)
    print("Original post length is " + str(len(post_content)))



    def find_most(db):
        maxcount = max(len(v) for v in db.values())
        return [k for k, v in db.items() if len(v) == maxcount]

    def select(train, selec_indices):
        temp = []
        for i in range(len(train)):
            ele = list(train[i])
            temp.append([ele[i] for i in selec_indices])
            #   temp.append(np.array(train[i])[selec_indices])
        return temp

#     def balance_event(data, event_list):
#         id = find_most(event_list)[0]
#         remove_indice = random.sample(range(min(event_list[id]), \
#                                             max(event_list[id])), int(len(event_list[id]) * 0.9))
#         select_indices = np.delete(range(len(data[0])), remove_indice)
#         return select(data, select_indices)

    def paired(text_only = False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event= []
        label = []
        post_id = []
        image_id_list = []
        #image = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post']) #词
                ordered_post.append(post.iloc[i]['post_text'])   #句子
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id)


                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=int)
        ordered_event = np.array(ordered_event, dtype=int)





        #
        if flag == "test":
            y = np.zeros(len(ordered_post))
        else:
            y = []


        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id":np.array(post_id),
                "image_id":image_id_list}
        #print(data['image'][0])


        print("data size is " + str(len(data["post_text"])))
        
        return data

    paired_data = paired(text_only)  #dict 8 {'post_text': array(2230),''

    print("paired post length is "+str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data


def load_data(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text'])+list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text





def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)



def get_data(text_only,args):
    #text_only = False

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    if args.datasetname == 'Twitter16':
        data_path = '../data/twitter16/'
        label_path = '../data/Twitter16_label_All.txt'
    elif args.datasetname == 'Twitter15':
        data_path = '../data/twitter15/'
        label_path = '../data/Twitter15_label_All.txt'
    global label2id
    # print(raw_data)
    # raw_data.sort_values(by='count', inplace=True)
    if 'Twitter' in args.datasetname:
        label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
        }
    elif 'Pheme' in args.datasetname:
        label2id = {
            "rumor": 0,
            "non-rumor": 1, }
        
    


    train_data = write_data("train", image_list, text_only,args.datalist,args)
    valiate_data = write_data("validate", image_list, text_only,args.datalist,args)
    test_data = write_data("test", image_list, text_only,args.datalist,args)

    print("loading data...")

    # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
    vocab, all_text = load_data(train_data, valiate_data, test_data)  # 3745 {‘移’：89.0，‘民’：848.0..} 3355 ['移民 分配 村子..‘,..]
    # print(str(len(all_text)))

    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))

    #
    #
    # word_embedding_path = "../data/"+args.datasetname+'_'+args.data_division+"_w2v.pickle"


    # w2v=pickle.load(open(word_embedding_path, 'rb'), encoding='latin')  # 4626 dict {’耀‘：array 32 [0.34510726,0.9283621,....],'挂'：}
    # # print(temp)
    # # #
    # print("word2vec loaded!")


    # add_unknown_words(w2v, vocab)
    # W, word_idx_map = get_W(w2v)

    # W2 = rand_vecs = {}
    # w_file = open("../data/"+args.datasetname+args.data_division+"_word_embedding.pickle", "wb")
    # pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    # w_file.close()
    return train_data, valiate_data, test_data



def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    #parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= True, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--datasetname', type=str, default='Pheme', help='')
    parser.add_argument('--data_division', type=str, default='random', help='')
    return parser

if __name__ == "__main__":
    image_list = []
    text_only=True
    datasetname='Pheme'
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output ='../data/output/Pheme/'
    args = parser.parse_args([train, test, output])
    args.datasetname=datasetname
    args.data_division='time'
    if datasetname == 'Twitter16':
        data_path = '../data/twitter16/'
        label_path = '../data/Twitter16_label_All.txt'
    elif datasetname == 'Twitter15':
        data_path = '../data/twitter15/'
        label_path = '../data/Twitter15_label_All.txt'
    if 'Twitter' in datasetname:
        label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
        }
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldDataT(datasetname, data_path,
                                                                                           label_path, 3)
    elif 'Pheme' in datasetname:
        label2id = {
            "rumor": 0,
            "non-rumor": 1, }
        fold0_x_test, fold0_x_val, fold0_x_train = load5foldDataP(datasetname, 1)
    # print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))

    # fold0_x_train_8.extend(fold0_x_train_2)
    # datalist = [datasetname, fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_val), len(fold0_x_train))
    datalist = [args.datasetname, fold0_x_test, fold0_x_train, fold0_x_val, fold0_x_val]

    train_data = write_data("train", image_list, text_only, datalist,args)
    valiate_data = write_data("validate", image_list, text_only, datalist,args)
    test_data = write_data("test", image_list, text_only, datalist,args)


    # print("loading data...")
    # # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
    vocab, all_text = load_data(train_data,valiate_data, test_data)
    #
    # # print(str(len(all_text)))
    #
    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))
    #
    # #
    # #

    word_embedding_path = "../data/" + datasetname +'_'+args.data_division+ "_w2v.pickle"
    if not os.path.exists(word_embedding_path):
        min_count = 1
        size = 32
        window = 4

        w2v = Word2Vec(all_text, min_count=min_count, vector_size=size, window=window)

        temp = {}
        for word in w2v.wv.index_to_key:
            temp[word] = w2v.wv[word]

        w2v = temp
        pickle.dump(w2v, open(word_embedding_path, 'wb+'))
    else:
        w2v = pickle.load(open(word_embedding_path, 'rb'))
    # print(temp)
    # # #
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    W2 = rand_vecs = {}
    w_file = open("../data/"+args.datasetname+'_'+args.data_division+"_word_embedding.pickle", "wb")
    pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    w_file.close()
    # # w2v = add_unknown_words(w2v, vocab)
    # Whole_data = {}
    # file_path = "../Data/weibo/event_clustering.pickle"
    # # if not os.path.exists(file_path):
    # #     data = []
    # #     for l in train_data["post_text"]:
    # #         line_data = []
    # #         for word in l:
    # #             line_data.append(w2v[word])
    # #         line_data = np.matrix(line_data)
    # #         line_data = np.array(np.mean(line_data, 0))[0]
    # #         data.append(line_data)
    # #
    # #     data = np.array(data)
    # #
    # #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
    # #     cluster.fit(data)
    # #     y = np.array(cluster.labels_)
    # #     pickle.dump(y, open(file_path, 'wb+'))
    # # else:
    # # y = pickle.load(open(file_path, 'rb'))
    # # print("Event length is " + str(len(y)))
    # # center_count = {}
    # # for k, i in enumerate(y):
    # #     if i not in center_count:
    # #         center_count[i] = 1
    # #     else:
    # #         center_count[i] += 1
    # # print(center_count)
    # # train_data['event_label'] = y
    #
    # #
    # print("word2vec loaded!")
    # print("num words already in word2vec: " + str(len(w2v)))
    # add_unknown_words(w2v, vocab)
    # W, word_idx_map = get_W(w2v)
    # # # rand_vecs = {}
    # # # add_unknown_words(rand_vecs, vocab)
    # W2 = rand_vecs = {}
    # pickle.dump([W, W2, word_idx_map, vocab, max_l], open("../Data/weibo/word_embedding.pickle", "wb"))
    # print("dataset created!")



