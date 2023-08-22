
from collections import defaultdict
from collections import Counter
import sys, re

from gensim.models import Word2Vec
import jieba
from sklearn.cluster import AgglomerativeClustering

from rand5fold_early import *
from rand5fold_pheme_early_domain import *
import numpy as np
import torch
from sklearn import manifold
from test_TSNE import *
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

def stopwordslist(filepath = 'stop_words.txt'):
    stopwords = []
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        line = line.strip()
        stopwords.append(line)
    # print(stopwords)
    return stopwords


def clean_str_sst(sent):
    sent = str(sent)
    url_1 = re.compile(r'http://[a-zA-Z0-9.?/&=:]*')
    url_2 = re.compile(r'https://[a-zA-Z0-9.?/&=:]*')
    sent = url_1.sub(" ", sent)
    sent = url_2.sub(" ", sent)
    sent = re.sub(u"[，。 :,.；|-“”——_/+&;@、√《》～（）%())#！：【】……'［］•]", " ", sent)

    return sent.strip().lower()


def read_post(flag,datalist,args):
    stop_words = stopwordslist() #{'$': 1, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    datasetname,fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train=datalist[0],datalist[1],datalist[2],datalist[3],datalist[4]
        
    raw_data = pd.read_csv('../data/raw_data_'+datasetname+'.csv') 
        
    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
    raw_data['label']= raw_data['label'].apply(lambda x:label2id[x])

    raw_data_test = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_test)]
    raw_data_test_100 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_100_train)]
    raw_data_train_8 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_8)]
    raw_data_train_2 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_2)]
    print(len(raw_data_test), len(raw_data_test_100), len(raw_data_train_2), len(raw_data_train_8))

    raw_data_test = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_test)]
    raw_data_test_100 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_100_train)]
    raw_data_train_8 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_8)]
    raw_data_train_2 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_2)]
    print(len(raw_data_test), len(raw_data_test_100), len(raw_data_train_2), len(raw_data_train_8))

    raw_data_test = raw_data_test[['id','text_comments', 'label']]
    raw_data_test = raw_data_test.rename(columns={'text_comments': 'text'})
    raw_data_test_100 = raw_data_test_100[['id','text_comments', 'label']]
    raw_data_test_100 = raw_data_test_100.rename(columns={'text_comments': 'text'})
    raw_data_train_2 = raw_data_train_2[['id','text_comments', 'label']]
    raw_data_train_2 = raw_data_train_2.rename(columns={'text_comments': 'text'})
    raw_data_train_8 = raw_data_train_8[['id','text_comments', 'label']]
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

    if flag == "train_8":
        data_frame = train_8
    elif flag == "train_2":
        data_frame = train_2
    elif flag == "train_100":
        data_frame = val_100
    elif flag == "test":
            data_frame = val
    post_content = []
    labels = []
    image_ids = []
    twitter_ids = []
    data = []
    column = ['post_id',  'original_post', 'post_text', 'label', 'event_label']
    key = -1
    map_id = {}
    top_data = []
    
 
    for i in range(len(data_frame)):
        line_data = [str(data_frame.iloc[i]['id'])]
        label=data_frame.iloc[i]['label']
        l=data_frame.iloc[i]['text_split']
        l = clean_str_sst(l)

        sent = str(l)
        seg_list = sent.split()
        new_seg_list = " ".join(seg_list)
        

        clean_l = new_seg_list
        # print(clean_l)
        post_content.append(l)
        line_data.append(l)
        line_data.append(clean_l)
        line_data.append(label)
        if flag=='train_2' or flag=='train_8' :
            event=0
        elif flag=='test' or flag=='train_100' :
             event=1
 

        line_data.append(event)

        data.append(line_data)


            # print(data)
            #     return post_content
       
    data_df = pd.DataFrame(np.array(data), columns=column)

    return data_df


def create_vocab(dataframe):

    vocab = defaultdict(float)
    corpus_temp = dataframe['post_text']
    corpus = []
    for sentence in corpus_temp:
        # print(sentence)
        # print(len(sentence))
        word_list = sentence.split()
        corpus.append(word_list)
        # print(word_list)
        for word in word_list:
            # print(word)
            # if word != ' ':
            vocab[word] += 1
    # print(vocab)
    # print(len(vocab))
    return corpus, vocab


def add_known_words(word_vecs, vocab, min_df=1, k=32):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(word_vecs, k=32):
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def event_clustering(data, word_dict):
    event_data = []
    for sentence in data['post_text']:
        sentence_word = []
        sentence_word = sentence.split()
        line_data = []
        for word in sentence_word:
            line_data.append(word_dict[word])
        line_data = np.matrix(line_data)
        line_data = np.array(np.mean(line_data, 0))[0] 
        event_data.append(line_data)
    event_data = np.array(event_data)

    n_clusters = 5 
    affinity = 'cosine' # 'euclidean','l1','l2','mantattan','cosine'
    linkage = 'complete'
    event_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    event_cluster.fit(event_data)
    labels = np.array(event_cluster.labels_)
    return labels


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
 

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0) 

    total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)  

    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_mul // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 


def MMD(source,target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # batch_size = int(source.size()[0])
    # kernels = guassian_kernel(source, target, kernel_mul=kernel_mul,kernel_num=kernel_num, fix_sigma=fix_sigma)
    # XX = kernels[:batch_size, :batch_size]  # source <-> source
    # YY = kernels[batch_size:, batch_size:]  # target <-> target
    # XY = kernels[:batch_size, batch_size:]  # source <-> target
    # YX = kernels[batch_size:, :batch_size]  # target <-> source

    # loss = torch.mean(XX + YY - XY - YX) 


    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) 

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) 
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  
    	
    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def get_data(args):



    
    if args.datasetname=='Twitter16':
        data_path = '../../data/twitter16/'
        label_path = '../data/Twitter16_label_All.txt'
    elif args.datasetname=='Twitter15':
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
            "non-rumor": 1,            }
        
    
    source_train_data = read_post("train_8",args.datalist,args)
    validation_data = read_post("train_2",args.datalist,args)
    test_data = read_post("test",args.datalist,args)
    target_train_data = read_post("train_100",args.datalist,args)

    print("loading data...")


    source_train_data.dropna(subset=['post_text'], inplace=True)
    target_train_data.dropna(subset=['post_text'], inplace=True)
    test_data.dropna(subset=['post_text'], inplace=True)

    # post_length = []
    # for sent in COVID_data['post_text']:
    #     post_length.append(len(sent))
    # COVID_data['length'] = post_length
    # COVID_data = COVID_data.loc[COVID_data['length']<350]
    # for i in COVID_data['length']:
    #     print(i)


    """
    vocab size is: 81944
    number of sentences: 21586
    the max sentence length is: 349
    label number is: 21586
    fake news number is: 10178
    real news number is: 11408
    """

    all_data = source_train_data._append(target_train_data)
    all_data=all_data._append(test_data)
    all_data.reset_index(drop=True, inplace=True)

    all_text, vocab = create_vocab(all_data)

    # with open('vocab_dict.csv', 'w', newline='') as f:
    #     [f.write('{0} {1}\n'.format(key, value)) for key, value in vocab.items()]
    # f.close()
    print("vocabulary construction completed")
    # print(vocab)
    print("data loaded!")
    print("vocab size is: " + str(len(vocab)))
    print("number of sentences: " + str(len(all_text)))

    temp_sentence_length = 0
    for sentence in all_text:
        # print(sentence)
        length = len(sentence)
        if temp_sentence_length <= length:
            temp_sentence_length = length
            # print(sentence)
    max_sentence_length = temp_sentence_length
    print("the max sentence length is: " + str(max_sentence_length))




    # # # gensim.word2vec
    # min_count = 1 
    # size = 32 
    # window = 4 
    # sg = 0 
    # hs = 1
    
    # w2v = Word2Vec(all_text, hs=hs, min_count=min_count, window=window, vector_size=size)
    # temp = {}
    # for word in w2v.wv.index_to_key:
    #     temp[word] = w2v.wv[word]
    # w2v = temp
    # # print(w2v)
    # # print(w2v.keys())
    
    # print("loading word2vec vectors...")
    # print("number words already in word2vec: " + str(len(w2v)))
    
 
    # add_known_words(w2v, vocab)
    # W, word_idx_map = get_W(w2v)
    # W2 = rand_vecs = {}
    # add_known_words(rand_vecs, vocab)
    
    # pickle.dump([W, W2, word_idx_map, vocab, max_sentence_length], open("../data/"+args.datasetname+"_"+args.data_division+"_word_embedding.pickle", "wb"))


    # print(a)

    word_weight = pickle.load(open("../data/"+args.datasetname+"_"+args.data_division+"_word_embedding.pickle", 'rb'))
    W, W2, word_idx_map, vocab, max_sentence_length = word_weight[0], word_weight[1], word_weight[2], word_weight[3],\
                                                        word_weight[4]

    print("loading word2vec vectors...")
    print("number words already in word2vec: " + str(len(word_idx_map)))


    word_vector_dict = defaultdict(float)
    for k, v in word_idx_map.items():
        word_vector_dict[k] = W[v]



 
    data_0_feature = []
    data_1_feature = []
    # data_4_feature = []
    # COVID_feature = []
    #
    # MMD_data_0 = data_0.sample(n=600, replace=False, random_state=None, axis=0)
    # MMD_data_3 = data_3.sample(n=600, replace=False, random_state=None, axis=0)
    # MMD_data_4 = data_4.sample(n=600, replace=False, random_state=None, axis=0)
    # COVID_data = COVID_data.sample(n=600, replace=False, random_state=None, axis=0)
    #
    # print(source_train_data)

    for sentence in source_train_data['post_text']:
  
        sentence = sentence.split()
 
        line_data = []
        for word in sentence:
            line_data.append(word_vector_dict[word])
        line_data = np.matrix(line_data)
        line_data = np.array(np.mean(line_data, 0))[0]
        data_0_feature.append(line_data)
    
    data_0_feature = np.array(data_0_feature)
    data_0_event_label = source_train_data['label']

    for sentence in target_train_data['post_text']:
        sentence = sentence.split()
        line_data = []
        for word in sentence:
            line_data.append(word_vector_dict[word])
        line_data = np.matrix(line_data)
        line_data = np.array(np.mean(line_data, 0))[0]
        data_1_feature.append(line_data)
    data_1_feature = np.array(data_1_feature)
    
    data_1_event_label = target_train_data['label']
    
    


   
    # # source_train_data1=source_train_data.sample(n=300)
    # source_train_data1=source_train_data
    # data_0_feature1=[]
    # for sentence in source_train_data1['post_text']:
    #     # print(sentence)
    #     sentence = sentence.split()
    #     # print(sentence)
    #     line_data = []
    #     for word in sentence:
    #         line_data.append(word_vector_dict[word])
    #     line_data = np.matrix(line_data)
    #     line_data = np.array(np.mean(line_data, 0))[0]
    #     data_0_feature1.append(line_data)
    
    # data_0_feature1 = np.array(data_0_feature1)
    # data_0_event_label1 = source_train_data1['label']
    
    # target_data = pd.concat([target_train_data, test_data], axis=0)
    # # target_data1=target_data.sample(n=200)
    # target_data1=target_data
    # data_1_feature1=[]
    # for sentence in target_data1['post_text']:
    #     sentence = sentence.split()
    #     line_data = []
    #     for word in sentence:
    #         line_data.append(word_vector_dict[word])
    #     line_data = np.matrix(line_data)
    #     line_data = np.array(np.mean(line_data, 0))[0]
    #     data_1_feature1.append(line_data)
    # data_1_feature1 = np.array(data_1_feature1)
    
    # data_1_event_label1 = target_data1['label']
    
    # data_0 = np.array(data_0_feature1)
    # label_0 = np.array(data_0_event_label1)
    # data_1 = np.array(data_1_feature1)
    # label_1 = np.array(data_1_event_label1)
    # # print(data_np.shape)
    # # print(label_np.shape)
    # # print(weight_np)
    
    # features_data=np.concatenate((data_0,data_1),axis=0)
    # tsne = manifold.TSNE(n_components=2, init='pca')
    # features_tsne = tsne.fit_transform(features_data)
    # # features_tsne_1 = tsne.fit_transform(data_1)
    # print(features_tsne.shape)
    
    
    # # print(features_tsne.shape)
    # label=np.concatenate((label_0,label_1),axis=0)
    # domain_0=np.zeros(len(label_0),dtype = int)
    # domain_1=np.ones(len(label_1),dtype = int)
    # domain=np.concatenate((domain_0,domain_1),axis=0)
    # print(label.shape)
    # print(label)
    # print(domain.shape)
    # print(domain)
    
    # # print(features_tsne)
    # #
    # plot_embedding(args.datasetname,features_tsne, label,domain, "")
    
    # plt.show()
    # plt.savefig('./t-SNE/'+args.datasetname)
    
    # print(a)
    #
    # data_distribution = np.concatenate((data_0_feature, data_3_feature, data_4_feature, COVID_feature), axis=0)
    # data_distribution_event_label = data_0_event_label.append(data_3_event_label)
    # data_distribution_event_label = data_distribution_event_label.append(data_4_event_label)
    # data_distribution_event_label = data_distribution_event_label.append(COVID_data_event_label)
    #
    # pickle.dump([data_distribution, data_distribution_event_label], open("cn_data_distribution.pickle", "wb"))


   
    # X = torch.from_numpy(data_0_feature).type(torch.float32)

    data_0_feature = torch.from_numpy(data_0_feature)
    data_1_feature = torch.from_numpy(data_1_feature)
   
    MMD_0_1 = MMD(data_0_feature, data_1_feature)
    
    print("MMD distance between 0 and 1 is: ", MMD_0_1)
    # print(a)
    

 


    MMD_distance = MMD_0_1


    source_train_data['event_label'] = 0
    target_train_data['event_label'] = 1
    validation_data['event_label'] = 0
    test_data['event_label'] = 1

    print("Dataset is created!")

    return source_train_data, target_train_data, validation_data, test_data, MMD_distance


if __name__ == '__main__':
    get_data()


