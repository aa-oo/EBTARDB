
from sklearn.preprocessing import LabelEncoder

from rand5fold_early import *

import pandas as pd
import torch

from torch.nn import CrossEntropyLoss, MSELoss

from utils import get_split, get_natural_split, get_fixed_split, get_split_100
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from domain_discriminator import *
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    BertweetTokenizer,
    AutoModel,
    AutoTokenizer
)

from transformers import glue_convert_examples_to_features as convert_examples_to_features

from transformers.data.processors.utils import InputExample, DataProcessor
from evaluate import *
import logging
#%%

label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }
def get_predata(datasetname,datalist,data_division):
    raw_data = pd.read_csv('./data/raw_data_'+datasetname+'.csv') 

    raw_data.sort_values(by='count', inplace=True)

    fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = datalist[0],datalist[1],datalist[2],datalist[3]


    # raw_data['label'] = LabelEncoder().fit_transform(raw_data['label'])
    raw_data['label'] = raw_data['label'].apply(lambda x: label2id[x])
    print(raw_data)
    raw_data_test = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_test)]
    raw_data_test_100 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_100_train)]
    raw_data_train_8 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_8)]
    raw_data_train_2 = raw_data[raw_data.id.apply(lambda x: str(x) in fold0_x_train_2)]
    print(len(raw_data_test), len(raw_data_test_100), len(raw_data_train_2), len(raw_data_train_8))

    raw_data_test = raw_data_test[['text_comments', 'label']]
    raw_data_test = raw_data_test.rename(columns={'text_comments': 'text'})
    raw_data_test_100 = raw_data_test_100[['text_comments', 'label']]
    raw_data_test_100 = raw_data_test_100.rename(columns={'text_comments': 'text'})
    raw_data_train_2 = raw_data_train_2[['text_comments', 'label']]
    raw_data_train_2 = raw_data_train_2.rename(columns={'text_comments': 'text'})
    raw_data_train_8 = raw_data_train_8[['text_comments', 'label']]
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

    # data_train = pd.concat([data_train_8, data_train_2], axis=0)
    data_test = pd.concat([data_test,data_test_100], axis=0)
    data_test.reset_index(drop=True, inplace=True)
    data_test_100.reset_index(drop=True, inplace=True)
    data_train_8.reset_index(drop=True, inplace=True)
    data_train_2.reset_index(drop=True, inplace=True)
    data_train=data_train_8
    data_train.reset_index(drop=True, inplace=True)

    train_tmp = data_train.copy()
    train_tmp['text_split'] = data_train['text'].apply(get_split)
    # train_tmp['text_split'] = train['text'].apply(get_fixed_split)
    # train_tmp['text_split'] = train['text'].apply(get_natural_split)
    train = train_tmp
    train.head()

    val_tmp = data_test.copy()
    val_tmp_100 = data_test_100.copy()


    if data_division=='random':
        val_tmp['text_split'] = data_test['text'].apply(get_split)
        val_tmp_100['text_split'] = data_test_100['text'].apply(get_split)
    elif data_division=='event':
        val_tmp['text_split'] = data_test['text'].apply(get_split_100)
        val_tmp_100['text_split'] = data_test_100['text'].apply(get_split_100)


    val = val_tmp
    val_100 = val_tmp_100

    train_l = []  # Segmented Text
    label_l = []  # Label of Each Text
    index_l = []  # The Index of Each Text Before Segmentation
    for idx, row in train.iterrows():
        for l in row['text_split']:
            train_l.append(l)
            label_l.append(row['label'])
            index_l.append(idx)
    len(train_l), len(label_l), len(index_l)
    val_l = []
    val_label_l = []
    val_index_l = []
    for idx, row in val.iterrows():
        for l in row['text_split']:
            val_l.append(l)
            val_label_l.append(row['label'])
            val_index_l.append(idx)
    len(val_l), len(val_label_l), len(val_index_l)
    val_l_100 = []
    val_label_l_100 = []
    val_index_l_100 = []
    for idx, row in val_100.iterrows():
        for l in row['text_split']:
            val_l_100.append(l)
            val_label_l_100.append(row['label'])
            val_index_l_100.append(idx)
    len(val_l_100), len(val_label_l_100), len(val_index_l_100)
  
    train_df = pd.DataFrame({'text': train_l, 'label': label_l})
    train_df.head()

    val_df = pd.DataFrame({'text': val_l, 'label': val_label_l})
    val_df.head()

    val_df_100 = pd.DataFrame({'text': val_l_100, 'label': val_label_l_100})
    val_df_100.head()
   
    train_InputExamples = train_df.apply(
        lambda x: InputExample(guid=None, text_a=x['text'], text_b=None, label=x['label']),
        axis=1)

    val_InputExamples = val_df.apply(lambda x: InputExample(guid=None, text_a=x['text'], text_b=None, label=x['label']),
                                     axis=1)

    val_InputExamples_100 = val_df_100.apply(
        lambda x: InputExample(guid=None, text_a=x['text'], text_b=None, label=x['label']), axis=1)
    return train_InputExamples,val_InputExamples,val_InputExamples_100


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output, sequence_output,)

        if labels is not None:

            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # loss, logits, pooled_output, sequence_output



def get_config(args):
   
    MODEL_CLASSES = {
        "bert": (BertConfig, BertTokenizer),
        "bertweet": (BertConfig, BertweetTokenizer)
    }
    ### Step 3.1 : Load Pre-training Models & Prepare Training Data
  
    # Load Pre-training Models


    config_class, tokenizer_class = MODEL_CLASSES["bert"]
    model_class = BertForClassification

    config = config_class.from_pretrained(
        args["config_name"],
        finetuning_task="",
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args["tokenizer_name"],
        do_lower_case=True,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
        args["model_name_or_path"],
        from_tf=bool(".ckpt" in args["model_name_or_path"]),
        config=config,
        cache_dir=None,
    )
    return model,tokenizer,args

def get_data(train_InputExamples,val_InputExamples,val_InputExamples_100,tokenizer,my_label_list,MAX_SEQ_LENGTH):
    train_features = convert_examples_to_features(train_InputExamples, tokenizer, label_list=my_label_list,
                                                  output_mode="classification", max_length=MAX_SEQ_LENGTH)

    train_features_100 = convert_examples_to_features(val_InputExamples_100, tokenizer, label_list=my_label_list,
                                                      output_mode="classification", max_length=MAX_SEQ_LENGTH)
    val_features = convert_examples_to_features(val_InputExamples, tokenizer, label_list=my_label_list,
                                                output_mode="classification", max_length=MAX_SEQ_LENGTH)

 
    input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    the_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, the_labels)

    input_ids_100 = torch.tensor([f.input_ids for f in train_features_100], dtype=torch.long)
    attention_mask_100 = torch.tensor([f.attention_mask for f in train_features_100], dtype=torch.long)
    token_type_ids_100 = torch.tensor([f.token_type_ids for f in train_features_100], dtype=torch.long)
    the_labels_100 = torch.tensor([f.label for f in train_features_100], dtype=torch.long)

    dataset_100 = TensorDataset(input_ids_100, attention_mask_100, token_type_ids_100, the_labels_100)


    val_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
    val_attention_mask = torch.tensor([f.attention_mask for f in val_features], dtype=torch.long)
    val_token_type_ids = torch.tensor([f.token_type_ids for f in val_features], dtype=torch.long)
    val_the_labels = torch.tensor([f.label for f in val_features], dtype=torch.long)

    eval_dataset = TensorDataset(val_input_ids, val_attention_mask, val_token_type_ids, val_the_labels)
    return dataset,dataset_100,eval_dataset


def train(train_dataset, train_dataset_100, model, tokenizer,args,seed,data_division,datasetname):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    t_total = len(train_dataset) // 5
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # *********************
    logger.info("*****Running training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", 5)

    epochs_trained = 0
    global_step = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, 5, desc="Epoch", disable=False)

    for k in train_iterator:  # 5 epoch

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False) #306

        train_sampler_100 = RandomSampler(train_dataset_100)
        train_dataloader_100 = DataLoader(train_dataset_100, sampler=train_sampler_100, batch_size=16)
        epoch_iterator_100 = tqdm(train_dataloader_100, desc="Iteration", disable=False) #11

        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            src_label_loss = outputs[0]



           
            loss=src_label_loss
            
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        logger.info("average loss:" + str(tr_loss / global_step))

    model.save_pretrained("./model_all_domain/"+datasetname+"/"+data_division+"/70_30_classification_models_" + model_path+'_'+seed)
    tokenizer.save_pretrained("./model_all_domain/"+datasetname+"/"+data_division+"/70_30_classification_models_" + model_path+'_'+seed)

    torch.save(args, os.path.join("./model_all_domain/"+datasetname+"/"+data_division+"/70_30_classification_models_" + model_path+'_'+seed, "training_args.bin"))
    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluate(
        model, tokenizer, eval_dataset)
    return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4


def evaluate(model, tokenizer, eval_dataset):
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 16)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=16)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)
    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
        preds,out_label_ids)
    return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return acc, f1


def setup_seed(seed):
    th.manual_seed(seed) 
    th.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

model_path = 'text_comments'
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

my_label_list = [0, 1,2,3]

MAX_SEQ_LENGTH = 200
args = {"model_name_or_path": "bert-base-uncased",
        "config_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
        }
# args_eval = {"model_name_or_path": "./trained_models/classification_models_" + model_path,
#              "config_name": "./trained_models/classification_models_" + model_path,
#              "tokenizer_name": "./trained_models/classification_models_" + model_path,
#              }
def get_args(seed):
    args_eval = {"model_name_or_path": "./model_all_domain/classification_models_" + model_path+'_'+seed,
             "config_name": "./model_all_domain/classification_models_" + model_path+'_'+seed,
             "tokenizer_name": "./model_all_domain/classification_models_" + model_path+'_'+seed,
             }
    return args_eval


iterations=1
# datasetname='Twitter15'
datasetname=sys.argv[3]
if datasetname == 'Twitter16':
    data_path = './data/twitter16/'
    laebl_path = './data/Twitter16_label_All.txt'
elif datasetname == 'Twitter15':
    data_path = './data/twitter15/'
    laebl_path = './data/Twitter15_label_All.txt'
    
# type='train'
# data_division='random'
data_division=sys.argv[2]
type=sys.argv[1]
if type == 'train' and data_division=='event':
    print('TRAIN')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    th.backends.cudnn.enabled = False
    for iter in range(iterations):
        setup_seed(20)
        print('seed=20, t=0.8')

        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        fold0_x_train_8.extend(fold0_x_train_2)
        datalist = [ fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        # torch.cuda.empty_cache()
        accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0, acc4_0, pre4_0, rec4_0, F4_0 =train(dataset, dataset_100, model, tokenizer, args1,'20',data_division,datasetname)

        setup_seed(30)
        print('seed=30, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        fold0_x_train_8.extend(fold0_x_train_2)
        datalist = [ fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)


        accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 ,acc3_1, pre3_1, rec3_1, F3_1, acc4_1, pre4_1, rec4_1, F4_1  = train(dataset, dataset_100, model, tokenizer, args1,'30',data_division,datasetname)
        
        setup_seed(40)
        print('seed=40, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,laebl_path,3)
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8), len(fold0_x_train_2), len(fold0_x_100_train))
        fold0_x_train_8.extend(fold0_x_train_2)
        datalist = [ fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 ,acc3_2, pre3_2, rec3_2, F3_2, acc4_2, pre4_2, rec4_2, F4_2= train(dataset, dataset_100, model, tokenizer, args1,'40',data_division,datasetname)
        
        setup_seed(50)
        print('seed=50, t=0.8')
        fold0_x_train_8.extend(fold0_x_train_2)
        datalist = [ fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)
        accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3, acc3_3, pre3_3, rec3_3, F3_3, acc4_3, pre4_3, rec4_3, F4_3= train(dataset, dataset_100, model, tokenizer, args1,'50',data_division,datasetname)
        
        setup_seed(60)
        print('seed=60, t=0.8')
        fold0_x_train_8.extend(fold0_x_train_2)
        datalist = [ fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4, acc4_4, pre4_4, rec4_4, F4_4 = train(dataset, dataset_100, model, tokenizer, args1,'60',data_division,datasetname)

        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4) / 5)
        PRE4.append((pre4_0 + pre4_1 + pre4_2 + pre4_3 + pre4_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4) / 5)
        REC4.append((rec4_0 + rec4_1 + rec4_2 + rec4_3 + rec4_4) / 5)
        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        F4.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

    print(
        "Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}|pre4: {:.4f}"
        "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|rec4: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|F4: {:.4f}".format(
            sum(test_accs) / iterations, sum(ACC1) / iterations,
            sum(ACC2) / iterations, sum(ACC3) / iterations,
            sum(ACC4) / iterations, sum(PRE1) / iterations,
            sum(PRE2) / iterations, sum(PRE3) / iterations,
            sum(PRE4) / iterations,
            sum(REC1) / iterations, sum(REC2) / iterations,
            sum(REC3) / iterations, sum(REC4) / iterations,
            sum(F1) / iterations, sum(F2) / iterations,
            sum(F3) / iterations, sum(F4) / iterations))


elif type == 'train' and data_division=='random':
    print('TRAIN')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    th.backends.cudnn.enabled = False
    for iter in range(iterations):
        setup_seed(20)
        print('seed=20, t=0.8')

        fold0_test,fold0_val,fold0_train,\
            fold1_test,fold1_val,fold1_train,\
            fold2_test,fold2_val,fold2_train,\
            fold3_test,fold3_val,fold3_train,\
            fold4_test, fold4_val,fold4_train= load5foldData(datasetname,data_path,laebl_path,4)
    
        
        print('fold0 shape: ', len(fold0_test), len(fold0_train),len(fold0_val))
        print('fold1 shape: ', len(fold1_test), len(fold1_train),len(fold1_val))
        print('fold2 shape: ', len(fold2_test), len(fold2_train),len(fold2_val))
        print('fold3 shape: ', len(fold3_test), len(fold3_train),len(fold3_val))
        print('fold4 shape: ', len(fold4_test), len(fold4_train),len(fold4_val))
        datalist = [fold0_test,fold0_train,fold0_val,fold0_val]

        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        # torch.cuda.empty_cache()
        accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0, acc4_0, pre4_0, rec4_0, F4_0 =train(dataset, dataset_100, model, tokenizer, args1,'20',data_division,datasetname)

       
        print('seed=30, t=0.8')
        datalist = [fold1_test,fold1_train,fold1_val,fold1_val]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)


        accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 ,acc3_1, pre3_1, rec3_1, F3_1, acc4_1, pre4_1, rec4_1, F4_1  = train(dataset, dataset_100, model, tokenizer, args1,'30',data_division,datasetname)
        
        
        print('seed=40, t=0.8')
        datalist = [fold2_test,fold2_train,fold2_val,fold2_val]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 ,acc3_2, pre3_2, rec3_2, F3_2, acc4_2, pre4_2, rec4_2, F4_2= train(dataset, dataset_100, model, tokenizer, args1,'40',data_division,datasetname)
        
        
        print('seed=50, t=0.8')
        datalist = [fold3_test,fold3_train,fold3_val,fold3_val]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)
        accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3, acc3_3, pre3_3, rec3_3, F3_3, acc4_3, pre4_3, rec4_3, F4_3= train(dataset, dataset_100, model, tokenizer, args1,'50',data_division,datasetname)
        
        
        print('seed=60, t=0.8')
        datalist = [fold4_test,fold4_train,fold4_val,fold4_val]
        train_InputExamples, val_InputExamples, val_InputExamples_100 = get_predata(datasetname,datalist,data_division)
        model, tokenizer, args1 = get_config(args)
        model.to(device)
        dataset, dataset_100, eval_dataset = get_data(train_InputExamples, val_InputExamples, val_InputExamples_100,
                                                      tokenizer, my_label_list, MAX_SEQ_LENGTH)

        accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4, acc4_4, pre4_4, rec4_4, F4_4 = train(dataset, dataset_100, model, tokenizer, args1,'60',data_division,datasetname)

        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
        ACC4.append((acc4_0 + acc4_1 + acc4_2 + acc4_3 + acc4_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4) / 5)
        PRE4.append((pre4_0 + pre4_1 + pre4_2 + pre4_3 + pre4_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4) / 5)
        REC4.append((rec4_0 + rec4_1 + rec4_2 + rec4_3 + rec4_4) / 5)
        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        F4.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

    print(
        "Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|acc4: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}|pre4: {:.4f}"
        "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|rec4: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|F4: {:.4f}".format(
            sum(test_accs) / iterations, sum(ACC1) / iterations,
            sum(ACC2) / iterations, sum(ACC3) / iterations,
            sum(ACC4) / iterations, sum(PRE1) / iterations,
            sum(PRE2) / iterations, sum(PRE3) / iterations,
            sum(PRE4) / iterations,
            sum(REC1) / iterations, sum(REC2) / iterations,
            sum(REC3) / iterations, sum(REC4) / iterations,
            sum(F1) / iterations, sum(F2) / iterations,
            sum(F3) / iterations, sum(F4) / iterations))



elif type == 'test_data':
    print('test_data')
    test_accs, ACC1, ACC2,ACC3, ACC4, PRE1, PRE2,PRE3, PRE4, REC1, REC2,REC3, REC4, F1, F2 ,F3, F4 = [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []

    th.backends.cudnn.enabled = False
    len_train=0
    len_test=0
    for iter in range(iterations):
        l1=l2=l3=l4=0
        l5=l6=l7=l8=0
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['non-rumor'], ['false'], ['true'], ['unverified']
        if datasetname=='Twitter15':
            label_path = './data/Twitter15_label_All.txt'
            data_path = './data/twitter15/'
            label_path1 = './data/label_15.json'
        elif datasetname=='Twitter16':
            label_path = './data/Twitter16_label_All.txt'
            data_path = './data/twitter16/'
            label_path1 = './data/label_16.json'
        with open(label_path1, encoding='utf-8') as f:
            json_inf = json.load(f)
        print('The len of file_list: ', len(json_inf))
        setup_seed(20)
        print('seed=20, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,label_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
        setup_seed(30)
        print('seed=30, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,label_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
       
        setup_seed(40)
        print('seed=40, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,label_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))

        setup_seed(50)
        print('seed=50, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,label_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))
    
        setup_seed(60)
        print('seed=60, t=0.8')
        fold0_x_test, fold0_x_train_8, fold0_x_train_2, fold0_x_100_train = load5foldData(datasetname,data_path,label_path,3)
        fold0_x_train_8.extend(fold0_x_train_2)
        fold0_x_test.extend(fold0_x_100_train)
        for x in fold0_x_train_8:
            label=json_inf[x]
            if label in labelset_nonR: 
                l1 += 1
            if label in labelset_f: # F
                l2 += 1
            if label in labelset_t: # T
                l3 += 1
            if label in labelset_u: # U
                l4 += 1
        for x in fold0_x_test:
            label=json_inf[x]
            if label in labelset_nonR: 
                l5 += 1
            if label in labelset_f: # F
                l6 += 1
            if label in labelset_t: # T
                l7 += 1
            if label in labelset_u: # U
                l8 += 1
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_8))

        l1=l1/5
        l2=l2/5
        l3=l3/5
        l4=l4/5
        l5=l5/5
        l6=l6/5
        l7=l7/5
        l8=l8/5
        print(round(l1),round(l2),round(l4),round(l3))
        print(round(l5),round(l6),round(l8),round(l7))







