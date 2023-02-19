#!/usr/bin/env python
# coding: utf-8

# # Fine tune with MAG dataset by resampling


import torch
import pickle
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from utils.data_helper import read_mag_file
from utils.lazydataset import LazyTextMAG_Dataset
from utils.evaluate import evaluate

#=======================================================Edit your configeration=========================================================
RS_EXP = 0.3
RW_EXP = 0
# Model_path indicates the checkpoint path to be loaded. If the value is empty, the original BERT model is used
Model_path = 'some_model_saved_by_further_pretrain.py'
EPOCH_NUM = 100
Train_batch_size = 128
Validation_batch_size = 128
SAVE_CHECKPOINT_PATH = './checkpoint/'
WARMUP_STEP = 10000
TOTAL_STEP = 500000
Learning_rate = 5e-5
DEVICE_NAME = "cuda:0"
#===========================================================End Configeration===========================================================


start_time = int(time.time())
log_time = start_time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=str(log_time) + ".log"
)
logger = logging.getLogger(__name__)

logger.info("***** Running Fine-tuning *****")
logger.info("=====================Configeration===================")
logger.info(f"RS_EXP:{RS_EXP}")
logger.info(f"RW_EXP:{RW_EXP}")
logger.info(f"Model_path:{Model_path}")
logger.info(f"EPOCH_NUM:{EPOCH_NUM}")
logger.info(f"Train_batch_size:{Train_batch_size}")
logger.info(f"Validation_batch_size:{Validation_batch_size}")
logger.info(f"SAVE_CHECKPOINT_PATH:{SAVE_CHECKPOINT_PATH}")
logger.info(f"Learning_rate:{Learning_rate}")
logger.info("=================End Configeration===================")

save_pkl_root = '/home/datamerge/ACL/Data/210422/pkl/'
save_train_root = '/home/datamerge/ACL/Data/210422/train/'
save_test_root = '/home/datamerge/ACL/Data/210422/test/'
save_open_root = '/home/datamerge/ACL/Data/210422/open/'
save_dev_root = '/home/datamerge/ACL/Data/210422/dev/'

afid2nor = pickle.load(open(save_pkl_root+"afid2nor.pkl", "rb"))
nor2afid = pickle.load(open(save_pkl_root+"nor2afid.pkl", "rb"))


train_mid2label_dict = pickle.load(open(save_pkl_root+'train_mid2label_dict.pkl', 'rb'))
train_label2mid_dict = pickle.load(open(save_pkl_root+'train_label2mid_dict.pkl', 'rb'))


train_filepath = save_train_root+'train_part.txt'
dev_filepath = save_dev_root+'dev.txt'
test_filepath = save_test_root+'test.txt'


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_dataset = LazyTextMAG_Dataset(tokenizer, train_filepath, train_label2mid_dict)
dev_dataset = LazyTextMAG_Dataset(tokenizer, dev_filepath, train_label2mid_dict)
test_dataset = LazyTextMAG_Dataset(tokenizer, test_filepath, train_label2mid_dict)


device = torch.device(DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')

class BertForAffiliationNameNormalization(torch.nn.Module):
    
    def __init__(self, num_of_classes):
        super(BertForAffiliationNameNormalization, self).__init__()
        self.num_of_classes = num_of_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.dropout = nn.Dropout(p=0.1, inplace=False).to(device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_of_classes, bias=True).to(device)
        
        
    def forward(self, input_ids, attention_mask):
        pooled_out = self.bert(input_ids, attention_mask=attention_mask)
        pooled_out = self.dropout(pooled_out[1])
        logits = self.classifier(pooled_out)     
        
        return logits


NUM_OF_CLASS = len(train_mid2label_dict)
if Model_path == '':
    model = BertForAffiliationNameNormalization(NUM_OF_CLASS)
else:
    model = torch.load(Model_path)

if isinstance(model,torch.nn.DataParallel):
	model = model.module
model.to(device)
# model = nn.DataParallel(model, device_ids=[0,1,2,3])



device = torch.device(DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')


class_len_l = pickle.load(open(save_pkl_root+'train_len_l.pkl', 'rb'))
nor2len_train_part_dict = pickle.load(open(save_pkl_root+'nor2len_train_part_dict.pkl', 'rb'))
nor2len_dict = pickle.load(open(save_pkl_root+'210422_nor2len_dict.pkl', 'rb'))
rs_weights = np.array([ 1./class_len**RS_EXP for class_len in class_len_l])
rw_weights = [(1./nor2len_train_part_dict[train_mid2label_dict[i]])**(RW_EXP) for i in range(len(train_mid2label_dict))]

if RS_EXP>=1.0 or RS_EXP<0:
    train_loader = DataLoader(train_dataset, batch_size=Train_batch_size) 
else:
    samples_weight = torch.from_numpy(rs_weights)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=Train_batch_size, sampler = sampler) 


optim = AdamW(model.parameters(), lr=Learning_rate)
num_warmup_steps = WARMUP_STEP
num_total_steps = TOTAL_STEP

# scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps, -1)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_total_steps, -1)

writer = SummaryWriter('./tensorboard')

completed_step = 0
for epoch in range(EPOCH_NUM):
    for i,batch in tqdm(enumerate(train_loader)):
        model.train()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids, attention_mask=attention_mask)   
        loss = loss_func(logits.view(-1, NUM_OF_CLASS), labels.view(-1))
        writer.add_scalar('train/loss_step', loss, completed_step)
        writer.add_scalar('train/lr_step', optim.param_groups[0]["lr"], completed_step)
        loss.backward()
        optim.step()
        scheduler.step()
        
        if i%100 == 0:
            print("Epoch: ", epoch, " , step: ", i)
            print("training loss: ", loss.item())
            print("training lr: ", optim.param_groups[0]["lr"])
            logger.info(f"step:{completed_step}, training loss:{loss.item()}")
            logger.info(f"step:{completed_step}, training lr:{optim.param_groups[0]['lr']}")
            
        completed_step += 1
    writer.add_scalar('train/loss_epoch', loss, completed_step)

    dev_overall, dev_part = evaluate(model, dev_dataset, nor2len_dict, train_mid2label_dict, Validation_batch_size, DEVICE_NAME)
    dev_overall_acc, dev_overall_p, dev_overall_r, dev_overall_f1 = dev_overall['accuracy'], dev_overall['precision'], dev_overall['recall'], dev_overall['f1']
    dev_high_acc, dev_high_p, dev_high_r, dev_high_f1 = dev_part['high']['accuracy'], dev_part['high']['precision'], dev_part['high']['recall'], dev_part['high']['f1']
    dev_middle_acc, dev_middle_p, dev_middle_r, dev_middle_f1 = dev_part['middle']['accuracy'], dev_part['middle']['precision'], dev_part['middle']['recall'], dev_part['middle']['f1']
    dev_few_acc, dev_few_p, dev_few_r, dev_few_f1 = dev_part['few']['accuracy'], dev_part['few']['precision'], dev_part['few']['recall'], dev_part['few']['f1']

    logger.info(f"Epoch: {epoch}")
    logger.info("Validation info:")
    logger.info(dev_overall)
    logger.info(dev_part)

    writer.add_scalar('overall/dev_overall_acc', dev_overall_acc, completed_step)
    writer.add_scalar('overall/dev_overall_p', dev_overall_p, completed_step)
    writer.add_scalar('overall/dev_overall_r', dev_overall_r, completed_step)
    writer.add_scalar('overall/dev_overall_f1', dev_overall_f1, completed_step)

    writer.add_scalar('high/dev_high_acc', dev_high_acc, completed_step)
    writer.add_scalar('high/dev_high_p', dev_high_p, completed_step)
    writer.add_scalar('high/dev_high_r', dev_high_r, completed_step)
    writer.add_scalar('high/dev_high_f1', dev_high_f1, completed_step)

    writer.add_scalar('middle/dev_middle_acc', dev_middle_acc, completed_step)
    writer.add_scalar('middle/dev_middle_p', dev_middle_p, completed_step)
    writer.add_scalar('middle/dev_middle_r', dev_middle_r, completed_step)
    writer.add_scalar('middle/dev_middle_f1', dev_middle_f1, completed_step)

    writer.add_scalar('few/dev_few_acc', dev_few_acc, completed_step)
    writer.add_scalar('few/dev_few_p', dev_few_p, completed_step)
    writer.add_scalar('few/dev_few_r', dev_few_r, completed_step)
    writer.add_scalar('few/dev_few_f1', dev_few_f1, completed_step)

    torch.save(model, SAVE_CHECKPOINT_PATH+'/epoch_' + str(epoch) + '_bert.pkl')  