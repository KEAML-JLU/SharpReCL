import os
import json
import random
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import logging, AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import f1_score

from model import TempModel
from data_utils import load_data
from loss import LogitAdjust, SupConLoss, SiamBS_SPM
from utils import sample_generator


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def batch_level_get_cls_num_list(batch_size, num_classes, targets):
    label_one_hot = torch.zeros(batch_size, num_classes).scatter(1, targets.view(-1,1), 1)
    cls_num_list = label_one_hot.sum(dim=0)
    return cls_num_list


def _train(model, dataloader, criterion, optimizer, train_cls_num_list, ratio_h, n_pos=20, n_neg=60, k_pos=20, k_neg=20, alpha=0.5):
    train_loss, n_correct, n_train = 0, 0, 0
    for inputs, targets in tqdm(dataloader, disable=False, ascii=' >='):

        inputs_1, inputs_2 = inputs[0], inputs[1]

        inputs_1 = {k: v.to(torch.device('cuda')) for k, v in inputs_1.items()}
        targets = targets.to(torch.device('cuda'))
        feats_1, proto_feats, logits_1 = model(inputs_1)
        #proto_feats_lis.append(proto_feats)
        # sup_loss = ce_criterion(logits, targets)

        n_correct += (torch.argmax(logits_1, -1) == targets).sum().item()
        n_train += targets.size(0)

        inputs_2 = {k: v.to(torch.device('cuda')) for k, v in inputs_2.items()}
        feats_2, _, logits_2 = model(inputs_2)

        logits = torch.cat([logits_1, logits_2], dim=0)
        labels = torch.cat([targets, targets], dim=0)
        syn_samples = []
        num_classes = int(train_cls_num_list.shape[0])
                          
        for class_index in range(num_classes):
            pos_samples_c = sample_generator(class_index=class_index, feature_1=feats_1, feature_2= feats_2,
                                                      protos=proto_feats, targets=targets, ratio_h=ratio_h,
                                                      n_gen=n_pos, k=k_pos, alpha=alpha, mode=1)
            neg_samples_c = sample_generator(class_index=class_index, feature_1=feats_1, feature_2= feats_2,
                                                      protos=proto_feats, targets=targets, ratio_h=ratio_h,
                                                      n_gen=n_neg, k=k_neg, alpha=alpha, mode=0)
            syn_samples.append(torch.cat([pos_samples_c, neg_samples_c], dim=0).unsqueeze(0))
        syn_samples = torch.cat(syn_samples, dim=0)


        sim_con, labels_con = model.hn_scl(syn_samples=syn_samples, feats_1=feats_1, feats_2=feats_2, protos=proto_feats, targets=targets, 
                                 num_classes=num_classes, n_pos=n_pos, n_neg=n_neg)

        loss_cls, loss_con, loss = criterion(sim_con, labels_con, logits, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * targets.size(0)
        n_correct += (torch.argmax(logits_1, -1) == targets).sum().item()
        n_train += targets.size(0)

        torch.cuda.empty_cache()
        
    return train_loss / n_train, n_correct / n_train

def _test(model, dataloader, criterion):
    test_loss, n_correct, n_test = 0, 0, 0
    model.eval()
    
    pred = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, disable=False, ascii=' >='):
            inputs, _ = inputs[0], inputs[1]
            inputs = {k: v.to(torch.device('cuda')) for k, v in inputs.items()}
            # print(inputs)
            targets = targets.to(torch.device('cuda'))
            feats_1, proto_feats, logits = model(inputs)
            
            loss = criterion(logits, targets)
            
            test_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(logits, -1) == targets).sum().item()
            n_test += targets.size(0)
            torch.cuda.empty_cache()
            
            pred.append(torch.argmax(logits, -1))
            labels.append(targets)
            
    labels = torch.cat(labels, dim=0).cpu().numpy().flatten()
    pred = torch.cat(pred, dim=0).cpu().numpy().flatten()
            
    return test_loss / n_test, n_correct / n_test, f1_score(y_true=labels, y_pred=pred, average='macro')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--dataset", type=str, default="ohsumed", 
                            help="dataset")
    parser.add_argument("--ir", type=int, default=0, 
                            help="imbalance ratio dataset")
    parser.add_argument("--n_pos", type=int, default=0, 
                            help="generation number for positive samples per class")
    parser.add_argument("--n_neg", type=int, default=0, 
                            help="generation number for positive samples per class")
    parser.add_argument("--temperature", type=float, default=0.3, 
                            help="temperature in CL config")
    parser.add_argument("--lr", type=float, default=5e-5, 
                            help="learning rate for the model")
    parser.add_argument("--wd", type=float, default=4e-5, 
                            help="weight decay for the model")
    parser.add_argument("--con_weight", type=float, default=1.0, 
                            help="weight for CL loss")
    parser.add_argument("--batch_size", type=int, default=128, 
                            help="batch size")
    
    
    params = parser.parse_args()
    print(params)
    
    setup_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    base_model = AutoModel.from_pretrained('bert-base-uncased')
    
    params = parser.parse_args()
    dataset = params.dataset
    ir = params.ir
    n_pos = params.n_pos
    n_neg = params.n_neg
    batch_size = params.batch_size

    out_dim = 100
    lamda_0 = 0.5
    # ratio_h = min(lamda_0+(1-lamda_0)*epoch/Epoch, 1)
    # n_pos, n_neg = 20, 60
    k_pos, k_neg = 20, 20
    alpha = 0.5
    Epoch = 30
    temperature = params.temperature
    con_weight = params.con_weight
    effective_num_beta = 0.9999

    lr = params.lr
    weight_decay = params.wd
    
    print(params)
    

    train_dataloader, test_dataloader, train_cls_num_list, test_cls_num_list = load_data(dataset=dataset,
                                      tokenizer=tokenizer,
                                      train_batch_size=batch_size,
                                      test_batch_size=batch_size//4,
                                      model_name='bert',
                                      workers=0,
                                      mode=ir)


    num_classes = int(train_cls_num_list.shape[0])
    
    model = TempModel(base_model=base_model, num_classes=num_classes, feat_dim=out_dim)
    model.to(torch.device('cuda'))
    _params = filter(lambda p: p.requires_grad, model.parameters())

    # adjce_criterion = LogitAdjust(train_cls_num_list)
    ce_criterion = nn.CrossEntropyLoss()

    train_criterion = SiamBS_SPM(train_cls_num_list,
                                     queue_size_per_cls=n_pos+n_neg, temperature=temperature, 
                                     con_weight=con_weight, effective_num_beta=effective_num_beta).cuda()
    # cl_criterion = SupConLoss()
    optimizer = torch.optim.AdamW(_params, lr=lr, weight_decay=weight_decay)
    best_loss, best_acc, best_f1 = 0, 0, 0

    for epoch in range(Epoch):    
        ratio_h = min(lamda_0+(1-lamda_0)*epoch/Epoch, 1)
        # ratio_h = 0
        train_loss, train_acc = _train(model, train_dataloader, train_criterion, optimizer,
                                       train_cls_num_list, ratio_h, n_pos, n_neg, k_pos, k_neg, alpha)
        test_loss, test_acc, test_f1 = _test(model, test_dataloader, ce_criterion)

        if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
            best_loss, best_acc, best_f1 = test_loss, test_acc, test_f1

        print('{}/{} - {:.2f}%'.format(epoch+1, Epoch, 100*(epoch+1)/Epoch))
        print('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
        print('[test] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}'.format(test_loss, test_acc*100, test_f1*100))


    print('best loss: {:.4f}, best acc: {:.2f}, f1: {:.2f}'.format(best_loss, best_acc*100, best_f1*100))
