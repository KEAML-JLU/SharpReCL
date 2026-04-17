import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
import nltk
from tqdm import tqdm
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import torch
from nlpaug.util import Action

class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, method):
        # label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        # sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = list()
        # need add cls_num_list
        cls_num_list = torch.zeros(len(label_dict)).view(-1, 1)
        # print(cls_num_list.shape)
        
        if method == 'train':
        
            for data in raw_data:
                tokens = data['text'].lower().split(' ')
                aug_tokens = data['aug_text'].lower().split(' ')
                label_id = label_dict[data['label']]
                cls_num_list[label_id] += 1
                # dataset.append((label_list + sep_token + tokens, label_id))
                dataset.append((tokens, aug_tokens, label_id))
        
        else:
            
            for data in raw_data:
                tokens = data['text'].lower().split(' ')
                label_id = label_dict[data['label']]
                cls_num_list[label_id] += 1
                # dataset.append((label_list + sep_token + tokens, label_id))
                dataset.append((tokens, label_id))
                
        self._dataset = dataset
        self._cls_num_list = cls_num_list
        

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)
    
    def get_cls_num_list(self):
        return self._cls_num_list


def my_collate(batch, tokenizer, method, num_classes):
    
    if method == 'train':
        tokens, aug_tokens, label_ids = map(list, zip(*batch))
        text_ids = tokenizer(tokens,
                             padding=True,
                             truncation=True,
                             max_length=256,
                             is_split_into_words=True,
                             add_special_tokens=True,
                             return_tensors='pt')
        aug_text_ids = tokenizer(aug_tokens,
                             padding=True,
                             truncation=True,
                             max_length=256,
                             is_split_into_words=True,
                             add_special_tokens=True,
                             return_tensors='pt')
    else:
        tokens, label_ids = map(list, zip(*batch))
        text_ids = tokenizer(tokens,
                             padding=True,
                             truncation=True,
                             max_length=256,
                             is_split_into_words=True,
                             add_special_tokens=True,
                             return_tensors='pt')
        aug_text_ids = {}

        
    return [text_ids, aug_text_ids], torch.tensor(label_ids)


def load_data(dataset, tokenizer, train_batch_size, test_batch_size, model_name, workers,
              data_dir='./data', mode=0):
    
    train_dir = data_dir
    if(mode != 0):
        train_dir += '/imb/' + str(mode)
    
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(train_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'TREC':
        train_data = json.load(open(os.path.join(train_dir, 'TREC_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'CR':
        train_data = json.load(open(os.path.join(train_dir, 'CR_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'SUBJ':
        train_data = json.load(open(os.path.join(train_dir, 'SUBJ_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(train_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'StackOverflow':
        train_data = json.load(open(os.path.join(train_dir, 'StackOverflow_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'StackOverflow_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'1': 0, '19': 1, '9': 2, '2': 3, '20': 4, '7': 5, '16': 6, '6': 7, '8': 8, '5': 9,
                     '11': 10, '18': 11, '15': 12, '17': 13, '10': 14, '3': 15, '13': 16, '4': 17, '12': 18, '14': 19}
    elif dataset == 'ohsumed':
        train_data = json.load(open(os.path.join(train_dir, 'ohsumed_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'ohsumed_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'C09': 0, 'C01': 1, 'C19': 2, 'C11': 3, 'C14': 4, 'C17': 5, 'C04': 6, 'C23': 7, 'C10': 8, 'C20': 9, 'C02': 10,
                     'C06': 11, 'C22': 12, 'C12': 13, 'C08': 14, 'C07': 15, 'C18': 16, 'C03': 17, 'C16': 18, 'C15': 19, 'C21': 20,
                     'C05': 21, 'C13': 22}
    elif dataset == 'snipptes':
        train_data = json.load(open(os.path.join(train_dir, 'snipptes_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'snipptes_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'0': 0, '1': 1, '7': 2, '3': 3, '5': 4, '6': 5, '4': 6, '2': 7}
        
    elif dataset == 'TagMyNews':
        train_data = json.load(open(os.path.join(train_dir, 'TagMyNews_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TagMyNews_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'sci_tech': 0, 'health': 1, 'sport': 2, 'world': 3, 'business': 4, 'us': 5, 'entertainment': 6}
        
    elif dataset == 'dblp':
        train_data = json.load(open(os.path.join(train_dir, 'dblp_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'dblp_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'1': 0, '3': 1, '5': 2, '4': 3, '6': 4, '2': 5}
        
    elif dataset == 'Biomedical':
        train_data = json.load(open(os.path.join(train_dir, 'Biomedical_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'Biomedical_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'1': 0, '19': 1, '9': 2, '2': 3, '20': 4, '7': 5, '16': 6, '6': 7, '8': 8,'5': 9, '11': 10,
                      '18': 11, '15': 12, '17': 13, '10': 14, '3': 15, '13': 16, '4': 17, '12': 18, '14': 19}
        
    elif dataset == 'chemprot':
        train_data = json.load(open(os.path.join(train_dir, 'chemprot_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'chemprot_Test.json'), 'r', encoding='utf-8'))
        label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        
        
    elif dataset == 'mr':
        train_data = json.load(open(os.path.join(train_dir, 'mr_Train_aug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'mr_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'0': 0, '1': 1}
        
    
    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, method='train')
    # print(trainset)
    train_cls_num_list = trainset.get_cls_num_list()
    testset = MyDataset(test_data, label_dict, method='test')
    test_cls_num_list = testset.get_cls_num_list()
    collate_fn_train = partial(my_collate, tokenizer=tokenizer, method='train', num_classes=len(label_dict))
    collate_fn_test = partial(my_collate, tokenizer=tokenizer, method='test', num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn_train, pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn_test, pin_memory=True)
    return train_dataloader, test_dataloader, train_cls_num_list, test_cls_num_list


def text_aug(dataset_name, base_dir='./data/balanced/'):
    
    src_file = base_dir + dataset_name + '_Train.json'
    tgt_file = base_dir + dataset_name +'_Train_aug.json'
    
    origin_data = json.load(open(src_file, 'r', encoding='utf-8'))
    # origin_data_test = json.load(open(src_file, 'r', encoding='utf-8'))['test']
    
    print('running ', dataset_name, '.......')
    lines_in = [item['text'] for item in origin_data]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    batch_size = 128
    
    augmenter = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", \
            aug_min=1, aug_p=0.2, device="cuda")
    
    """
    for i in tqdm(range(0, len(lines_in), batch_size)):
        lines_batch = lines_in[i:i+batch_size]
        for j, p in enumerate(lines_batch):
            aug_text = random.choice(augmenter.augment(data=p, n=10))
            origin_data[str(i+j)]['aug_text'] = aug_text
        del lines_batch
        del aug_text

    for i in tqdm(range(0, len(lines_in_test), batch_size)):
        lines_batch = lines_in_test[i:i+batch_size]
        for j, p in enumerate(lines_batch):
            aug_text = random.choice(augmenter.augment(data=p, n=10))
            origin_data_test[str(i+j)]['aug_text'] = aug_text
        del lines_batch
        del aug_text
    """
    for i in tqdm(range(0, len(lines_in), batch_size)):
        lines_batch = lines_in[i:i+batch_size]
        aug_text = augmenter.augment(data=lines_batch)
        for j, p in enumerate(aug_text):
            origin_data[str(i+j)]['aug_text'] = p
        del lines_batch
        del aug_text

        
    json.dump(origin_data, open(tgt_file, 'w'),
              ensure_ascii=False)

    
    print(dataset_name, 'done!')
