import torch
import numpy as np
import torch.nn.functional as F

def get_batch_cls_num_list(targets, num_classes):
    label_one_hot = torch.zeros(len(targets), num_classes).scatter(1, targets.view(-1,1), 1)
    cls_num_list = label_one_hot.sum(dim=0)
    return cls_num_list


def sample_generator(class_index, feature_1, feature_2, protos, targets, ratio_h, n_gen, k, alpha=0.5, mode=0):
    
    device = (torch.device('cuda')
                  if feature_1.is_cuda
                  else torch.device('cpu'))
    # mode == 1:pos, mode==0, neg
    
    num_classes = protos.shape[0]
    proto_targets = torch.arange(num_classes).to(device)
    # print(targets.shape, proto_targets.shape)
    labels = torch.cat([targets, targets, proto_targets], dim=0)
    feats = torch.cat([feature_1, feature_2, protos], dim=0)
    feats = F.normalize(feats, dim=1)
    protos = F.normalize(protos, dim=1)
    
    if(mode == 1):
        cls_ind = torch.where(labels == class_index)[0]
    else:
        cls_ind = torch.where(labels != class_index)[0]
        
    cls_feat = feats[cls_ind]
    cls_sim = torch.matmul(protos[class_index], cls_feat.T)
    
    if(mode == 1):
        cls_sim = -cls_sim
    
    n_hard = min(k ,len(cls_ind))
    _, idx_hard = torch.topk(cls_sim, k=n_hard, dim=-1, sorted=False)
    
    hard_num = int(ratio_h*n_gen)

    idx_1, idx_2 = torch.randint(n_hard, size=(2, hard_num)).to(device)

    candidate_1 = cls_feat[torch.gather(idx_hard, dim=0, index=idx_1)]
    candidate_2 = cls_feat[torch.gather(idx_hard, dim=0, index=idx_2)]

    beta_lamda = torch.tensor(np.random.beta(a=alpha, b=alpha, size=(len(candidate_1), 1)).astype(np.float32)).to(device)
    hard_samples = beta_lamda*candidate_1 + (1-beta_lamda)*candidate_2
    hard_samples = F.normalize(hard_samples, dim=1)
    
    easy_num = n_gen - hard_num

    eazy_idx = torch.randint(len(cls_ind), size=(1, easy_num))[0].to(device)

    # easy_samples = feats[torch.gather(torch.arange(len(cls_ind)), dim=0, index=eazy_idx)]
    easy_samples = feats[torch.gather(cls_ind, dim=0, index=eazy_idx)]
    
    # return hard_samples, easy_samples
    return torch.cat([hard_samples, easy_samples])

def hard_neg_generater(feat, orig_targets, protos):
    feat = feat.cpu()
    orig_targets = orig_targets.cpu()
    protos = protos.cpu()
    batch_size = feat.shape[0]
    num_classes = protos.shape[0]
    orig_targets = orig_targets.view(-1, 1)
    
    feature = F.normalize(feat, dim=1)
    protos = F.normalize(protos, dim=1)
    proto_targets = torch.arange(num_classes).view(-1, 1)
    
    all_targets = torch.cat([orig_targets, proto_targets], dim=0)
    all_feats = torch.cat([feature, protos], dim=0)
    cls_num_list_b = get_batch_cls_num_list(all_targets, num_classes)
    
    hn_feats = []
    hn_targets = []
    
    for src in range(num_classes):
        num_to_gen = (max(cls_num_list_b) - cls_num_list_b[src])/(num_classes-1)
        num_to_gen = int(num_to_gen/2)+1
        if(num_to_gen == 0):
            continue
        
        src_idx = torch.arange(batch_size+num_classes).view(-1,1)[all_targets == src]
        src_feats = all_feats[src_idx]
        sim_src_proto = torch.matmul(protos, src_feats.T)
        
        for tgt in range(num_classes):
            
            if tgt == src:
                continue
                
            n_hard = int(min(cls_num_list_b[src], 32))
            _, idx_hard = torch.topk(sim_src_proto[tgt], k=n_hard, dim=-1, sorted=False)
            idx_1, idx_2 = torch.randint(n_hard, size=(2, num_to_gen))
            neg1_hard = src_feats[torch.gather(idx_hard, dim=0, index=idx_1)].clone().detach()
            neg2_hard = src_feats[torch.gather(idx_hard, dim=0, index=idx_2)].clone().detach()
            alpha = torch.rand((num_to_gen, 1))
            hard_negative = alpha*neg1_hard + (1-alpha)*neg2_hard
            hard_negative = F.normalize(hard_negative, dim=-1).detach()
            hard_negative_targets = torch.ones(size=(num_to_gen, 1))*src
            
            hn_feats.append(hard_negative)
            hn_targets.append(hard_negative_targets)
    
    hn_feats = torch.cat(hn_feats, dim=0)
    hn_targets = torch.cat(hn_targets, dim=0).long().squeeze(dim=1)
    
    return hn_feats.cuda(), hn_targets.cuda()


def cal_pos_to_replace(batch_size, words_feats, cls_feats, valid_mask, input_ids):
    # valid_mask = inputs['attention_mask']
    # input_ids = inputs['input_ids']
    
    # (batch_size, sentence_len)
    #  sim_cls_words[i, j]: sim(word[i, j], text[i])
    sim_cls_words = torch.bmm(words_feats, cls_feats.unsqueeze(-1)).squeeze()*valid_mask[:,1:]
    text_len = torch.count_nonzero(input_ids[:, 1:], dim=1) # notice [CLS] 101, [SEP] 102 not within
    
    sorted_sim_cls_words = torch.sort(sim_cls_words, dim=1, descending=True)[0]
    top_index = torch.trunc(text_len/2).numpy()
    delta_sentence_word_sim = torch.stack([sim[int(top_len)] for top_len, sim in zip(top_index, sorted_sim_cls_words)])
    
    replaceable_mask = torch.stack([sim>delta for sim, delta in zip(sim_cls_words, delta_sentence_word_sim)])*valid_mask[:,1:]
    replaceable_mask = torch.cat([torch.zeros(batch_size, 1), replaceable_mask], dim=1)
    
    return replaceable_mask

def cal_wrds_to_replace(batch_size, num_classes, words_feats, proto_feats, delta_sim_proto, valid_mask, input_ids):
    # valid_mask = inputs['attention_mask']
    # input_ids = inputs['input_ids']
    
    # (batch_size, sentence_len, num_classes)
    # sim_proto_words[i, j, k]: sim(word[i, j], prototype[k])
    sim_proto_words = torch.matmul(words_feats, proto_feats.T)
    

    replacing_mask = (sim_proto_words>delta_sim_proto)*(valid_mask[:,1:].unsqueeze(-1))
    replacing_mask = torch.cat([torch.zeros(batch_size, 1, num_classes), replacing_mask], dim=1)
    
    masked_index = replacing_mask*(input_ids.unsqueeze(-1))
    
    replacing_word_list = []
    for _ in range(num_classes):
        replacing_word_list.append(set())
    
    nonzero_index = torch.nonzero(masked_index)
    for idx in nonzero_index:
        i, j, k = idx[0], idx[1], idx[2]
        replacing_word_list[k].add(int(input_ids[i,j]))
        
    return replacing_word_list