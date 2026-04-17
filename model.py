import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TempModel(nn.Module):

    def __init__(self, base_model, num_classes, feat_dim):
        super().__init__()
        # Bert/ RoBerta
        self.base_model = base_model
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # self.method = method
        # classification for prototype
        self.fc = nn.Linear(base_model.config.hidden_size, num_classes)
        # MLP for feat
        self.feat_head = nn.Sequential(nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
                                            nn.BatchNorm1d(base_model.config.hidden_size), nn.ReLU(inplace=True),
                                      nn.Linear(base_model.config.hidden_size, feat_dim))

        self.dropout = nn.Dropout(0.5)
        # MLP for prototypes
        self.head_fc = nn.Sequential(nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
                                            nn.BatchNorm1d(base_model.config.hidden_size), nn.ReLU(inplace=True),
                                      nn.Linear(base_model.config.hidden_size, feat_dim))
        for param in base_model.parameters():
            param.requires_grad_(True)
            
            

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        batch_sz = hiddens.shape[0]
        hidden_size = self.base_model.config.hidden_size
        cls = hiddens[:, 0, :]
        # wrd = hiddens[:, 1:, :]
        # [cls] token embedding
        
        cls_feats = F.normalize(self.feat_head(cls), dim=1)
        # cls_feats = self.feat_head(cls)
        
        # [prototype] embedding
        logits = self.fc(cls)
        # nomalize to imporove
        logits = F.normalize(logits, dim=-1)
        proto_feats = F.normalize(self.head_fc(self.fc.weight), dim=1)
        # proto_feats = self.head_fc(self.fc.weight)
        
        # [word] token embedding in the text
        # words_feats = F.normalize(self.feat_head(wrd.contiguous().view(-1, hidden_size)), dim=1)
        # words_feats = words_feats.view(batch_sz, -1, self.feat_dim)
        
        # all feats returned without normalize, remember to perform L-2 norm in loss func!!
#         outputs = {
#             'cls_feats': cls_feats,
#             'proto_feats': proto_feats,
#             'logits':logits
#             # 'words_feats':words_feats
#         }

        return cls_feats, proto_feats, logits

    def hn_scl(self, syn_samples, feats_1, feats_2, protos, targets, num_classes, n_pos, n_neg):
        
        device = (torch.device('cuda')
                      if feats_1.is_cuda
                      else torch.device('cpu'))

        syn_samples = torch.cat(torch.unbind(syn_samples, dim=0), dim=0)
        feats = torch.cat([feats_1, feats_2, protos], dim=0)
        pos_idx = torch.zeros(num_classes, n_pos).long().to(device)
        neg_idx = torch.zeros(num_classes, n_neg).long().to(device)
        q_cls = n_pos + n_neg
        for i in range(num_classes):
            pos_idx[i,:] = torch.arange(n_pos) + i*q_cls
            neg_idx[i,:] = torch.arange(n_pos, q_cls) + i*q_cls

        sim_con_queue = torch.einsum('ik,jk->ij',[feats, syn_samples])
        # labels = torch.cat([targets, targets], dim=0)
        proto_targets = torch.arange(num_classes).to(device)
        # print(targets.shape, proto_targets.shape)
        labels = torch.cat([targets, targets, proto_targets], dim=0)
        sim_con_pos = torch.gather(sim_con_queue, 1, pos_idx[labels, :])
        sim_con_neg = torch.gather(sim_con_queue, 1, neg_idx[labels, :])

        sim_con_batch = feats @ feats.T
        mask = torch.ones_like(sim_con_batch).scatter_(1, torch.arange(len(feats)).unsqueeze(1).to(device), 0.)
        sim_con_batch = sim_con_batch[mask.bool()].view(sim_con_batch.shape[0], sim_con_batch.shape[1] - 1)
        sim_con = torch.cat([sim_con_batch, sim_con_pos, sim_con_neg], dim=1)

        labels_con_batch = torch.eq(labels[:, None], labels[None, :]).float()
        labels_con_batch = labels_con_batch[mask.bool()].view(labels_con_batch.shape[0], labels_con_batch.shape[1] - 1)
        labels_con = torch.cat([labels_con_batch, torch.ones_like(sim_con_pos), torch.zeros_like(sim_con_neg)], dim=1)

        return sim_con, labels_con
