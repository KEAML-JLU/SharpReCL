import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list).cuda()
        cls_p_list = (cls_num_list + 1) / (cls_num_list.sum() + cls_num_list.shape[0])
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
    

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class SiamBS_SPM(nn.Module):
    def __init__(self, cls_num_list, queue_size_per_cls, temperature=1, con_weight=1.0, effective_num_beta=0.999):
        super(SiamBS_SPM, self).__init__()
        self.temperature = temperature
        self.queue_size_per_cls = queue_size_per_cls
        self.con_weight = con_weight
        self.effective_num_beta = effective_num_beta

        self.criterion_cls = nn.CrossEntropyLoss()
        self.log_sfx = nn.LogSoftmax(dim=1)
        self.criterion_con = torch.nn.KLDivLoss(reduction='none')

        self._cal_prior_weight_for_classes(cls_num_list)
        
        # 
        self.queue_label = torch.repeat_interleave(torch.arange(len(cls_num_list)), self.queue_size_per_cls)
        self.instance_prior = self.class_prior.squeeze()[self.queue_label]

    def _cal_prior_weight_for_classes(self, cls_num_list):
        
        ### logit prior for BalSfx
        cls_num_list_tensor = torch.Tensor(cls_num_list).view(1, len(cls_num_list))
        self.class_prior = cls_num_list_tensor / cls_num_list_tensor.sum()
        self.class_prior = self.class_prior.to(torch.device('cuda'))
        
        # effective number on the orginal-batch level
        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))
            
        else:
            self.class_weight  = torch.FloatTensor(torch.ones(len(cls_num_list))).to(torch.device('cuda'))
        

    # Supervised contrastvie style
    def forward(self, sim_con, labels_con, logits_cls, labels):
        ### Siamese Balanced Softmax
        logits_cls = logits_cls + torch.log(self.class_prior + 1e-9)
        loss_cls = self.criterion_cls(logits_cls, labels)
        
        
        sim_con = self.log_sfx(sim_con / self.temperature)
        
        loss_con = self.criterion_con(sim_con, labels_con)
        loss_con = loss_con.sum(dim=1) / (labels_con.sum(dim=1) + 1e-9)
        
        
        device = (torch.device('cuda')
                  if labels.is_cuda
                  else torch.device('cpu'))
        
        proto_labels = torch.arange(len(self.class_prior)).to(device)
        # print(targets.shape, proto_targets.shape)
        labels = torch.cat([labels , proto_labels], dim=0)
        instance_weight = self.class_weight.squeeze()[proto_labels.squeeze()]
        loss_con = (instance_weight * loss_con).mean()

        # Total loss
        loss = loss_cls + self.con_weight * loss_con
        
        return loss_cls, loss_con, loss