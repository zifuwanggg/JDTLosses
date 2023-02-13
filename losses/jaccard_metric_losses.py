import collections

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

    
class JaccardMetricLoss(_Loss):
    def __init__(self, loss="JML1", soft_label=False, classes="present", per_image=False, log_loss=False, T=1, smooth=1, threshold=0.1, ignore_index=None):
        super().__init__()
        self.loss = loss
        self.soft_label = soft_label
        self.classes = classes
        self.per_image = per_image
        self.log_loss = log_loss
        self.T = T
        self.smooth = smooth
        self.threshold = threshold
        self.ignore_index = ignore_index
        
    
    def forward(self, logits, label, not_ignore=None):
        prob = (logits / self.T).log_softmax(dim=1).exp()
                        
        if self.per_image:
            return self.forward_per_image(prob, label, not_ignore)
        else:
            return self.forward_per_batch(prob, label, not_ignore)
    
    
    def compute_score(self, prob, label):    
        if self.loss == "JML1":
            cardinality = torch.sum(prob + label)
            difference = torch.sum(torch.abs(prob - label))
            intersection = cardinality - difference
            union = cardinality + difference
            score = (intersection + self.smooth) / (union + self.smooth)
        elif self.loss == "JML2":
            difference = torch.sum(torch.abs(prob - label))
            intersection = torch.sum(prob * label)
            union = difference + intersection
            score = (intersection + self.smooth) / (union + self.smooth)
        
        return score
        
    
    def forward_per_image(self, prob, label, not_ignore):
        losses = []
        
        batch_size, num_classes, _, _ = prob.shape
            
        for i in range(batch_size):
            losses_i = []
            
            if self.classes == "all":
                idx = torch.arange(num_classes)
            elif self.classes == "present":
                if not self.soft_label:
                    idx = label[i, :, :].unique()
                else:
                    idx = torch.argmax(label[i, :, :, :], dim=0).unique()
            elif self.classes == "prob":
                idx = torch.amax(prob[i, :, :, :], dim=(1,2)) > self.threshold
            elif self.classes == "label":
                assert self.soft_label
                idx = torch.amax(label[i, :, :, :], dim=(1,2)) > self.threshold
            elif self.classes == "both":
                assert self.soft_label
                idx = torch.amax((prob + label)[i, :, :, :], dim=(1,2)) > self.threshold
            elif isinstance(self.classes, collections.Iterable):
                idx = torch.tensor(self.classes)
            else:
                raise NotImplementedError
                
            if idx[-1] == self.ignore_index:
                idx = idx[:-1]
                
            idx = idx.cpu()
                
            classes_i = set(torch.arange(num_classes)[idx].tolist())
            
            if self.ignore_index in classes_i:
                classes_i.remove(self.ignore_index)
                            
            if len(classes_i) < 1:
                continue
            
            prob_i = prob[i, :, :, :].view(num_classes, -1)
            not_ignore_i = not_ignore[i, :, :].view(-1)
            
            if self.soft_label:
                label_i = label[i, :, :, :].view(num_classes, -1)
            else:
                label_i = label[i, :, :].view(-1)
                label_i = F.one_hot(label_i, num_classes + 1)  
                label_i = label_i.permute(1, 0)

            for j in classes_i:
                prob_j = prob_i[j, :][not_ignore_i]
                label_j = label_i[j, :][not_ignore_i]
                
                if self.classes == "all" and torch.sum(prob_j + label_j) < self.threshold:
                    continue
                                
                score = self.compute_score(prob_j, label_j)
                
                if self.log_loss:
                    losses_i.append(-torch.log(score))
                else:
                    losses_i.append(1.0 - score)
                    
            losses.append(sum(losses_i) / len(losses_i))
            
        return sum(losses) / len(losses)
    
    
    def forward_per_batch(self, prob, label, not_ignore):
        losses = []
        
        batch_size, num_classes, _, _ = prob.shape

        if self.classes == "all":
            idx = torch.arange(num_classes)
        elif self.classes == "present":
            if not self.soft_label:
                idx = label.unique()
            else:
                idx = torch.argmax(label, dim=1).unique()
        elif self.classes == "prob":
            idx = torch.amax(prob, dim=(0,2,3)) > self.threshold
        elif self.classes == "label":
            assert self.soft_label
            idx = torch.amax(label, dim=(0,2,3)) > self.threshold
        elif self.classes == "both":
            assert self.soft_label
            idx = torch.amax(prob + label, dim=(0,2,3)) > self.threshold
        elif isinstance(self.classes, collections.Iterable):
            idx = torch.tensor(self.classes)
        else:
            raise NotImplementedError
        
        if idx[-1] == self.ignore_index:
            idx = idx[:-1]
            
        idx = idx.cpu()
        
        if idx.size(0) > 1:
            classes = set(torch.arange(num_classes)[idx].tolist())
        else:
            classes = list(idx)
 
        if self.ignore_index in classes:
            classes.remove(self.ignore_index)
                
        if len(classes) < 1:
            return 0. * prob.sum()
                    
        prob = prob.view(batch_size, num_classes, -1)
        not_ignore = not_ignore.view(batch_size, -1)
                
        if self.soft_label:
            label = label.view(batch_size, num_classes, -1) 
        else:
            label = label.view(batch_size, -1)            
            label = F.one_hot(label, num_classes + 1)  
            label = label.permute(0, 2, 1)
        
        for j in classes:
            prob_j = prob[:, j, :][not_ignore]
            label_j = label[:, j, :][not_ignore]
            
            if self.classes == "all" and torch.sum(prob_j + label_j) < self.threshold:
                continue
            
            score = self.compute_score(prob_j, label_j)
            
            if self.log_loss:
                losses.append(-torch.log(score))
            else:
                losses.append(1.0 - score)
    
        return sum(losses) / len(losses)