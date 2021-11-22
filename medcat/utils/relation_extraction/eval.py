import torch
import torch.nn as nn
import logging
import os
from itertools import combinations

from medcat.utils.relation_extraction.utils import load_bin_file

class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize
        
        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1 / (torch.norm(f1_vec)*torch.norm(f2_vec))
        else:
            factor = 1.0
        
        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor*torch.dot(f1_vec, f2_vec)))
        else:
            p = factor*torch.dot(f1_vec, f2_vec)
        return p
    
    def dot_(self, f1_vec, f2_vec):
        return -torch.dot(f1_vec, f2_vec)
    
    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels, verbose=False):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, enumerate)
        blank_labels: (batch_size, 0 or 1)
        '''
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]
        
        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blank_logits[pos1, :], blank_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
        
        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :]))

        neg_logits = torch.stack(neg_logits, dim=0)

        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]
        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)
        
        
        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()
            pos_logits = pos_logits.cuda()
            neg_logits = neg_logits.cuda()

        lm_loss = self.LM_criterion(lm_logits, target=lm_labels.long())


        blank_loss = self.BCE_criterion(torch.cat([pos_logits, neg_logits], dim=0), \
                                        blank_labels_)

        if verbose:
            print("LM loss, blank_loss for last batch: %.5f, %.5f" % (lm_loss, blank_loss))
           
        total_loss = lm_loss + blank_loss
        return total_loss

    @classmethod
    def load_state(net, optimizer, scheduler, model_name="BERT", load_best=False):
        """ Loads saved model and optimizer states if exists """
        base_path = "./data/"
        amp_checkpoint = None
        checkpoint_path = os.path.join(base_path,"test_checkpoint_%s.pth.tar" % model_name)
        best_path = os.path.join(base_path,"test_model_best_%s.pth.tar" % model_name)
        start_epoch, best_pred, checkpoint = 0, 0, None
        if (load_best == True) and os.path.isfile(best_path):
            checkpoint = torch.load(best_path)
            logging.info("Loaded best model.")
        elif os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            logging.info("Loaded checkpoint model.")
        if checkpoint != None:
            start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            amp_checkpoint = checkpoint['amp']
            logging.info("Loaded model and optimizer.")    
        return start_epoch, best_pred, amp_checkpoint

    @classmethod
    def load_results(cls, model_name="BERT"):
        """ Loads saved results if exists """
        losses_path = "./data/test_losses_per_epoch_%s.pkl" % model_name
        accuracy_path = "./data/test_accuracy_per_epoch_%s.pkl" % model_name
        if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
            losses_per_epoch = load_bin_file("test_losses_per_epoch_%s.pkl" % model_name)
            accuracy_per_epoch = load_bin_file("test_accuracy_per_epoch_%s.pkl" % model_name)
            logging.info("Loaded results buffer")
        else:
            losses_per_epoch, accuracy_per_epoch = [], []
        return losses_per_epoch, accuracy_per_epoch
    
    @classmethod
    def evaluate_results(cls, lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=False):
        '''
        evaluate must be called after loss.backward()
        '''
     
        lm_logits_pred_ids = torch.softmax(input=lm_logits, dim=-1).max(1)[1]
        lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum().float()/len(masked_for_pred)).item()
        
        if print_:
            print("Predicted masked tokens: \n")
            print(tokenizer.decode(lm_logits_pred_ids.cpu().numpy() if lm_logits_pred_ids.is_cuda else \
                                lm_logits_pred_ids.numpy()))
            print("\n Masked labels tokens: \n")
            print(tokenizer.decode(masked_for_pred.cpu().numpy() if masked_for_pred.is_cuda else \
                                masked_for_pred.numpy()))
  
        return lm_accuracy