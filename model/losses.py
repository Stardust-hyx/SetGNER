
from cv2 import transpose
from fastNLP import LossBase
import torch
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self, num_labels, eos_coef=1, null_coef=1, word_coef=1, cls_coef=1,
                num_ent_types=None, rel_type_weights=None, use_cls_loss=True):
        super().__init__()
        self.null_tag_index = num_labels
        self.eos_coef = eos_coef
        self.cls_coef = cls_coef
        self.null_coef = null_coef
        self.word_coef = word_coef
        self.num_ent_types = num_ent_types
        self.use_cls_loss = use_cls_loss
        if rel_type_weights is not None:
            self.num_rel_types = len(rel_type_weights)
            self.rel_type_weights = torch.FloatTensor(rel_type_weights)
        
        weight = torch.ones(self.null_tag_index+1)
        if self.eos_coef != 1:
            weight[0] = self.eos_coef
        if self.cls_coef != 1:
            weight[1: self.null_tag_index] = self.cls_coef
        if self.null_coef != 1:
            weight[self.null_tag_index] = self.null_coef

        if rel_type_weights is not None:
            weight[1+self.num_ent_types: 1 + self.num_ent_types + self.num_rel_types] = self.rel_type_weights
        self.weight = weight.to(torch.device('cuda'))

        self.detection_criterion = torch.nn.BCEWithLogitsLoss()

        cnt_weight = torch.FloatTensor([1, 1, 1.5, 2, 2.5]).to(torch.device('cuda'))
        self.cnt_weight = cnt_weight

    def get_loss(self, tgt_tokens, tgt_seq_len, type_of_head_token, pred, encoder_logits, seq_indices,
                    cnt_start_logits, cnt_start):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        # print(tgt_tokens.shape, flush=True)
        # print(tgt_seq_len, flush=True)

        tgt_tokens = tgt_tokens.index_select(dim=0, index=seq_indices)
        tgt_seq_len = tgt_seq_len.index_select(dim=0, index=seq_indices)

        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)

        # print(tgt_tokens[:2], flush=True)
        # print(pred.shape, flush=True)

        """ """
        
        weight = pred.new_ones(pred.size(-1))
        weight[:self.null_tag_index+1] = self.weight
        weight[self.null_tag_index+1:] = self.word_coef

        assert tgt_tokens.size(0) == pred.size(0)
        loss_gen = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2), weight=weight)

        if self.use_cls_loss:
            loss_cls = self.detection_criterion(encoder_logits, type_of_head_token)
            loss_cnt = F.cross_entropy(target=cnt_start, input=cnt_start_logits.transpose(1, 2), weight=self.cnt_weight)
            return loss_gen + loss_cls * 0.8 + loss_cnt * 0.6
        else:
            return loss_gen
