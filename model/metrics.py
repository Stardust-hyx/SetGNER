from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from itertools import chain
import numpy as np


class EntMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, num_rel_types=0, reverse=False):
        super(EntMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 1  # +1是由于前面有个特殊符号eos
        self.rel_tag_end_index = self.word_start_index - 2  # -2是由于后2个是special tokens
        self.rel_tag_start_index = self.word_start_index - 2 - num_rel_types
        self.reverse = reverse

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.sample_idx_2_pred_ent_2_tag = dict()

    def evaluate(self, target_span, sample_idx, pred):
        target_span = list(chain(*target_span))
        sample_idx = list(chain(*sample_idx))

        for idx in sample_idx:
            self.sample_idx_2_pred_ent_2_tag[idx] = dict()


        for i, (ts, pred_seqs) in enumerate(zip(target_span, pred)):

            pairs = []
            for ps in pred_seqs:
                ps = ps.tolist()
                cur_pair = []

                for j in ps:
                    if j == self.eos_token_id:
                        break

                    if j < self.rel_tag_start_index:
                        if len(cur_pair) > 0:
                            if (not self.reverse and all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)])) \
                                or (self.reverse and all([cur_pair[i] > cur_pair[i + 1] for i in range(len(cur_pair) - 1)])):
                                pairs.append(tuple([cur_pair[0], cur_pair[-1]] + [j]))
                                self.sample_idx_2_pred_ent_2_tag[sample_idx[i]][(cur_pair[0], cur_pair[-1])] = j
                        cur_pair = []
                    elif self.rel_tag_end_index <= j < self.word_start_index:
                        cur_pair = []
                    elif j < self.word_start_index:
                        continue
                    else:
                        cur_pair.append(j)


            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn += fn
            self.tp += tp
            self.fp += fp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        res['f'] = round(f, 4) * 100
        res['rec'] = round(rec, 4) * 100
        res['pre'] = round(pre, 4) * 100
        res['sample_idx_2_pred_ent_2_tag'] = self.sample_idx_2_pred_ent_2_tag

        if reset:
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.sample_idx_2_pred_ent_2_tag = dict()

        return res

def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (set, list, np.ndarray)):
        ts = {tuple([key[0], key[-2], key[-1]]): 1 for key in list(ts)}
    if isinstance(ps, (set, list, np.ndarray)):
        ps = {tuple(key): 1 for key in list(ps)}

    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp

