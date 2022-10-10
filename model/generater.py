r"""undocumented"""

from numpy import single
import torch
from torch import nn
from fastNLP.models.seq2seq_model import Seq2SeqModel
from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.core.utils import _get_model_device
from functools import partial
from copy import deepcopy


class SequenceGeneratorModel(nn.Module):
    """
    用于封装Seq2SeqModel使其可以做生成任务

    """

    def __init__(self, seq2seq_model: Seq2SeqModel, eos_token_id=None, max_length=30, max_len_a=0.0,
                 num_beams=1, do_sample=True,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0,
                 restricter=None):
        """

        :param Seq2SeqModel seq2seq_model: 序列到序列模型. 会使用seq2seq_model的decoder进行生成
        :param int,None eos_token_id: 句子结束的token id
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.restricter = restricter
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id,
                                           restricter=restricter)

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, head_token_indexs=None, num_head_token=None, tail_token_indexs=None,
                cnt_start=None):
        """
        透传调用seq2seq_model的forward

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, head_token_indexs, num_head_token, tail_token_indexs, cnt_start)

    def predict(self, src_tokens, tgt_tokens, src_seq_len=None, head_token_indexs=None, num_head_token=None, tail_token_indexs=None):
        """
        给定source的内容，输出generate的内容

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        state = self.seq2seq_model.prepare_state(src_tokens, src_seq_len, head_token_indexs, num_head_token, tail_token_indexs)
        result = self.generator.generate(state, tokens=tgt_tokens[:, :1])
        return {'pred': result}
    
    def set_selective_gen(self, selective_gen):
        self.seq2seq_model.decoder.set_selective_gen(selective_gen)

    def set_reverse(self, reverse):
        self.seq2seq_model.decoder.set_reverse(reverse)


r"""

"""

__all__ = [
    'SequenceGenerator'
]


class SequenceGenerator:
    """
    给定一个Seq2SeqDecoder，decode出句子

    """
    def __init__(self, decoder: Seq2SeqDecoder, max_length=20, max_len_a=0.0, num_beams=1,
                 do_sample=False, eos_token_id=None,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0, restricter=None):
        """

        :param Seq2SeqDecoder decoder: Decoder对象
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param int,None eos_token_id: 句子结束的token id
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        self.generate_func = partial(greedy_generate, decoder=decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams, eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=pad_token_id,
                                     restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a

    def set_new_generator(self, max_length=-1, max_len_a=-1, num_beams=-1,
                          repetition_penalty=-1, length_penalty=-1, restricter=-1):
        if max_length == -1:
            max_length = self.max_length
        if max_len_a == -1:
            max_len_a = self.max_len_a
        if num_beams == -1:
            num_beams = self.num_beams
        if repetition_penalty == -1:
            repetition_penalty = self.repetition_penalty
        if length_penalty == -1:
            length_penalty = self.length_penalty
        if restricter == -1:
            restricter = self.restricter
        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams, eos_token_id=self.eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=self.pad_token_id,
                                     restricter=restricter)

    @torch.no_grad()
    def generate(self, state, tokens=None):
        """

        :param State state: encoder结果的State, 是与Decoder配套是用的
        :param torch.LongTensor,None tokens: batch_size x length, 开始的token
        :return: bsz x max_length' 生成的token序列。如果eos_token_id不为None, 每个sequence的结尾一定是eos_token_id
        """

        return self.generate_func(tokens=tokens, state=state)


@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=1, eos_token_id=None, pad_token_id=0,
                    repetition_penalty=1, length_penalty=1.0, restricter=None):
    """
    贪婪地搜索句子

    :param Decoder decoder: Decoder对象
    :param State state: 应该包含encoder的一些输出。
    :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
    :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
    :param int num_beams: 使用多大的beam进行解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param int pad_token_id: pad的token id
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """

    # token_ids = _no_beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a, eos_token_id=eos_token_id,
    #                                         repetition_penalty=repetition_penalty, length_penalty=length_penalty,
    #                                         pad_token_id=pad_token_id, restricter=restricter)

    token_ids = _beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                        num_beams=num_beams, eos_token_id=eos_token_id,
                                        repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                        pad_token_id=pad_token_id, restricter=restricter)

    return token_ids


def _no_beam_search_generate(decoder: Seq2SeqDecoder, state, tokens=None, max_length=20, max_len_a=0.0, eos_token_id=None,
                             repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0,
                             restricter=None):
    device = _get_model_device(decoder)
    batch_size = tokens.size(0)
    num_word_per_sent = state.num_head_token
    encoder_pad_mask = state.encoder_mask
    indices = torch.arange(encoder_pad_mask.size(0), dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_word_per_sent, dim=0)
    encoder_pad_mask = encoder_pad_mask.index_select(dim=0, index=indices)

    """ """    

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores, indices, seq_indices, _ = decoder.decode(tokens=tokens, state=state)  # 主要是为了update state
    state.set_selected(indices, seq_indices)
    # print(head_token_indexs)
    # print(tokens)
    # assert batch_size == encoder_pad_mask.size(0)
    
    # 这里需要考虑如果在第一个位置就结束的情况
    # if _eos_token_id!=-1:
    #     scores[:, _eos_token_id] = -1e12

    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        next_tokens = torch.full_like(tokens, eos_token_id)
        next_tokens[seq_indices] = scores.argmax(dim=-1, keepdim=True)
        # next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))
    # tokens = tokens[:, -1:]

    if max_len_a!=0:
        # (bsz x num_beams, )
        max_lengths = (encoder_pad_mask.sum(dim=1).float()*max_len_a).long() + max_length
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        max_lengths = encoder_pad_mask.new_ones(encoder_pad_mask.size(0)).long()*max_length

    while cur_len < real_max_length:
        scores, _, _, _ = decoder.decode(tokens=token_ids, state=state)  # batch_size x vocab_size

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = torch.full_like(tokens, eos_token_id)
            next_tokens[state.past_seq_indices] = scores.argmax(dim=-1, keepdim=True)
            
        next_tokens = next_tokens.squeeze(-1)

        # 如果已经达到对应的sequence长度了，就直接填为eos了
        if _eos_token_id!=-1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    # if eos_token_id is not None:
    #     tokens.scatter(index=max_lengths[:, None], dim=1, value=eos_token_id)  # 将最大长度位置设置为eos
    # if cur_len == max_length:
    #     token_ids[:, -1].masked_fill_(~dones, eos_token_id)  # 若到最长长度仍未到EOS，则强制将最后一个词替换成eos
    return token_ids

def _beam_search_generate(decoder: Seq2SeqDecoder, state, tokens=None, max_length=20, max_len_a=0.0, num_beams=4, eos_token_id=None, 
                          repetition_penalty=1.0, length_penalty=None, pad_token_id=0,
                          restricter=None):

    # 进行beam search
    device = _get_model_device(decoder)
    batch_size = tokens.size(0)

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores, indices, seq_indices, cnt_selected_start = decoder.decode(tokens=tokens, state=state)

    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

    # single_mask = cnt_selected_start.eq(1)
    # single_scores = scores.masked_select(single_mask)
    # single_indices = indices.masked_select(single_mask)
    # single_seq_indices = seq_indices.masked_select(single_mask)
    # single_state = deepcopy(state)
    # single_state.set_selected(single_indices, single_seq_indices)

    # print('\n[indices.shape]', indices.shape)
    if cnt_selected_start.size(0) > 0:
        multi_mask = cnt_selected_start > 1
        multi_scores = scores[multi_mask]
        multi_indices = indices[multi_mask]
        multi_seq_indices = seq_indices[multi_mask]
        cnt_selected_start = cnt_selected_start[multi_mask]
        multi_batchsize = multi_seq_indices.size(0)
        assert multi_batchsize == multi_scores.size(0) == multi_indices.size(0) == cnt_selected_start.size(0)
        # print('[multi_indices.shape]', multi_indices.shape)
        # (multi_batchsize, 1)
        tokens_ = deepcopy(tokens[multi_seq_indices])
    else:
        multi_seq_indices = []


    result = [ [] for _ in range(batch_size) ]

    """ generate single sequence for each selected head word """
    state_ = deepcopy(state)
    state_.set_selected(indices, seq_indices)

    num_word_per_sent = state_.num_head_token
    encoder_pad_mask = state_.encoder_mask
    indices = torch.arange(encoder_pad_mask.size(0), dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_word_per_sent, dim=0)
    encoder_pad_mask = encoder_pad_mask.index_select(dim=0, index=indices)

    next_tokens = torch.full_like(tokens, eos_token_id)
    next_tokens[seq_indices] = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))

    if max_len_a!=0:
        # (bsz x num_beams, )
        max_lengths = (encoder_pad_mask.sum(dim=1).float()*max_len_a).long() + max_length
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        max_lengths = encoder_pad_mask.new_ones(encoder_pad_mask.size(0)).long()*max_length

    while cur_len < real_max_length:
        scores, _, _, _ = decoder.decode(tokens=token_ids, state=state_)  # batch_size x vocab_size

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

        if restricter is not None:
            _, next_tokens = restricter(state_, token_ids, scores, 1)
        else:
            next_tokens = torch.full_like(tokens, eos_token_id)
            next_tokens[state_.past_seq_indices] = scores.argmax(dim=-1, keepdim=True)
            
        next_tokens = next_tokens.squeeze(-1)

        # 如果已经达到对应的sequence长度了，就直接填为eos了
        if _eos_token_id!=-1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    for i in seq_indices:
        if i not in multi_seq_indices:
            result[i].append(token_ids[i])

    if len(multi_seq_indices) > 0:
        """ generate multi sequence for selected head word """
        scores = F.log_softmax(multi_scores, dim=-1)  # (multi_batchsize, vocab_size)

        # 是 multi_batchsize x (num_beams+1)大小的东西
        _next_scores, _next_tokens = torch.topk(scores, num_beams+1, dim=1, largest=True, sorted=True)

        state.encoder_output = state.encoder_output.index_select(dim=0, index=multi_indices)
        state.encoder_mask = state.encoder_mask.index_select(dim=0, index=multi_indices)
        state.src_tokens = state.src_tokens.index_select(dim=0, index=multi_indices)
        state.num_head_token = state.num_head_token.index_select(dim=0, index=multi_indices)
        state.src_word_embeds = state.src_word_embeds.index_select(dim=0, index=multi_indices)
        to_multi_indices = multi_mask.nonzero(as_tuple=True)[0]
        state.start_offset = state.start_offset.index_select(dim=0, index=to_multi_indices)
        state.src_output_word_embeds = state.src_output_word_embeds.index_select(dim=0, index=to_multi_indices)
        state.reorder_past_key_values(to_multi_indices)

        # print('[multi_mask.shape]', multi_mask.shape)
        # print('[state.num_head_token.shape]', state.num_head_token.shape)
        # print('[state.src_output_word_embeds]', state.src_output_word_embeds.shape)
        # print('to_multi_indices', to_multi_indices)

        # 根据index来做顺序的调转
        indices = torch.arange(multi_batchsize, dtype=torch.long).to(device)
        indices = indices.repeat_interleave(num_beams)
        state.reorder_state(indices)
        tokens_ = tokens_.index_select(dim=0, index=indices)  # multi_batchsize * num_beams x length

        # print(state.encoder_output.shape)
        # print(state.encoder_mask.shape)
        # print(state.src_tokens.shape)
        # print(state.num_head_token.shape)
        # print(state.start_offset.shape)
        # print(state.src_word_embeds.shape)
        # print(state.src_output_word_embeds.shape)

        if max_len_a!=0:
            # (bsz x num_beams, )
            if state.encoder_mask is not None:
                max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
            else:
                max_lengths = tokens_.new_full((multi_batchsize*num_beams, ), fill_value=max_length, dtype=torch.long)
            real_max_length = max_lengths.max().item()
        else:
            real_max_length = max_length
            if state.encoder_mask is not None:
                max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
            else:
                max_lengths = tokens_.new_full((multi_batchsize*num_beams,), fill_value=max_length, dtype=torch.long)
        hypos = [
            BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(multi_batchsize)
        ]

        not_eos_mask = _next_tokens.ne(_eos_token_id)  # 为1的地方不是eos
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
        keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

        next_tokens = _next_tokens.masked_select(keep_mask).view(multi_batchsize, num_beams)  # 这是真的接下来要继续的
        next_scores = _next_scores.masked_select(keep_mask).view(multi_batchsize, num_beams)

        rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

        if len(rows)>0:  # 说明有的开头就结束了
            for row, col in zip(rows.tolist(), cols.tolist()):
                _token = torch.cat([tokens_[row*num_beams], _next_tokens[row, col:col+1]], dim=0)
                hypos[row].add(_token.clone(), _next_scores[row, col].item())

        # 记录生成好的token (multi_batchsize', cur_len)
        token_ids = torch.cat([tokens_, next_tokens.view(-1, 1)], dim=-1)
        dones = [False] * multi_batchsize

        beam_scores = next_scores.view(-1)  # multi_batchsize * num_beams

        #  用来记录已经生成好的token的长度
        cur_len = token_ids.size(1)

        # 0, num_beams, 2*num_beams, ...
        batch_inds_with_numbeams_interval = (torch.arange(multi_batchsize) * num_beams).view(-1, 1).to(token_ids)

        while cur_len < real_max_length:
            scores = decoder.simple_decode(token_ids, state)  # (multi_batchsize x num_beams, vocab_size)
            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                scores.scatter_(dim=1, index=token_ids, src=token_scores)

            if _eos_token_id!=-1:
                max_len_eos_mask = max_lengths.eq(cur_len+1)
                eos_scores = scores[:, _eos_token_id]
                # 如果已经达到最大长度，就把eos的分数加大
                scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores+1e32, eos_scores)

            scores = F.log_softmax(scores, dim=-1)  # (multi_batchsize * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None]  # (multi_batchsize * num_beams, vocab_size)
            _scores = _scores.view(multi_batchsize, -1)  # (multi_batchsize, num_beams*vocab_size)
            # TODO 把限制加到这个位置
            if restricter is not None:
                next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
            else:
                next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
            from_which_beam = ids // vocab_size  # (multi_batchsize, 2*num_beams)
            next_tokens = ids % vocab_size  # (multi_batchsize, 2*num_beams)

            #  接下来需要组装下一个batch的结果。
            #  需要选定哪些留下来
            # next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
            # next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
            # from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)

            not_eos_mask = next_tokens.ne(_eos_token_id)  # 为1的地方不是eos
            keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
            keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

            _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
            _from_which_beam = from_which_beam.masked_select(keep_mask).view(multi_batchsize, num_beams)  # 上面的token是来自哪个beam
            _next_scores = next_scores.masked_select(keep_mask).view(multi_batchsize, num_beams)
            beam_scores = _next_scores.view(-1)

            flag = True
            if cur_len+1 == real_max_length:
                eos_batch_idx = torch.arange(multi_batchsize).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
                eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(multi_batchsize)  # 表示的是indice
                eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)  # 表示的是从哪个beam获取得到的
            else:
                # 将每个batch中在num_beam内的序列添加到结束中, 为1的地方需要结束了
                effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # multi_batchsize x num_beams
                if effective_eos_mask.sum().gt(0):
                    eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                    # 是由于from_which_beam是 (multi_batchsize, 2*num_beams)的，所以需要2*num_beams
                    eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                    eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]  # 获取真实的从哪个beam获取的eos
                else:
                    flag = False

            if flag:
                _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
                for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                        eos_beam_idx.tolist()):
                    if not dones[batch_idx]:
                        score = next_scores[batch_idx, beam_ind].item()
                        # 之后需要在结尾新增一个eos
                        if _eos_token_id!=-1:
                            hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                        else:
                            hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

            # 更改state状态, 重组token_ids
            reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten成一维
            state.reorder_state(reorder_inds)
            # 重新组织token_ids的状态
            token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

            for batch_idx in range(multi_batchsize):
                dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                                max_lengths[batch_idx*num_beams]==cur_len+1

            cur_len += 1

            if all(dones):
                break

        for i, hypotheses in enumerate(hypos):
            preds = sorted(hypotheses.hyp, key=lambda x: x[0], reverse=True)[ :cnt_selected_start[i] ]
            preds = [torch.cat([x[1], x[1].new_ones(1)*_eos_token_id]) for x in preds]  # 把上面替换为非eos的词替换回eos
            result[ multi_seq_indices[i] ] = preds

    return result


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
