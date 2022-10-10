from pickle import NONE
from numpy import select
import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
from fastNLP.core.utils import _get_model_device
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=False)
        encoder_outputs = dict.last_hidden_state
        return encoder_outputs, mask


class CaGFBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, maxpool=True, dropout=0.3, pos_interval=3, confidence=0.5, dropout_hidden=0.3,
                 selective_gen=False):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.bart_decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([2]+label_ids)  # 2 is eos_token_id
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.hidden_size = hidden_size
        self.encoder_mlp1 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.25),
                                             nn.ReLU(),)
        self.cls_head = nn.Linear(hidden_size, 4)

        self.encoder_mlp2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.15),
                                             nn.ReLU(),)
        self.cnt_head = nn.Linear(hidden_size, 5)

        self.maxpool = maxpool
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_hidden = nn.Dropout(dropout_hidden)
        self.pos_interval = pos_interval
        self.selective_gen = selective_gen
        self.reverse = False

        self.confidence = confidence

    def forward(self, tokens, state):
        device = _get_model_device(self)
        cnt_start = state.cnt_start

        encoder_outputs_ = state.encoder_output
        use_pos_cache = tokens.size(1) > 1 and not self.training
        num_word_per_sent = state.num_head_token
        max_num_word_per_sent = torch.max(num_word_per_sent)
        head_token_indexs = state.head_token_indexs
        tail_token_indexs = state.tail_token_indexs
        head_token_indexs_unsqueeze = head_token_indexs.unsqueeze(2).repeat(1, 1, self.hidden_size)
        tail_token_indexs_unsqueeze = tail_token_indexs.unsqueeze(2).repeat(1, 1, self.hidden_size)
        head_token_mask = seq_len_to_mask(num_word_per_sent, max_len=max_num_word_per_sent)
        src_tokens = state.src_tokens
        embed_tokens = self.bart_decoder.embed_tokens
        input_embed = embed_tokens(src_tokens)
        
        # print(head_token_indexs)
        """ mask label info during decoding """
        tmp_mask = src_tokens.eq(2)
        tmp_mask_cumsum = tmp_mask.cumsum(dim=1)
        encoder_pad_mask = tmp_mask.__or__(tmp_mask_cumsum.lt(1)).__and__(tmp_mask_cumsum.lt(2))
        # print(src_tokens[:5])
        # print(encoder_pad_mask[:5])
        # print('')

        if state.past_indices is not None:
            assert state.past_seq_indices is not None
            encoder_logits = None
            indices = state.past_indices
            seq_indices = state.past_seq_indices

            cnt_start_logits = None
            cnt_selected_start = None
        else:
            src_features = self.encoder_mlp1((encoder_outputs_ + input_embed) / 2)
            src_features_heads = src_features.gather(dim=1, index=head_token_indexs_unsqueeze)
            src_features_tails = src_features.gather(dim=1, index=tail_token_indexs_unsqueeze)
            if self.maxpool:
                src_features_ = torch.where(src_features_heads > src_features_tails, src_features_heads, src_features_tails)
            else:
                src_features_ = (src_features_heads + src_features_tails) / 2
            encoder_logits = self.cls_head(src_features_)
            
            if self.selective_gen:
                if self.reverse:
                    pred_index_mask = (encoder_logits[:, :, -2] > self.confidence).__and__(head_token_mask)
                else:
                    pred_index_mask = (encoder_logits[:, :, -1] > self.confidence).__and__(head_token_mask)
                if self.training:
                    pred_index_mask = pred_index_mask.__or__(torch.rand(pred_index_mask.shape, device=device) < 0.3).__and__(head_token_mask)
            else:
                pred_index_mask = head_token_mask
            
            encoder_logits = encoder_logits[head_token_mask]    # for calculating BCEloss
            pred_ent_index = pred_index_mask.nonzero(as_tuple=False)
            # print(encoder_logits.shape)
            # print(pred_index_mask.shape)
            # print(pred_ent_index.shape)
            # print(pred_ent_index)
            
            cnt_features = self.encoder_mlp2(encoder_outputs_)
            cnt_features_heads = cnt_features.gather(dim=1, index=head_token_indexs_unsqueeze)
            cnt_features_tails = cnt_features.gather(dim=1, index=tail_token_indexs_unsqueeze)
            if self.maxpool:
                cnt_features_ = torch.where(cnt_features_heads > cnt_features_tails, cnt_features_heads, cnt_features_tails)
            else:
                cnt_features_ = (cnt_features_heads + cnt_features_tails) / 2
            cnt_start_logits = self.cnt_head(cnt_features_)
            if not self.training:
                cnt_start = torch.argmax(cnt_start_logits, dim=-1)
                
            cnt_start[cnt_start.le(0)] = 1
            cnt_start = cnt_start.masked_fill(head_token_mask.eq(0), 0)

            cnt_selected_start = cnt_start.masked_select(pred_index_mask)

            if pred_ent_index.size(0)==0:
                indices = torch.tensor([0], dtype=torch.long, device=device)
                s = [torch.tensor(0, dtype=torch.long, device=device)]
            else:
                # indices = pred_ent_index.index_select(dim=1, index=torch.tensor([0], device=device)).squeeze(1)
                indices = pred_ent_index[:, 0]
                if self.training:
                    indices = indices.repeat_interleave(cnt_selected_start)
                    cumsum = cnt_start.cumsum(dim=-1)
                    bases = cumsum[:, -1].cumsum(dim=-1)
                    s = []
                    for t in pred_ent_index:
                        base = bases[t[0] - 1] if t[0]>0 else 0
                        tmp = torch.arange(base + (cumsum[t[0]][t[1]-1] if t[1]>0 else 0), base + cumsum[t[0]][t[1]],
                                            dtype=torch.long, device=device)
                        s.extend(tmp)
                else:
                    cumsum = num_word_per_sent.cumsum(dim=0)
                    s = [t[1] + (cumsum[t[0] - 1] if t[0]>0 else 0) for t in pred_ent_index]
            seq_indices = torch.stack(s, dim=0)

        encoder_outputs = encoder_outputs_.index_select(dim=0, index=indices)
        encoder_pad_mask = encoder_pad_mask.index_select(dim=0, index=indices)

        """ offset for decoder position embedding"""
        if self.reverse:
            start_offset = [torch.arange(start=64, end=64+l).unsqueeze(1) for l in num_word_per_sent]
        else:
            start_offset = [torch.arange(l).unsqueeze(1) for l in num_word_per_sent]

        start_offset = torch.cat(start_offset).to(device)
        if self.training:
            cnt_head_start = cnt_start.masked_select(head_token_mask)
            # print('[cnt_head_start.shape]', cnt_head_start.shape)
            # print('[start_offset.shape]', start_offset.shape)
            start_offset = start_offset.repeat_interleave(cnt_head_start, dim=0)
        
        start_offset = start_offset.index_select(dim=0, index=seq_indices)
        start_offset *= self.pos_interval

        tokens = tokens.index_select(dim=0, index=seq_indices)

        # tokens第一个0之后的0全是padding，因为0是eos, 在pipe中规定的
        cumsum = tokens.eq(0).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        tokens_shape = tokens.shape
        # print('[tokens_shape]', tokens_shape)
        if use_pos_cache:
            tokens = tokens[:, -1:]

        # 把输入做一下映射
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]
        word_embeds = embed_tokens(tag_mapped_tokens)
        
        src_words_index = tokens - self.src_start_index -1 # bsz x num_src_token

        # src_tokens_embeds = input_embed
        src_tokens_embeds = (encoder_outputs_ + input_embed) / 2
        src_word_embeds_heads = src_tokens_embeds.gather(dim=1, index=head_token_indexs_unsqueeze)
        src_word_embeds_tails = src_tokens_embeds.gather(dim=1, index=tail_token_indexs_unsqueeze)
        if self.maxpool:
            src_word_embeds = torch.where(src_word_embeds_heads > src_word_embeds_tails,
                                            src_word_embeds_heads, src_word_embeds_tails)
        else:
            src_word_embeds = (src_word_embeds_heads + src_word_embeds_tails) / 2

        # print('[src_word_embeds.shape]', src_word_embeds.shape)
        # print('[indices.shape]', indices.shape)
        if use_pos_cache:
            for i in range(src_words_index.size(0)):
                word_index = src_words_index[i][0]
                if word_index < 0:
                    continue
                word_embeds[i, 0] = src_word_embeds[indices[i]][word_index]

        else:
            # print('[word_embeds.shape]', word_embeds.shape)
            # print('[pred_index_mask]', pred_index_mask.shape)
            if self.training and cnt_selected_start.size(0) > 0:
                head_word_embeds = src_word_embeds.masked_select(pred_index_mask.unsqueeze(2)).view(-1, self.hidden_size)
                head_word_embeds = head_word_embeds.repeat_interleave(cnt_selected_start, dim=0)
                # print(head_word_embeds.shape)
                word_embeds[:, 0] = head_word_embeds
            else:
                head_word_embeds = src_word_embeds.masked_select(head_token_mask.unsqueeze(2)).view(-1, self.hidden_size)
                word_embeds[:, 0] = head_word_embeds.index_select(dim=0, index=seq_indices)

            for i in range(src_words_index.size(0)):
                embeds = src_word_embeds[indices[i]]
                for j in range(1, src_words_index.size(1)):
                    word_index = src_words_index[i][j]
                    if word_index < 0:
                        continue
                    word_embeds[i, j] = embeds[word_index]

        word_embeds = word_embeds * self.bart_decoder.embed_scale

        if self.training:
            word_embeds = word_embeds[:, :-1]
            # decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.bart_decoder(input_ids=None,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=tgt_pad_mask[:, :-1],
                                decoder_causal_mask=self.causal_masks[:word_embeds.size(1), :word_embeds.size(1)],
                                start_offset=start_offset,
                                return_dict=True,
                                reverse = self.reverse,
                                input_embeds=word_embeds,
                                )
        else:
            past_key_values = state.past_key_values
            dict = self.bart_decoder(input_ids=None,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                start_offset=start_offset,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=use_pos_cache,
                                reverse = self.reverse,
                                tokens_shape = tokens_shape,
                                input_embeds=word_embeds,
                                )
                                
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_hidden(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+max_num_word_per_sent+1),
                                       fill_value=-1e32)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens.weight[2:3]))  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        src_outputs = (encoder_outputs_ + input_embed) / 2

        # print(state.word_bpe_span)
        # print(state.num_head_token)
        # print(src_outputs.shape)
        # print(indices)
        # print(logits.shape)

        src_output_word_embeds_heads = src_outputs.gather(dim=1, index=head_token_indexs_unsqueeze)
        src_output_word_embeds_tails = src_outputs.gather(dim=1, index=tail_token_indexs_unsqueeze)
        if self.maxpool:
            src_output_word_embeds = torch.where(src_output_word_embeds_heads > src_output_word_embeds_tails,
                                                src_output_word_embeds_heads, src_output_word_embeds_tails)
        else:
            src_output_word_embeds = (src_output_word_embeds_heads + src_output_word_embeds_tails) / 2

        src_output_word_embeds = src_output_word_embeds.index_select(dim=0, index=indices)
        src_output_word_embeds = self.dropout_layer(src_output_word_embeds)

        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_output_word_embeds)  # bsz x max_len x max_word_len

        mask = head_token_mask.eq(0).index_select(dim=0, index=indices).unsqueeze(1)
        word_scores = word_scores.masked_fill(mask, -1e32)
        
        logits[:, :, 0:1] = eos_scores
        logits[:, :, 1:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index+1:] = word_scores

        if not self.training and not use_pos_cache:
            state.start_offset = start_offset
            state.src_word_embeds = src_word_embeds
            state.src_output_word_embeds = src_output_word_embeds

        return logits, encoder_logits, indices, seq_indices, cnt_start_logits, cnt_selected_start
    
    def decode(self, tokens, state):
        forward_output = self(tokens, state)
        return forward_output[0][:, -1], forward_output[2], forward_output[3], forward_output[5]
    
    def simple_decode(self, tokens, state):
        encoder_outputs = state.encoder_output
        use_pos_cache = tokens.size(1) > 1 and not self.training
        assert use_pos_cache
        num_word_per_sent = state.num_head_token
        src_output_word_embeds = state.src_output_word_embeds
        max_num_word_per_sent = src_output_word_embeds.size(1)
        head_token_mask = seq_len_to_mask(num_word_per_sent, max_len=max_num_word_per_sent)
        src_tokens = state.src_tokens
        embed_tokens = self.bart_decoder.embed_tokens
        
        """ mask label info during decoding """
        tmp_mask = src_tokens.eq(2)
        tmp_mask_cumsum = tmp_mask.cumsum(dim=1)
        encoder_pad_mask = tmp_mask.__or__(tmp_mask_cumsum.lt(1)).__and__(tmp_mask_cumsum.lt(2))

        """ offset for decoder position embedding"""
        start_offset = state.start_offset

        tokens_shape = tokens.shape
        if use_pos_cache:
            tokens = tokens[:, -1:]

        # 把输入做一下映射
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]
        word_embeds = embed_tokens(tag_mapped_tokens)
        
        src_words_index = tokens - self.src_start_index -1 # bsz x num_src_token
        src_word_embeds = state.src_word_embeds
        
        for i in range(src_words_index.size(0)):
            word_index = src_words_index[i][0]
            if word_index < 0:
                continue
            word_embeds[i, 0] = src_word_embeds[i][word_index]

        word_embeds = word_embeds * self.bart_decoder.embed_scale

        past_key_values = state.past_key_values
        dict = self.bart_decoder(input_ids=None,
                            encoder_hidden_states=encoder_outputs,
                            encoder_padding_mask=encoder_pad_mask,
                            decoder_padding_mask=None,
                            decoder_causal_mask=None,
                            start_offset=start_offset,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                            use_pos_cache=use_pos_cache,
                            reverse = self.reverse,
                            tokens_shape = tokens_shape,
                            input_embeds=word_embeds,
                            )
                                
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_hidden(hidden_state)
        
        state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+max_num_word_per_sent+1),
                                       fill_value=-1e32)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens.weight[2:3]))  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_output_word_embeds)  # bsz x max_len x max_word_len

        mask = head_token_mask.eq(0).unsqueeze(1)
        word_scores = word_scores.masked_fill(mask, -1e32)
        
        logits[:, :, 0:1] = eos_scores
        logits[:, :, 1:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index+1:] = word_scores

        return logits[:, -1]

    def set_selective_gen(self, selective_gen):
        self.selective_gen = selective_gen
        print('selective_gen =', self.selective_gen)
    
    def set_reverse(self, reverse):
        self.reverse = reverse
        print('reverse =', self.reverse)


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, dropout=0.03, decoder_dropout=0.03, pos_interval=3,
                    confidence=0.5, dropout_hidden=0.3):
        model = BartModel.from_pretrained(bart_model, dropout=dropout)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        # print('model.encoder.embed_tokens.weight.shape[0]', num_tokens)
        # print('len(tokenizer.unique_no_split_tokens)', len(tokenizer.unique_no_split_tokens))
        model.resize_token_embeddings(len(label_ids)*2 + num_tokens)
        # print('model.encoder.embed_tokens.weight.shape[0]', model.encoder.embed_tokens.weight.shape[0])
        assert model.encoder.embed_tokens.weight.shape ==  model.decoder.embed_tokens.weight.shape

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                if token[:3]!='<<<':
                    t = token[2:-2]
                else:
                    t = token[3:-3]
                t = ' ' + ' '.join(t.split('_'))
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(t))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed
                assert model.encoder.embed_tokens.weight.data.shape[1] == 1024
                for i in range(model.encoder.embed_tokens.weight.data.shape[1]):
                    assert model.encoder.embed_tokens.weight.data[index, i] == model.decoder.embed_tokens.weight.data[index, i]

        encoder = model.encoder
        decoder = model.decoder
        decoder.set_dropout(decoder_dropout)

        encoder = FBartEncoder(encoder)
        if decoder_type == 'maxpool':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids, confidence=confidence,
                                      maxpool=True, pos_interval=pos_interval, dropout_hidden=dropout_hidden)
        else:
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids, confidence=confidence,
                                      maxpool=False, pos_interval=pos_interval, dropout_hidden=dropout_hidden)

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, head_token_indexs=None, num_head_token=None, tail_token_indexs=None,
                        cnt_start=None):
        encoder_outputs, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = BartState(encoder_outputs, encoder_mask, src_tokens, head_token_indexs, num_head_token, tail_token_indexs, cnt_start)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, head_token_indexs, num_head_token, tail_token_indexs, cnt_start):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor src_seq_len: src的长度
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, head_token_indexs, num_head_token, tail_token_indexs, cnt_start)
        decoder_output, encoder_logits, _, seq_indices, cnt_start_logits, _ = self.decoder(tgt_tokens, state)
        return {'pred': decoder_output, 'encoder_logits': encoder_logits, 'seq_indices': seq_indices,
                'cnt_start_logits': cnt_start_logits}


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, head_token_indexs, num_head_token, tail_token_indexs, cnt_start):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.head_token_indexs = head_token_indexs
        self.num_head_token = num_head_token
        self.tail_token_indexs = tail_token_indexs
        
        self.past_indices = None
        self.past_seq_indices = None
        self.cnt_start = cnt_start

        self.start_offset = None
        self.src_word_embeds = None
        self.src_output_word_embeds = None

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.past_key_values is not None:
            self.reorder_past_key_values(indices)

        self.num_head_token = self._reorder_state(self.num_head_token, indices)
        self.start_offset = self._reorder_state(self.start_offset, indices)
        self.src_word_embeds = self._reorder_state(self.src_word_embeds, indices)
        self.src_output_word_embeds = self._reorder_state(self.src_output_word_embeds, indices)

    def reorder_past_key_values(self, indices: torch.LongTensor):
        assert self.past_key_values is not None
        new = []
        for layer in self.past_key_values:
            new_layer = {}
            for key1 in list(layer.keys()):
                new_layer_ = {}
                for key2 in list(layer[key1].keys()):
                    if layer[key1][key2] is not None:
                        layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                        # print(key1, key2, layer[key1][key2].shape)
                    new_layer_[key2] = layer[key1][key2]
                new_layer[key1] = new_layer_
            new.append(new_layer)
        self.past_key_values = new

    def set_selected(self, indices, seq_indices):
        self.past_indices = indices
        self.past_seq_indices = seq_indices
