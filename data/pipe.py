from cv2 import repeat
from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from transformers import BartTokenizer
from fastNLP.core.metrics import _bio_tag_to_spans
from fastNLP.io import DataBundle
import numpy as np
from itertools import chain
from fastNLP import Const
from functools import cmp_to_key
import json
from copy import deepcopy
from tqdm import tqdm
import torch


class BartNERPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-large', dataset_name='conll2003', target_type='word'):
        """

        :param tokenizer:
        :param dataset_name:
        """
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

        if dataset_name == 'conll2003':
            self.mapping = {'loc': '<<location>>', 'per': '<<person>>', 'org': '<<organization>>', 'misc': '<<misc>>'}
        elif dataset_name == 'en_ace04' or dataset_name == 'en_ace05':
            self.mapping = {
                'loc': '<<location>>', "gpe": "<<geo_political_region>>", "wea": "<<weapon>>", 'veh': "<<vehicle>>",
                'per': '<<person>>',
                'org': '<<organization>>',
                'fac': '<<facility_buildings>>',
            }  # 记录的是原始tag与转换后的tag的str的匹配关系
        elif dataset_name == 'CADEC':
            self.mapping = {'adr': '<<adverse_drug_reaction>>'}
        elif dataset_name == 'Share_2013':
            self.mapping = {'disorder': '<<disorder>>'}

        cur_num_tokens = self.tokenizer.vocab_size
        self.num_token_in_orig_tokenizer = cur_num_tokens
        self.target_type = target_type
        self.dataset_name = dataset_name

        if dataset_name == 'conll2003':
            self.label_names = ['<<<location>>>', '<<<person>>>', '<<<organization>>>', '<<<misc>>>']
        elif dataset_name == 'CADEC':
            self.label_names = ['<<<adverse_drug_reaction>>>']
        elif dataset_name == 'Share_2013':
            self.label_names = ['<<<disorder>>>']
        elif dataset_name == 'en_ace04' or dataset_name == 'en_ace05':
            self.label_names = ['<<<location>>>', '<<<geo_political_region>>>', '<<<weapon>>>', '<<<vehicle>>>', '<<<person>>>', '<<<organization>>>', '<<<facility_buildings>>>']

    def add_tags_to_special_tokens(self):
        mapping = self.mapping

        tokens_to_add = sorted(list(mapping.values()), key=lambda x: len(x), reverse=True)
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)
        """ Add null_token """
        sorted_add_tokens = sorted_add_tokens + ['<<fragment>>', '<<null>>'] + self.label_names
        
        self.mapping['part'] = '<<fragment>>'
        self.mapping['null'] = '<<null>>'


        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id
        self.tokenizer.add_tokens(sorted_add_tokens)

        self.mapping2id = {}  # 给定转换后的tag，输出的是在tokenizer中的id，用来初始化表示
        self.mapping2targetid = {}  # 给定原始tag，输出对应的数字

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.num_token_in_orig_tokenizer
            self.mapping2id[value] = key_id[0]  #
            self.mapping2targetid[key] = len(self.mapping2targetid)


    def process(self, data_bundle, style='normal', repeat=2):
        """
        支持的DataSet的field为
            raw_words: List[str] 句子的所有词
            entity_tags: List[str] 各entity的tag
            entity_spans: List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair, end是开区间

        :param data_bundle:
        :return:
        """
        self.add_tags_to_special_tokens()
        
        # 转换tag
        target_shift = len(self.mapping) + 1  # 是由于第一位是eos

        labels_info_ids = self.tokenizer.convert_tokens_to_ids(self.label_names)
        # print(labels_info_ids)
        assert self.tokenizer.unk_token_id not in labels_info_ids

        cnt_start_freq = {-100:0, 1:0, 2:0, 3:0, 4:0}

        print('[style]:', style)

        ds_names = data_bundle.get_dataset_names()
        for name in ds_names:
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for idx, ins in enumerate(ds):
                raw_words = ins['raw_words']
                word_bpes = [[self.tokenizer.bos_token_id]]
                for i, word in enumerate(raw_words):
                    bpes = self.tokenizer.tokenize(word, add_prefix_space=(False if i==0 else True))
                    bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                    word_bpes.append(bpes)
                word_bpes.append([self.tokenizer.eos_token_id])

                lens = list(map(len, word_bpes))
                cum_lens = np.cumsum(lens).tolist()
                word_head_token_pos = cum_lens[:-2]
                word_tail_token_pos = [x-1 for x in cum_lens[1:-1]]

                start_word_pos_2_pairs = [[] for _ in word_bpes[1:-1]]
                start_word_pos_2_tgt = [[] for _ in word_bpes[1:-1]]

                word_bpes.append(labels_info_ids)
                word_bpes = list(chain(*word_bpes))
                assert 2 < len(word_bpes) < 500

                entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]
                entity_tags = ins['entity_tags']  # [tag1, tag2...]
                assert len(entity_spans) == len(entity_tags)

                bpe_in_entity_head = []
                bpe_in_entity_tail = []
                bpe_in_entity = []

                type_of_head_token = []

                for (entity, tag) in zip(entity_spans, entity_tags):
                    cur_pair = []
                    num_span = len(entity) // 2
                    if 'reverse' in style:
                        start_word_pos = entity[-1] - 1
                    else:
                        start_word_pos = entity[0]
                    for i in range(num_span):
                        start = entity[2 * i]
                        end = entity[2 * i + 1]

                        if i == 0:
                            bpe_in_entity_head.extend(list(range(cum_lens[start], cum_lens[start + 1])))
                        if i == (num_span - 1):
                            bpe_in_entity_tail.extend(list(range(cum_lens[end - 1], cum_lens[end])))

                        if 'reverse' in style:
                            start = entity[2 * (num_span-1-i)]
                            end = entity[2 * (num_span-1-i) + 1]

                        if style == 'reverse':
                            cur_pair_ = [k+1 for k in range(end-1, start-1, -1)]
                        elif style == 'reverse_hard':
                            cur_pair_ = [end] if start+1 == end else [end, start+1]
                        elif style in ['normal', 'easy']:
                            cur_pair_ = [k+1 for k in range(start, end)]    # +1 是[bos]导致的偏置
                        elif style == 'hard':
                            cur_pair_ = [start+1] if start+1 == end else [start+1, end]
                        else:
                            raise ValueError(f"Unsupported style of target sequence: {style}.")

                        cur_pair.extend([p + target_shift for p in cur_pair_])
                        
                        bpe_in_entity.extend(list(range(cum_lens[start], cum_lens[end])))

                    cur_pair.append(self.mapping2targetid[tag] + 1)  # 加1是由于有shift
                    """ """
                    start_word_pos_2_pairs[start_word_pos].append(cur_pair)

                bpe_in_entity_head = set(bpe_in_entity_head)
                bpe_in_entity_tail = set(bpe_in_entity_tail)
                bpe_in_entity = set(bpe_in_entity)

                cnt_start = ins['cnt_start']
                # ignore the loss of cnt_start[i]==0 by setting -100
                cnt_start = [ (-100 if x == 0 else x) for x in cnt_start ]
                assert len(cnt_start) == len(start_word_pos_2_pairs)
                for i, pair in enumerate(start_word_pos_2_pairs):
                    cnt_start_freq[cnt_start[i]] += 1
                    if len(pair) == 0:
                        tag = self.mapping2targetid['part'] if word_head_token_pos[i] in bpe_in_entity else self.mapping2targetid['null']
                        start_word_pos_2_tgt[i] = [ [i+1 + target_shift, tag+1, 0] ]  # +1 是[bos]导致的偏置

                    else:
                        pair = pair[:4]     # beam size is 4
                        if name=='train' or ('conll2003' == self.dataset_name and name=='dev'):
                            start_word_pos_2_tgt[i] = [ seq + [0] for seq in pair ]     # 0是eos所在的id
                        else:
                            # For evalutaion, only one tgt seq is associated with a word
                            # and use beam search to generate multiple entities starting with the word
                            start_word_pos_2_tgt[i] = [ pair[0] ]

                    type_of_head_token_ = [0, 0, 0, 0]   # not_in_entity, in_entity, tail_of_entity, head_of_entity
                    if word_head_token_pos[i] in bpe_in_entity_head:
                        type_of_head_token_[3] = 1
                    if word_head_token_pos[i] in bpe_in_entity_tail:
                        type_of_head_token_[2] = 1
                    if word_head_token_pos[i] in bpe_in_entity:
                        type_of_head_token_[1] = 1
                    else:
                        type_of_head_token_[0] = 1
                    type_of_head_token.append(type_of_head_token_)

                start_word_pos_2_tgt = list(chain(*start_word_pos_2_tgt))

                new_ins = Instance(tgt_tokens=start_word_pos_2_tgt, head_token_indexs=word_head_token_pos,
                            type_of_head_token=type_of_head_token, tail_token_indexs=word_tail_token_pos, cnt_start=cnt_start)
                if name!='train':
                    new_ins.add_field('target_span', start_word_pos_2_pairs)
                    new_ins.add_field('sample_idx', [idx] * len(start_word_pos_2_pairs))

                # for i, word in enumerate(context_words):
                #     bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                #     bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                #     word_bpes.append(bpes)
                # word_bpes.append([self.tokenizer.eos_token_id])
                
                new_ins.add_field('src_tokens', word_bpes)
                new_ds.append(new_ins)

                # Over Sampling 
                if name == 'train':
                    for _ in range(max(cnt_start) - 1):
                        for _ in range(repeat):
                            repeat_ins = deepcopy(new_ins)
                            new_ds.append(repeat_ins)
        
                # if max(cnt_start) > 1:
                #     print(raw_words)
                #     print(word_bpes)
                #     print(entity_tags)
                #     print(entity_spans)
                #     print(start_word_pos_2_tgt)
                #     print(start_word_pos_2_pairs)
                #     print(type_of_head_token)
                #     print(cnt_start)
                #     print(word_head_token_pos)

            data_bundle.set_dataset(new_ds, name)

        data_bundle.set_ignore_type('target_span', 'sample_idx', 'tail_token_indexs')
        data_bundle.set_pad_val('tgt_tokens', 0)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='head_token_indexs', new_field_name='num_head_token')
        data_bundle.apply_field(lambda x: [len(y) for y in x], field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'head_token_indexs', 'tail_token_indexs', 'cnt_start')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'num_head_token', 'type_of_head_token',
                                'sample_idx', 'cnt_start')

        data_bundle.add_collate_fn(collate_fn)

        return data_bundle

    def process_from_file(self, paths, style='normal', repeat=2) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        if isinstance(paths, str):
            path = paths
        else:
            path = paths['train']
        if 'conll' in path:
            data_bundle = Conll2003NERLoader(style=style).load(paths)
        elif 'en_ace0' in path or 'genia' in path:
            data_bundle = NestedLoader(style=style).load(paths)
        else:
            data_bundle = DiscontinuousNERLoader(style=style).load(paths)

        data_bundle = self.process(data_bundle, style=style, repeat=repeat)
        return data_bundle


def collate_fn(batch):
    # print(batch[0])
    d = [x[1] for x in batch]

    src_tokens = [x['src_tokens'] for x in d]
    src_seq_len = [x['src_seq_len'] for x in d]
    tgt_tokens = [x['tgt_tokens'] for x in d]
    tgt_seq_len = [x['tgt_seq_len'] for x in d]
    head_token_indexs = [x['head_token_indexs'] for x in d]
    type_of_head_token = [x['type_of_head_token'] for x in d]
    num_head_token = [x['num_head_token'] for x in d]
    tail_token_indexs = [x['tail_token_indexs'] for x in d]
    cnt_start = [x['cnt_start'] for x in d]

    # print('src_seq_len', src_seq_len)
    # print('tgt_tokens', tgt_tokens)

    batch_max_length = max(src_seq_len)
    src_tokens = torch.LongTensor([ids + [1 for _ in range(batch_max_length - len(ids))]    # for BART, pad_token_id is 1
                                   for ids in src_tokens])
    src_seq_len = torch.LongTensor(src_seq_len)

    tgt_tokens = list(chain(*tgt_tokens))
    tgt_seq_len = list(chain(*tgt_seq_len))
    batch_max_length = max(tgt_seq_len)
    tgt_tokens = torch.LongTensor([ids + [0 for _ in range(batch_max_length - len(ids))]
                                   for ids in tgt_tokens])
    tgt_seq_len = torch.LongTensor(tgt_seq_len)

    # print(src_tokens.shape, flush=True)
    # print(tgt_tokens.shape, flush=True)

    batch_max_length = max(num_head_token)
    head_token_indexs = torch.LongTensor([ids + [0 for _ in range(batch_max_length - len(ids))]
                                   for ids in head_token_indexs])
    type_of_head_token = torch.FloatTensor(list(chain(*type_of_head_token)))
    num_head_token = torch.LongTensor(num_head_token)

    tail_token_indexs = torch.LongTensor([ids + [0 for _ in range(batch_max_length - len(ids))]
                                   for ids in tail_token_indexs])

    cnt_start = torch.LongTensor([ids + [-100 for _ in range(batch_max_length - len(ids))]
                                   for ids in cnt_start])

    input = {'src_tokens': src_tokens, 'tgt_tokens': tgt_tokens, 'src_seq_len': src_seq_len, 'head_token_indexs': head_token_indexs, 'num_head_token': num_head_token, 'tail_token_indexs': tail_token_indexs, 'cnt_start': cnt_start}
    target = {'tgt_tokens': tgt_tokens, 'tgt_seq_len': tgt_seq_len, 'type_of_head_token': type_of_head_token, 'cnt_start': cnt_start}

    # print(src_tokens.shape, src_seq_len.shape, tgt_tokens.shape, tgt_seq_len.shape)
    # exit(0)

    return input, target


class DiscontinuousNERLoader(Loader):
    def __init__(self, demo=False, style='normal'):
        super(DiscontinuousNERLoader, self).__init__()
        self.demo = demo
        self.style = style

    def _load(self, path):
        """
        entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
        entity_tags: 与上面一样长，是每个tag的分数
        raw_words: List[str]词语
        entity_spans: List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair

        :param path:
        :return:
        """
        print(path)
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        dataset = DataSet()
        
        max_span_len = 1e10
        max_sent_len = 128
        min_start_border_ent = max_sent_len

        for i in range(len(lines)):
            if i % 3 == 0:
                sentence = lines[i]
                ann = lines[i + 1]
                now_ins = Instance()
                sentence = sentence.strip().split(' ')  # 生成的空格
                entities = ann.strip().split('|')
                type_list = []
                entity_index_list = []
                entity_list = []
                all_spans = []
                exist_discont = False
                cnt_start = [0] * len(sentence)
                for entity in entities:
                    if len(entity) == 0:
                        continue
                    # print(entity)
                    span_, type_ = entity.split(' ')
                    span_ = span_.split(',')
                    if int(span_[0]) < max_sent_len <= int(span_[-1]):
                        min_start_border_ent = min(int(span_[0]), min_start_border_ent)
                        continue
                    if int(span_[0]) >= max_sent_len:
                        continue

                    span__ = []
                    for j in range(len(span_) // 2):
                        span__.append([int(span_[2 * j]), int(span_[2 * j + 1]) + 1])

                    if len(span__) > 1:
                        exist_discont = True

                    span__.sort(key=lambda x: x[0])
                    if span__[-1][1] - span__[0][0] > max_span_len:
                        continue
                    str_span__ = []
                    for start, end in span__:
                        str_span__.extend(sentence[start:end])
                    assert len(str_span__) > 0 and len(span__) > 0
                    type_list.append(type_.lower())  # 内部是str
                    entity_list.append(str_span__)
                    entity_index_list.append(list(chain(*span__)))  # 内部是数字
                    all_spans.append([type_.lower(), str_span__, list(chain(*span__))])

                    start = span__[0][0]
                    end = span__[-1][-1]
                    if 'reverse' in self.style:
                        cnt_start[end - 1] += 1
                    else:
                        cnt_start[start] += 1

                if 'reverse' in self.style:
                    all_spans = sorted(all_spans, key=cmp_to_key(cmp_veverse))
                else:
                    all_spans = sorted(all_spans, key=cmp_to_key(cmp))

                new_type_list = [span[0] for span in all_spans]
                new_entity_list = [span[1] for span in all_spans]
                new_entity_index_list = [span[2] for span in all_spans]

                if len(sentence) > 150:
                    continue

                sentence = sentence[:min_start_border_ent]
                cnt_start = cnt_start[:min_start_border_ent]
                cnt_start = [min(x, 4) for x in cnt_start]      # beam size is 4

                now_ins.add_field('entities', new_entity_list)
                now_ins.add_field('entity_tags', new_type_list)
                now_ins.add_field('raw_words', sentence)  # 以空格隔开的words
                now_ins.add_field('entity_spans', new_entity_index_list)
                now_ins.add_field('cnt_start', cnt_start)
                dataset.append(now_ins)
                if self.demo and len(dataset) > 30:
                    break
            else:
                continue

        return dataset


class NestedLoader(Loader):
    def __init__(self, demo=False, style='normal', **kwargs):
        super().__init__()
        self.demo = demo
        self.max_sent_len = 150
        self.style = style

    def _load(self, path):
        def cmp(v1, v2):
            v1 = v1[1]
            v2 = v2[1]
            if v1[0] == v2[0]:
                return v1[1] - v2[1]
            return v1[0] - v2[0]

        ds = DataSet()
        invalid_ent = 0
        max_len = 0
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), leave=False):
                data = eval(line.strip())

                all_entities = data['ners']
                all_sentences = data['sentences']

                assert len(all_entities) == len(all_sentences)

                # TODO 这里，一句话不要超过100个词吧
                new_all_sentences = []
                new_all_entities = []
                for idx, sent in enumerate(all_sentences):
                    ents = all_entities[idx]
                    if len(sent)>self.max_sent_len:
                        has_entity_cross = np.zeros(len(sent))
                        for (start, end, tag) in ents:
                            has_entity_cross[start:end + 1] = 1  # 如果1为1的地方说明是有span穿过的

                        punct_indexes = []
                        for idx, word in enumerate(sent):
                            if word.endswith('.') and has_entity_cross[idx]==0:
                                punct_indexes.append(idx)
                        last_index = 0
                        for idx in punct_indexes:
                            if idx-last_index>self.max_sent_len:
                                new_all_sentences.append(sent[last_index:idx+1])
                                new_ents = [(e[0]-last_index, e[1]-last_index, e[2]) for e in ents if last_index<=e[1]<=idx]  # 是闭区间
                                new_all_entities.append(new_ents)
                                last_index = idx+1
                        if last_index<len(sent):
                            new_all_sentences.append(sent[last_index:])
                            new_ents = [(e[0]-last_index, e[1]-last_index, e[2]) for e in ents if last_index <= e[1]]  # 是闭区间
                            new_all_entities.append(new_ents)
                    else:
                        new_all_sentences.append(sent)
                        new_all_entities.append(ents)
                if sum(map(len, all_entities))!=sum(map(len, new_all_entities)):
                    print("Mismatch number sentences")
                if sum(map(len, all_sentences))!=sum(map(len, new_all_sentences)):
                    print("Mismatch number entities")

                all_entities = new_all_entities
                all_sentences = new_all_sentences

                for i in range(len(all_entities)):
                    all_spans = []
                    raw_words = all_sentences[i]
                    max_len = max(len(raw_words), max_len)
                    ents = all_entities[i]
                    for start, end, tag in ents:
                        # assert start<=end, (start, end, i)
                        if start>end:
                            invalid_ent += 1
                            continue
                        all_spans.append((tag, (start, end+1)))
                        assert end<len(raw_words), (end, len(raw_words))

                    if 'reverse' in self.style:
                        all_spans = sorted(all_spans, key=cmp_to_key(cmp_veverse))
                    else:
                        all_spans = sorted(all_spans, key=cmp_to_key(cmp))

                    entities = []
                    entity_tags = []
                    entity_spans = []
                    cnt_start = [0] * len(raw_words)
                    for tag, (start, end) in all_spans:
                        entities.append(raw_words[start:end])
                        entity_tags.append(tag.lower())
                        entity_spans.append([start, end])

                        if 'reverse' in self.style:
                            cnt_start[end - 1] += 1
                        else:
                            cnt_start[start] += 1

                    prev_contxt = []
                    after_contxt = []

                    if i>0:
                        prev_contxt = all_sentences[:i]
                    if i<len(all_sentences)-1:
                        after_contxt = all_sentences[i+1:]

                    assert len(after_contxt)+len(prev_contxt)==len(all_sentences)-1

                    cnt_start = [min(x, 4) for x in cnt_start]      # beam size is 4

                    ds.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                                       entity_spans=entity_spans, cnt_start=cnt_start,
                                       prev_contxt=prev_contxt, after_contxt=after_contxt))
                if self.demo and len(ds) > 30:
                    break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        print(f"for `{path}`, {invalid_ent} invalid entities. max sentence has {max_len} tokens")
        return ds


class Conll2003NERLoader(ConllLoader):
    r"""
    用于读取conll2003任务的NER数据。每一行有4列内容，空行意味着隔开两个句子

    支持读取的内容如下
    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

        entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
        entity_tags: 与上面一样长，是每个tag的分数
        raw_words: List[str]词语
        entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair, end是开区间

    """

    def __init__(self, demo=False, style='normal'):
        headers = [
            'raw_words', 'target',
        ]
        super().__init__(headers=headers, indexes=[0, 1])
        self.demo = demo
        self.style = style

    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            raw_words = ins['raw_words']
            target = iob2(ins['target'])
            spans = _bio_tag_to_spans(target)
            entities = []
            entity_tags = []
            entity_spans = []
            cnt_start = [0] * len(raw_words)

            for tag, (start, end) in spans:
                entities.append(raw_words[start:end])
                entity_tags.append(tag.lower())
                entity_spans.append([start, end])

                if 'reverse' in self.style:
                    cnt_start[end - 1] += 1
                else:
                    cnt_start[start] += 1

            ds.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                               entity_spans=entity_spans, cnt_start=cnt_start))
            if self.demo and len(ds) > 30:
                break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds


def cmp(v1, v2):
    v1 = v1[-1]
    v2 = v2[-1]
    if v1[0] == v2[0]:
        return v1[-1] - v2[-1]
    return v1[0] - v2[0]

def cmp_veverse(v1, v2):
    v1 = v1[-1]
    v2 = v2[-1]
    if v1[-1] == v2[-1]:
        return v2[0] - v1[0]
    return v1[-1] - v2[-1]

