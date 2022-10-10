import sys
import os

sys.path.append('../')
if not os.path.exists('model_param'):
    os.mkdir('model_param')

import time
import numpy as np
import random, torch
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='Share_2013', type=str)
parser.add_argument('--visible_gpu', default='2', type=str)

args = parser.parse_args()
dataset_name = args.dataset_name


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
print('gpu:', args.visible_gpu)

import warnings

warnings.filterwarnings('ignore')
from data.pipe import BartNERPipe
from model.bart import BartSeq2SeqModel
from model.utils import lr_warmup

from fastNLP import Trainer, DataSetIter, Tester
from model.metrics import EntMetric
from model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, cache_results

from fastNLP.core.sampler import SortedSampler
from model.generater import SequenceGeneratorModel
from fastNLP.core.sampler import ConstTokenNumSampler
from fastNLP.core.utils import _move_dict_value_to_device, _build_args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# word: ; bpe: 生成所有的bpe; span: 每一段按照start end生成; span_bpe: 每一段都是start的所有bpe，end的所有bpe
args.bart_name = '/disk3/hyx/huggingface/bart-large'
args.schedule = 'linear'
args.num_beams = 4
data_repeat = 0
args.length_penalty = 1
max_len, max_len_a = 5, 0.5
eos_token_ptr = 0
clip_value = 5
eos_coef = 1.
null_coef = 1.
args.warmup_ratio = 0.01


if dataset_name == 'CADEC':
    decoder_type = 'meanpool'
    pos_interval = 4
    eos_coef = 1.5
    word_coef = 2.3
    cls_coef = 2.5
    encoder_lr = 7.5e-6
    decoder_lr = 1e-5
    normal_epochs = 55
    args.warmup_ratio = 0.1
    start_selective_gen = 35
    decoder_dropout = 0.001
    max_token = 280
    random_seed = 2022
    data_repeat = 2
elif dataset_name == 'Share_2013':
    decoder_type = 'meanpool'
    pos_interval = 3
    eos_coef = 1.5
    word_coef = 2.3
    cls_coef = 2.5
    encoder_lr = 7.5e-6
    decoder_lr = 1e-5
    normal_epochs = 50
    args.warmup_ratio = 0.1
    start_selective_gen = 35
    decoder_dropout = 0.001
    max_token = 280
    random_seed = 1998
    data_repeat = 1
elif dataset_name == 'en_ace04':
    decoder_type = 'maxpool'
    pos_interval = 2
    normal_epochs = 55
    start_selective_gen = 45
    encoder_lr = 1.2e-5
    decoder_lr = 1.6e-5
    word_coef = 1.1
    cls_coef = 1.1
    max_token = 190
    decoder_dropout = 0.02
    args.warmup_ratio = 0.01
    random_seed = 1996
    data_repeat = 1
elif dataset_name == 'en_ace05':
    decoder_type = 'meanpool'
    pos_interval = 4
    normal_epochs = 50
    start_selective_gen = 5
    encoder_lr = 1e-5
    decoder_lr = 1.3e-5
    word_coef = 1.1
    cls_coef = 1.1
    max_token = 220
    decoder_dropout = 0.01
    args.warmup_ratio = 0.01
    random_seed = 1999
    data_repeat = 1
elif dataset_name == 'conll2003':
    decoder_type = 'maxpool'
    pos_interval = 4
    normal_epochs = 50
    start_selective_gen = 45
    encoder_lr = 7.5e-6
    decoder_lr = 1e-5
    word_coef = 1.1
    cls_coef = 1.1
    max_token = 400
    decoder_dropout = 0.05
    args.warmup_ratio = 0.01
    args.num_beams = 1
    random_seed = 1921

num_beams = args.num_beams

length_penalty = args.length_penalty
bart_name = args.bart_name
schedule = args.schedule

#######hyper
#######hyper

cache_fn_ent_data = f"caches/data_{dataset_name}_Entity.pt"
@cache_results(cache_fn_ent_data, _refresh=False)
def get_ent_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name)
    data_bundle = pipe.process_from_file(f'./data/{dataset_name}', repeat=data_repeat)
    
    return data_bundle, pipe.tokenizer, pipe.mapping2id
ent_data_bundle, tokenizer, mapping2id = get_ent_data()

cache_fn_ent_data = f"caches/data_{dataset_name}_Entity_reverse.pt"
@cache_results(cache_fn_ent_data, _refresh=False)
def get_ent_data_reverse():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name)
    data_bundle = pipe.process_from_file(f'./data/{dataset_name}', repeat=data_repeat, style='reverse')
    
    return data_bundle
ent_data_bundle_reverse = get_ent_data_reverse()

print('[Random Seed]:', random_seed)

label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids, decoder_type=decoder_type, decoder_dropout=decoder_dropout,
                                     pos_interval=pos_interval)

model = SequenceGeneratorModel(model, eos_token_id=eos_token_ptr,
                               max_length=max_len, max_len_a=max_len_a, num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_ptr,
                               restricter=None)

model_rev = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids, decoder_type=decoder_type, decoder_dropout=decoder_dropout,
                                     pos_interval=pos_interval)
model_rev = SequenceGeneratorModel(model_rev, eos_token_id=eos_token_ptr,
                               max_length=max_len, max_len_a=max_len_a, num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_ptr,
                               restricter=None)

model_rev.seq2seq_model.encoder = model.seq2seq_model.encoder

import torch
from torch.nn.utils import clip_grad_value_
from datetime import timedelta, datetime

set_seed(random_seed)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print('cuda is unavailable!')
    exit(0)
model.train()
model.to(device)
model_rev.train()
model_rev.to(device)

no_decay = ['LayerNorm', 'layernorm', 'layer_norm']
parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bart_encoder' in n],
        'weight_decay': 1e-2,
        'lr': encoder_lr * 0.8
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bart_encoder' in n],
        'weight_decay': 0.0,
        'lr': encoder_lr * 0.8
    },
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and (not 'bart_encoder' in n)],
        'weight_decay': 1e-2,
        'lr': decoder_lr
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and (not 'bart_encoder' in n)],
        'weight_decay': 0.0,
        'lr': decoder_lr
    },
]

parameters_rev = [
    {
        'params': [p for n, p in model_rev.named_parameters() if not any(nd in n for nd in no_decay) and 'bart_encoder' in n],
        'weight_decay': 1e-2,
        'lr': encoder_lr * 0.6
    },
    {
        'params': [p for n, p in model_rev.named_parameters() if any(nd in n for nd in no_decay) and 'bart_encoder' in n],
        'weight_decay': 0.0,
        'lr': encoder_lr * 0.6
    },
    {
        'params': [p for n, p in model_rev.named_parameters() if not any(nd in n for nd in no_decay) and (not 'bart_encoder' in n)],
        'weight_decay': 1e-2,
        'lr': decoder_lr
    },
    {
        'params': [p for n, p in model_rev.named_parameters() if any(nd in n for nd in no_decay) and (not 'bart_encoder' in n)],
        'weight_decay': 0.0,
        'lr': decoder_lr
    },
]

ent_loss = Seq2SeqLoss(num_labels=len(label_ids), eos_coef=eos_coef, null_coef=null_coef, word_coef=word_coef, cls_coef=cls_coef)
loss_no_cls = Seq2SeqLoss(num_labels=len(label_ids), eos_coef=eos_coef, null_coef=null_coef, word_coef=word_coef, cls_coef=cls_coef,
                            use_cls_loss=False)

ent_metric = EntMetric(eos_token_ptr, num_labels=len(label_ids))
ent_metric_rev = EntMetric(eos_token_ptr, num_labels=len(label_ids), reverse=True)

ent_ds = ent_data_bundle.get_dataset('train')
ent_ds_reverse = ent_data_bundle_reverse.get_dataset('train')

if dataset_name in ['conll2003']:
    ent_ds.concat(ent_data_bundle.get_dataset('dev'))
    ent_data_bundle.delete_dataset('dev')
    ent_ds_reverse.concat(ent_data_bundle_reverse.get_dataset('dev'))
    ent_data_bundle_reverse.delete_dataset('dev')

    ent_eval_ds = ent_data_bundle.get_dataset('test')
    ent_eval_ds_reverse = ent_data_bundle_reverse.get_dataset('test')
else:
    ent_eval_ds = ent_data_bundle.get_dataset('dev')
    ent_eval_ds_reverse = ent_data_bundle_reverse.get_dataset('dev')

ent_test_ds = ent_data_bundle.get_dataset('test')

sampler = ConstTokenNumSampler('src_seq_len', max_token=max_token)
reverse_sampler = ConstTokenNumSampler('src_seq_len', max_token=max_token-20)
test_sampler = ConstTokenNumSampler('src_seq_len', max_token=800)

def train(epoch, train_data, dev_data, train_sampler, eval_sampler, losser, optimizer, warmup_ratio, start_datetime_str,
    eval_start_epoch=None, metric_name=None, metrics=None, metric_key='f', num_workers=4, save_model=False, start_selective_gen=100,
    train_data_rev=None, dev_data_rev=None, train_sampler_rev=None, optimizer_rev=None, metrics_rev=None):
    train_sampler(train_data)
    eval_sampler(dev_data)
    train_data_iter = DataSetIter(train_data, sampler=None, as_numpy=False, num_workers=num_workers, 
                            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=train_sampler)
    eval_data_iter = DataSetIter(dev_data, sampler=None, as_numpy=False, num_workers=num_workers, 
                            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=eval_sampler)

    total_step = len(train_data_iter) * epoch
    step = 0
    print('[total_step]:', total_step)

    if train_data_rev is not None:
        train_sampler_rev(train_data_rev)
        train_data_iter_rev = DataSetIter(train_data_rev, sampler=None, as_numpy=False, num_workers=num_workers, 
                            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=train_sampler_rev)
        eval_data_iter_rev = DataSetIter(dev_data_rev, sampler=None, as_numpy=False, num_workers=num_workers, 
                            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=test_sampler)

        total_step_rev = len(train_data_iter_rev) * epoch
        print('[Reverse][total_step]:', total_step_rev)

    
    best_result = -1
    best_epoch = 0
    best_result_str = ''
    best_res = None
    
    initial_lrs = []
    for group in optimizer.param_groups:
        initial_lrs.append(group['lr'])

    if optimizer_rev is not None:
        initial_lrs_rev = []
        for group in optimizer_rev.param_groups:
            initial_lrs_rev.append(group['lr'])

    if start_selective_gen != 0:
        model.set_selective_gen(False)
        model_rev.set_selective_gen(False)

    start_time = time.time()
    for i in range(epoch):
        if i == start_selective_gen:
            model.set_selective_gen(True)

        loss_list = []
        loss_list_rev = []
        train_data_iter_rev_ = train_data_iter_rev.__iter__()
        train_data_iter_rev_ = list(train_data_iter_rev)
        for cnt, (batch_x, batch_y) in enumerate(train_data_iter):
            step += 1
            # Inverse Generation Training
            if cnt < len(train_data_iter_rev_):
                optimizer_rev.zero_grad()
                batch_x_rev, batch_y_rev = train_data_iter_rev_[cnt]
                _move_dict_value_to_device(batch_x_rev, batch_y_rev, device=device)
                
                batch_x_rev = _build_args(model_rev.forward, **batch_x_rev)
                output = model_rev.forward(**batch_x_rev)

                loss = losser(output, batch_y_rev).mean()
                loss.backward()
                loss_list_rev.append(loss.item())

                clip_grad_value_(model_rev.parameters(), clip_value)
                lr_warmup(step/total_step_rev, warmup_ratio, initial_lrs_rev, optimizer_rev, beta=(1 if epoch>=40 else 0.9))
                optimizer_rev.step()

            optimizer.zero_grad()
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            
            batch_x = _build_args(model.forward, **batch_x)
            output = model.forward(**batch_x)

            loss = losser(output, batch_y).mean()
            loss.backward()
            loss_list.append(loss.item())

            clip_grad_value_(model.parameters(), clip_value)
            lr_warmup(step/total_step, warmup_ratio, initial_lrs, optimizer, beta=(1 if epoch>=40 else 0.9))
            optimizer.step()

        time_used = timedelta(seconds=round(time.time() - start_time))

        print('[Reverse][Epoch {:d}][Step {:d}/{:d}] Avg Loss: {:.6f}'.format(
                            i+1, step, total_step_rev, sum(loss_list_rev) / len(loss_list_rev)))
        print('[Epoch {:d}][Step {:d}/{:d}] Avg Loss: {:.6f}  Time: {}'.format(i+1, step, total_step, sum(loss_list) / len(loss_list), time_used))

        if metrics is not None and i >= eval_start_epoch:
            tester = Tester(eval_data_iter, model, metrics=metrics, verbose=0, use_tqdm=False)
            res = tester.test()
            res_str = tester._format_eval_results(res)
            print(res_str, flush=True)
            if res[metric_name][metric_key] > best_result:
                best_result = res[metric_name][metric_key]
                best_result_str = res_str
                best_res = res[metric_name]
                best_epoch = i + 1
                if save_model:
                    torch.save({'state_dict': model.state_dict()}, 'model_param/%s.model' % start_datetime_str)
        print('', flush=True)

    if save_model:
        torch.save({'state_dict': model.state_dict()}, 'model_param/%s.model_' % start_datetime_str)

    # Reset lr to initial value when finishing one training phase
    for lr, group in zip(initial_lrs, optimizer.param_groups):
        group['lr'] = lr
    if optimizer_rev is not None:
        for lr, group in zip(initial_lrs_rev, optimizer_rev.param_groups):
            group['lr'] = lr
    
    # test without training
    if epoch == 0:
        tester = Tester(eval_data_iter, model, metrics=metrics, verbose=0, use_tqdm=False)
        res = tester.test()
        res_str = tester._format_eval_results(res)
        print(res_str, flush=True)
        best_res = res[metric_name]
    elif metrics is not None:
        print(best_result_str, ', Achieved at Epoch %d' % best_epoch, flush=True)
    
    return best_res


start_datetime_str = datetime.now().strftime('%m-%d-%H-%M')
print("-"*5+"start training"+"-"*5, start_datetime_str)
model.set_reverse(False)
model_rev.set_reverse(True)
best_res = train(normal_epochs, ent_ds, ent_eval_ds, sampler, test_sampler,
                ent_loss, optim.AdamW(parameters), args.warmup_ratio, start_datetime_str,
                eval_start_epoch=1, metric_name='EntMetric', metrics=ent_metric,
                save_model=True, start_selective_gen=start_selective_gen,
                train_data_rev=ent_ds_reverse, dev_data_rev=ent_eval_ds_reverse,
                train_sampler_rev=reverse_sampler, optimizer_rev=optim.AdamW(parameters_rev), metrics_rev=ent_metric_rev,
                )

model.set_selective_gen(True)
print('[Evaluating on test data]')

test_sampler(ent_test_ds)
eval_data_iter = DataSetIter(ent_test_ds, num_workers=4, drop_last=False, batch_sampler=test_sampler)
tester = Tester(eval_data_iter, model, metrics=ent_metric, verbose=0, use_tqdm=False)
res = tester.test()
res_str = tester._format_eval_results(res)
print(res_str)

model.load_state_dict(torch.load('model_param/%s.model' % start_datetime_str)['state_dict'])
model.set_selective_gen(True)

print('[Evaluating on test data]')
test_sampler(ent_test_ds)
eval_data_iter = DataSetIter(ent_test_ds, num_workers=4, drop_last=False, batch_sampler=test_sampler)
tester = Tester(eval_data_iter, model, metrics=ent_metric, verbose=0, use_tqdm=False)
res = tester.test()
res_str = tester._format_eval_results(res)
print(res_str)
