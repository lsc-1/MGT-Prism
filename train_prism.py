


import torch
import argparse
from loguru import logger
from datetime import datetime
from torch.nn import DataParallel
from torch.optim import AdamW
from lib.data_loader_prism import get_data_loader,get_data_loader_all

from lib.models.networks import get_model, get_tokenizer

from lib.training.common_prism import test_acc,train_common_machine_mix_all,compute_statistics_all_mean
from lib.training.RecAdam import RecAdam
from lib.exp import get_num_labels, seed_everything
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import random
from tqdm import tqdm
from transformers import logging as hf_logging
import logging
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--type_base1_value', type=int, default=0.6)
    parser.add_argument("--scl_reg", default=1, type=float)
    parser.add_argument('--output_name', default='model.pt',
                    type=str, required=False, help='model save name')
    parser.add_argument('--device', default='1',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model', default='MGT-edited/roberta-base', help='pretrained model type',choices=['beyond/roberta-base','beyond/XLNet','beyond/ChatGLM2-6B'])
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--dataset', default='machine_human', help='training dataset')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        required=False, help='training batch size')
    parser.add_argument('--lr', default=2e-5, type=float,
                        required=False, help='learning rate')
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--shift_reg", default=0.0, type=float)
    parser.add_argument("--shift11", default=0, type=float)
    parser.add_argument('--log_step', default=200, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)
    parser.add_argument('--output_dir', default='model_prism/model_zh_2',#llama0456
                    type=str, required=False, help='save directory')
    parser.add_argument('--log_file', type=str, default='./model_prism/model_zh_2/train.log')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='save checkpoint every epoch')

    
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'recadam'])
    parser.add_argument("--type_base1", default=4, type=int)
    parser.add_argument("--type_base2", default=601, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--shift_now", default=(0.0), type=float)



    parser.add_argument("--loss_type", default='mae', choices=['ce', 'scl', 'mae'])

    parser.add_argument("--eval_metric", default='acc', type=str, choices=['acc', 'f1'])
    parser.add_argument("--save_steps", default=-1, type=int)
    args = parser.parse_args()
    return args
    

def main():
    args = args_parser()
    seed_everything(args.seed)
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    output_dir = args.output_dir
    output_name = args.output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels
    model = get_model(args)
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))

    if torch.cuda.device_count() > 1:
        logger.info("Let's use " + str(len(args.device.split(','))) + " GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(args.dataset, 'train', tokenizer, args.batch_size, shuffle=True)
    
    train_loader_all = get_data_loader_all(args.dataset, 'train', tokenizer, args.batch_size)
    
    val_loader = get_data_loader(args.dataset, 'dev', tokenizer,32)

    logger.info("dataset {} loaded".format(args.dataset))
    logger.info("num_labels: {}".format(num_labels))
    model.train()
    model.to(device)
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    if torch.cuda.device_count() > 0:
        pretrained_model = DataParallel(pretrained_model, device_ids=[int(i) for i in args.device.split(',')])
    pretrained_model.eval()
    no_decay = ["bias", "LayerNorm.weight"]
    if args.optimizer == 'adam':
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    logger.info('starting training')
    

    high_mean_global, mid_mean_global,low_mean_global = compute_statistics_all_mean(train_loader_all, pretrained_model,tokenizer, device, prominence=args.type_base1_value, sub_batch_size=args.batch_size)

    best_acc = 0
    step_counter = 0
    for epoch in range(args.epochs):
        model.train()
        pretrained_model.eval()
        logger.info("epoch {} start".format(epoch))
        start_time = datetime.now()
        
        step_counter = train_common_machine_mix_all(model, optimizer, train_loader,tokenizer, high_mean_global, mid_mean_global,low_mean_global,
                                    log_steps=args.log_step, pre_model=pretrained_model,
                                    shift11=args.shift11,
                                    loss_type=args.loss_type, scl_reg=args.scl_reg,
                                    save_steps=args.save_steps, save_dir=output_dir,step_counter=step_counter,type_base1=args.type_base1_value,device=args.device)

        acc, zero_acc, one_acc, f1,recall,human_recall,machine_recall,auroc= test_acc(model, val_loader, args.eval_metric)
        logger.info("epoch {} validation {}: {:.4f} ".format(epoch, args.eval_metric, acc))
        logger.info("epoch {} validation_f1: {:.4f} , human_zero_acc:{:.4f}, machine_one_acc:{:.4f}".format(epoch, f1,zero_acc,one_acc))
        logger.info("epoch {} val_recall:{:.4f},val_human_recall:{:.4f},machine_recall:{:.4f} ".format(epoch,recall,human_recall,machine_recall))
        logger.info("epoch {} AUROC {}: {:.4f} ".format(epoch, args.eval_metric, auroc))


        if acc > best_acc:
            best_acc = acc

            
            logger.info("best validation {} improved to {:.4f},type_base1:{}".format(
                args.eval_metric, best_acc,args.type_base1_value))
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/{}'.format(output_dir, output_name))
            logger.info("model saved to {}/{}".format(output_dir, output_name))
        if args.save_every_epoch:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/epoch{}_{}'.format(output_dir, epoch, output_name))
            logger.info("model saved to {}/epoch{}_{}".format(output_dir, epoch, output_name))
        end_time = datetime.now()
        logger.info('time for one epoch: {}'.format(end_time - start_time))
        
def test_acc2(output_name='model.pt', data_name='test', name='train'):
    args = args_parser()
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels

    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    test_loader = get_data_loader(args.dataset, 'test', tokenizer, 32)
    model = get_model(args)
    model.to(device)
    model.load_state_dict(torch.load('{}/{}'.format(args.output_dir, args.output_name)))
    logger.info("best model loaded")
    acc, zero_acc, one_acc, f1,recall,human_recall,machine_recall,auroc= test_acc(model, test_loader, args.eval_metric)
    logger.info("test {}: {:.4f}".format(args.eval_metric, acc))
    logger.info("test_f1:{:.4f}".format(f1))
    logger.info("test_recall:{:.4f}".format(recall))
    logger.info("human_recall:{:.4f},machine_recall:{:.4f}".format(human_recall,machine_recall))
    logger.info("test {}: {:.4f}".format(args.eval_metric, auroc))
    
    return acc, f1,recall,human_recall,machine_recall


if __name__ == '__main__':
    args = args_parser()
    #main()
    acc1, f1,recall,human_recall,machine_recall = test_acc2(output_name=args.output_name, data_name='test')
    print('train_acc:', acc1)
    print('train_f1:', f1)
    print('recall:', recall)
    print('train_human_recall:', human_recall)
    print('train_machine_recall:', machine_recall)

