import os
import json
import argparse
import torch
#import dataloaders
from dataloaders.landslide_dataset import LandslideDataSet
from torch.utils.data import DataLoader
#from dataloaders import landslide_dataset
from dataloaders.voc import VOC
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import BinaryDiceLoss, DiceLoss, abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch.nn.functional as F



def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    torch.manual_seed(42)
    train_logger = Logger()
    print(config["experim_name"])
    # DATA LOADERS
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']

    config['train_supervised']['num_steps_stop']=config['trainer']['epochs']
    config['train_unsupervised']['num_steps_stop']=config['trainer']['epochs']
    config['val_loader']['num_steps_stop']=config['trainer']['epochs']
    



    #train_dataset = LandslideDataSet(config['train_supervised'])
    #print(len(train_dataset)) # 5056
    #supervised_loader = DataLoader(train_dataset, batch_size = config['train_supervised']['batch_size'],pin_memory=True)
    
    supervised_loader = VOC(config['train_supervised'])
    val_loader = VOC(config['val_loader'])
    unsupervised_loader = VOC(config['train_unsupervised'])

    #val_dataset = LandslideDataSet(config['val_loader'])
    #val_loader = DataLoader(val_dataset, batch_size = config['val_loader']['batch_size'],pin_memory=True)
    #unsupervised_dataset = LandslideDataSet(config['train_unsupervised'])
    #unsupervised_loader = DataLoader(unsupervised_dataset,batch_size = config['train_unsupervised']['batch_size'], pin_memory=True)
  
    num_classes = config['num_classes']
    iter_per_epoch = len(supervised_loader) # need to adjust?
    #print(iter_per_epoch) # 1560/8=195
    pretrained = config['pretrained']
    ignore_index = config['ignore_index']
    #if not pretrained:
    #    pretrained = None
    #print(pretrained)

    def BCE_loss(input_logits, target_targets, curr_iter=None, epoch=None,ignore_index=None, temperature=1):
        #print(target_targets.shape) #torch.Size([2, 128, 128])
        #print(input_logits.shape) #torch.Size([2, 2, 128, 128])
        #target_targets = target_targets.long() # expected scalar type Long but found Float
        #print(target_targets.unique()) #tensor([0, 1], device='cuda:0')
    
        return F.binary_cross_entropy_with_logits(input_logits, target_targets)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'BCE':
        sup_loss = BCE_loss
    elif config['model']['sup_loss'] == 'FL':
        #alpha = get_alpha(supervised_loader,num_classes) # calculate class occurences -> frequencies
        alpha = [1,41] # from get_alpha
        #alpha = [1,25] # selection
        #alpha = [ 4, 15, 28, 200, 220, 30, 127, 3, 60, 60, 4, 10, 60] # updated to adjust
        print('train file alpha, ',alpha)
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, ignore_index = ignore_index, alpha = alpha, gamma = 2, smooth = 1e-5)
    elif config['model']['sup_loss'] == 'DL':
        sup_loss = DiceLoss(ignore_index=None, reduction='mean')
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=num_classes) #num_classes=val_loader.dataset.num_classes

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)
    #print(sup_loss) # abCE_loss()
    #print(cons_w_unsup) # <utils.losses.consistency_weight object at 0x7fea8c88ea10>
    model = models.CCT( num_classes=num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
                            ignore_index=ignore_index,pretrained=pretrained) #num_classes=val_loader.dataset.num_classes
    
    print(f'\n{model}\n') # Nbr of trainable parameters: 46710744
    
    
    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger,
        num_classes= num_classes)
    #print('trainer created')
    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_L4S.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=all, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    main(config, args.resume)
