import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path


class testDataset(Dataset):
    def __init__(self, images):
        mean = [0.485, 0.456, 0.406, 0.4]
        std = [0.229, 0.224, 0.225, 0.2]
        alti_path = images[:-7] + 'RGEALTI' # removed 'BDORTHO'
        #print(alti_path) # /srvgentjkd98p2/K/Projects/IEEE/val/Angers/RGEALTI
        images_path = Path(images)
        alti_path = Path(alti_path)
        self.filelist = list(images_path.glob("*.tif")) # changed .jpg to .tif 
        self.altilist = list(alti_path.glob("*.tif")) 
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image_path = self.filelist[index]
        image_id = str(image_path).split("/")[-1].split(".")[0]
        #image = Image.open(image_path)
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        #image = self.to_tensor(image) #
        imageSize = (image.shape[1],image.shape[0])
        # get alti
        alti_path = str(self.altilist[index])
        print('get item',alti_path)
        alti = np.asarray(cv2.imread(alti_path,cv2.IMREAD_UNCHANGED))
        alti = np.where(alti < -999, 0, alti)
        alti = cv2.resize(alti, imageSize)
        # stack image and alti
        image = np.dstack((image, alti))
        image = self.normalize(self.to_tensor(image))# no normalization
        # print(image.shape) # torch.Size([4, 2000, 1999])
        return image, image_id

def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    #print(H,W,image.size(4))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(scaled_img))

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(fliped_img))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    #print('config ', config) # config  {'name': 'CCT', 'experim_name': 'ABCE_70_30_unsup_alti', 'n_gpu':  ...
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    # DATA
    testdataset = testDataset(args.images) # loads BDORTHO & ALTI images
    loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    num_classes = config['num_classes'] # 16 # changed to 16
    palette = get_voc_pallete(num_classes)

    # MODEL
    print('model')
    config['model']['supervised'] = True; config['model']['semi'] = False
    
    # model copied from train.py
    pretrained = config['pretrained'] # if true, Loading pretrained model:models/backbones/pretrained/3x3resnet50-imagenet.pth
    print('pretrained in inference, ',pretrained)
    #model = models.CCT(num_classes=num_classes, conf=config['model'],
    #						sup_loss=sup_loss, 
    #						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
    #                        ignore_index=config['ignore_index'],pretrained=pretrained,testing=True) 
    model = models.CCT(num_classes=num_classes, conf=config['model'],pretrained=config['pretrained'],testing=True) 
    print(f'\n{model}\n')
    # goes to CCT, then encoder --- model = 
    print(args.model) #./saved/ABCE_70_30_unsup_alti/best_model.pth
    checkpoint = torch.load(args.model)
    print('checkpoint loaded')
    model = torch.nn.DataParallel(model)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    if args.save and not os.path.exists(args.output):
        print(f'{args.output} created')
        os.makedirs(args.output)

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    labels, predictions = [], []

    for index, data in enumerate(tbar):
        image, image_id = data
        image = image.cuda()

        # PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image, scales, num_classes)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
        ### chaning number of classes ###
        prediction = np.where(prediction == 8, 13, prediction)
        prediction = np.where(prediction == 9, 17, prediction)

        # SAVE RESULTS
        # # # #  remap empty classes # # # # 
        prediction_im = colorize_mask(prediction, palette)
        prediction_im.save(args.output+'/'+image_id[0]+'_prediction.tif')
        print('image saved at ',args.output+'/'+image_id[0]+'_prediction.tif')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default=None, type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--save', action = 'store_true', default = True)
    parser.add_argument('--output', default = 'outputs')
    parser.add_argument('--images', default="/srvgentjkd98p2/K/Projects/IEEE/unlabeled_train/Brest/BDORTHO", type=str,
                        help='Test images for Pascal VOC')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

