import argparse
from doctest import OutputChecker
#import scipy, math
#from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
#from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from base import BaseDataSet, BaseDataLoader
import os
from tqdm import tqdm
from math import ceil
from pathlib import Path
from skimage.transform import resize

import h5py


class testDataset(Dataset):
    def __init__(self, images):
        mean = [ -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819]
        std = [ 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913]
        images_path = Path(images)
        
        self.filelist = list(images_path.glob("*.h5")) # changed .jpg to .tif 
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        datafiles = self.files[index]

        with h5py.File(datafiles['img'], 'r') as hf:
            image = hf['img'][:]
        name = datafiles['name']
        print(name)
            
        image = np.asarray(image, np.float32)
        image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:]
        
        label_shape=(image.shape[-2],image.shape[-1])

        for i in range(len(self.mean)):
            image[i,:,:] -= self.mean[i]
            image[i,:,:] /= self.std[i]

        return image.copy(), name

class LandSlideDataset(Dataset):
    def __init__(self, images):
        self.mean = [ -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [ 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913]
        self.img_ids = []
        self.base_size = 320
        for file in os.listdir(images):
            if file.endswith('.h5'):
                self.img_ids.append(file) # ['image_1.h5', 'image_10.h5'
        #print(images_path) # /srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/ValidationData/img
        self.to_tensor = transforms.ToTensor()
        self.palette = [255,255,255] + [0,0,0] # white and black
        self.files = []
    
        for image_id in self.img_ids:
            self.files.append({
                'img': images + '/' + image_id,
                'name': image_id
            })

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        datafiles = self.files[index]

        with h5py.File(datafiles['img'], 'r') as hf:
            image = hf['img'][:]
        name = datafiles['name']            
        
        ## to resize to 320
        image = resize(image,(self.base_size,self.base_size,14))
            
        image = np.asarray(image, np.float32)
        image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:]
        
        label_shape=(image.shape[-2],image.shape[-1])

        '''
        # add NDVI, GNDVI, BRightness
        ndvi =  (image[[3],:,:] - image[[2],:,:]) / (image[[3],:,:] + image[[2],:,:]) # NIR - R
        image = np.concatenate((image,ndvi))
        gndvi = (image[[3],:,:] - image[[1],:,:]) / (image[[3],:,:] + image[[1],:,:]) # NIR - G
        image = np.concatenate((image,gndvi))
        br = 1 / (image[[2],:,:] +image[[1],:,:] +image[[0],:,:] ) # 1 / (R + G + B)
        image = np.concatenate((image,br))
        '''
        
        for i in range(len(self.mean)):
            image[i,:,:] -= self.mean[i]
            image[i,:,:] /= self.std[i]
        '''
        # remove 3 channels 
        image1 = image[:4,:,:] # first 4 channels ( B, G, R, IR)
        image2 = image[7:,:,:] # last 7 channels
        image = np.concatenate((image1,image2))
        '''
        #image = image[:4,:,:]
        #image = (self.to_tensor(image))
        return image, name    

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        self.batch_size = 1
        shuffle = False
        num_workers = 8
        self.dataset = LandSlideDataset(kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers)


def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    #print(H,W,image.size(4))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    #print('b',image.shape) # torch.Size([1, 4, 320, 320])
    #print(upsize) # (320, 320)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')
    #print(image.shape) # torch.Size([1, 4, 320, 320])

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        #print(scaled_img.shape) # torch.Size([1, 11, 115, 115]), ([1, 11, 128, 128]), ([1, 11, 140, 140])
        scaled_prediction = upsample(model(scaled_img))
        #print('s',scaled_prediction.shape) # torch.Size([1, 2, 320, 320])
        #print(scaled_prediction[0,0,10,10],scaled_prediction[0,1,10,10]) # tensor(0.4411, device='cuda:0') tensor(2.6165e-06, device='cuda:0')

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(fliped_img))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()
    print('images,',args.images)
    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    #print('config ', config) # config  {'name': 'CCT', 'experim_name': 'ABCE_70_30_unsup_alti', 'n_gpu':  ...
    #scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    scales = [0.90, 1, 1.10]
    #scales = [1]



    # DATA
    testdataset = VOC(args.images) # loads images

    num_classes = config['num_classes'] 
    palette = [255,255,255] + [0,0,0]
    print(len(testdataset))

    # MODEL
    print('model')
    config['model']['supervised'] = True; config['model']['semi'] = False
    
    # model copied from train.py
    pretrained = config['pretrained'] # if true, Loading pretrained model:models/backbones/pretrained/3x3resnet50-imagenet.pth
    
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
    tbar = tqdm(testdataset, ncols=100)

    for index, data in enumerate(tbar):
        image, image_id = data
        #print(image_id[0]) # image_10.h5
        image = image.cuda()

        # PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image, scales, num_classes)
        print('\n')
        print(output[0,50,50],output[1,50,50])
        print('min0',np.amin(output[0]),'max0',np.amax(output[0]))
        #print(output[0].shape) # (320, 320)
        print('min1',np.amin(output[1]),'max1',np.amax(output[1])) # min1 0.0 max1 0.144811749458313
        #output[1,:,:] = output[1,:,:]*int(np.amax(output[0]))
        #print('min1*10',np.amin(output[1]),'max1*10',np.amax(output[1]))
        #print(output[0,50,50],output[1,50,50])

        #output1 = output[1]
        #output1 = np.where(output1 > 0.1, 1, 0)
        #print('min1',np.amin(output1),'max1',np.amax(output1))
        #output1 = resize(output1,(128,128))

        output= resize(output,(2,128,128)) # THIS ONE!
        #print(output[0,50,50],output[1,50,50])
        #print(np.unique(output))
        #print(output.shape)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
        #print(prediction.shape) # (128, 128)
        unique = np.unique(prediction)
        print(unique)
        
        # SAVE RESULTS
        prediction_im = colorize_mask(prediction, palette)
        with h5py.File(args.output+'/mask'+image_id[0][5:],'w') as hf:
            hf.create_dataset('mask', data=prediction_im)
        print('image saved: ','mask'+ image_id[0][5:])
        

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

