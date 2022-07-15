from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        #print('kwargs: ',kwargs) #kwargs:  {'data_dir': '../IEEE', 'crop_size': 320, 'base_size': 320, 'scale': True, 'augment': True, 'flip': True, 'rotate': False, 'blur': False, 'split': 'train_supervised_320', 'n_labeled_examples': 432, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'ignore_index': 0}
        #self.num_classes = kwargs.pop('num_classes') # 16 # [0,15] # used for color palette function only
        #print('VOC: num_classes = ', self.num_classes)
        #[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128]
        c_0 = [35,31,32] # UA color palette
        c_1 = [219,95,87]
        c_2 = [219,151,87]
        c_3 = [219,208,87]
        c_4 = [173,219,87]
        c_5 = [117,219,87]
        c_6 = [123,196,123]
        c_7 = [88,177,88]
        c_8 = [212,246,212]
        c_9 = [176,226,176]
        c_10 = [0,128,0]
        c_11 = [88,176,167]
        c_12 = [153,93,19]
        c_13 = [87,155,219]
        c_14 = [0,98,255]
        c_15 = [35,31,32]
        self.palette = c_0+c_1+c_2+c_3+c_4+c_5+c_6+c_7+c_8+c_9+c_10+c_11+c_12+c_13+c_14+c_15
        self.palette = pallete.get_voc_pallete(self.num_classes)
        #print(self.palette)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        #print(self.root) # ../IEEE
        #print(self.split) # ./dataloaders/voc_splits/432_train_supervised.txt
        #print(self.n_labeled_examples) # 433
        file_list = self.split
        print(file_list)
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        #self.files, self.labels = list(zip(*file_list))
        self.files, self.alti, self.labels = list(zip(*file_list)) # added alti


    def _load_data(self, index):
        #print(self.files[index]) #/labeled_train/Nantes_Saint-Nazaire/BDORTHO/44-2013-0293-6716-LA93-0M50-E080.tif
        #print(self.files[index][1:]) #labeled_train/Nantes_Saint-Nazaire/BDORTHO/44-2013-0293-6716-LA93-0M50-E080.tif
        #image_path = os.path.join(self.root, self.files[index][1:])
        image_path = self.files[index]#[1:]
        
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0] #44-2013-0335-6707-LA93-0M50-E080
        
        # Pillow can't open multi-spectral imagery, use vc2, GDAL or rasterio
        alti_path = self.alti[index]
        imageSize = (image.shape[1],image.shape[0])
        alti = np.asarray(cv2.imread(alti_path,cv2.IMREAD_UNCHANGED)) 
        alti = np.where(alti < -999, 0, alti)
        alti = cv2.resize(alti, imageSize)
        #print('image shape,', image.shape)#image shape, (2001, 2000, 3)
        #print('alti shape,', alti.shape)#alti shape, (2001, 2000)
      
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = self.labels[index]
        
        # create label with 0 values
        if 'un' in image_path or 'val' in image_path:
            label = np.zeros((image.shape[0],image.shape[1]))
        else:
            label = np.asarray(Image.open(label_path), dtype=np.int32) # without resize
       
        
        image_stack = np.dstack((image, alti))
        ### chaning number of classes ###
        label = np.where(label == 13, 8, label)
        label = np.where(label == 14, 9, label) # changing number of classes!!! Don't forget to change output as well!!
        label = np.where(label == 15, 0, label)
        
        #print('image shape, ', image.shape ,'label shape, ', label.shape)#, 'stack shape, ', image_stack.shape)
        return image_stack, label, image_id #return image, label, image_id

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [ -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819] # add for L4S
        self.STD = [ 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913] #
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 0
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
