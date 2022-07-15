from re import A
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
from PIL import Image
from skimage.transform import resize
import albumentations as A

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        #print('kwargs: ',kwargs) #kwargs:  {'data_dir': '../IEEE', 'crop_size': 320, 'base_size': 320, 'scale': True, 'augment': True, 'flip': True, 'rotate': False, 'blur': False, 'split': 'train_supervised_320', 'n_labeled_examples': 432, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'ignore_index': 0}
        self.num_classes = kwargs.pop('num_classes') # 16 # [0,15] # used for color palette function only
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
        #self.palette = pallete.get_voc_pallete(self.num_classes)
        #print(self.palette)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        #print(self.root) # ../IEEE
        #print(self.split) # ./dataloaders/voc_splits/432_train_supervised.txt
        #print(self.n_labeled_examples) # 433
        file_list = self.split
        #print(file_list)
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

from torch.utils import data
import h5py

def get_training_augmentation():
    train_transform = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.Transpose(p=0.5),
    ]
    return A.Compose(train_transform)
def get_training_augmentation_2():
    train_transform = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.Transpose(p=0.5),
        # non-rigid transforms
        # not good for S-2
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.8),
    ]
    return A.Compose(train_transform)

def get_training_augmentation_3():
    train_transform = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.Transpose(p=0.5),
        # non-spatial transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,p=0.8), #Randomly change brightness and contrast of the input image. 
            A.RandomGamma(p=0.8) # only a little
        ],p=0.8),
    ]
    return A.Compose(train_transform)

class LandslideDataSet(Dataset):
    def __init__(self, **kwargs):
        list_path = kwargs['list_path']
        data_dir = kwargs["data_dir"]
        self.set = kwargs['set']
        self.mean = kwargs['mean']
        self.std = kwargs['std']
        self.base_size = kwargs['base_size']
        try:
            weak_labels_output = kwargs['weak_labels_output']
        except Exception:
            pass
        self.img_ids = [i_id.strip() for i_id in open(list_path)] #'TrainData/img/image_1.h5'
        #max_iters=kwargs["num_steps_stop"]*self.batch_size
        #self.palette = pallete.get_voc_pallete(self.num_classes)
        self.palette = [255,255,255] + [0,0,0] # white and black
        max_iters = None

        #self.augmentation = get_training_augmentation()
        self.augmentation = False

        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if self.set == 'mixed': ## don't need this one
            for name in self.img_ids:
                img_file = data_dir + name # name = TrainData/img/image_3402.h5 TEST: img/image_114.h5
                if name.startswith('img'):
                    img_file = '/srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/TestData/'+name
                    label_file = '/geomatics/gpuserver-0/andrea/Landslide/train_predictions/mask_'+ name[10:]
                    #print(label_file) # /geomatics/gpuserver-0/andrea/Landslide/train_predictions/mask_102.h5
                if name.startswith('Validation'):
                    label_file = data_dir + name.replace('img','mask_best').replace('image','mask')
                else:
                    label_file = data_dir + name.replace('img','mask').replace('image','mask')
                    #print(label_file)
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })


        if self.set=='labeled':
            for name in self.img_ids:
                #print(name) #TrainData/img/image_1.h5
                img_file = data_dir + name
                #print(img_file) #/srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/TrainData/img/image_1.h5
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                #print(label_file)
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })

        elif self.set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })   
        elif self.set=='weakly':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = weak_labels_output + name.replace('img','mask_best').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        if self.set=='labeled' or self.set=='weakly' or self.set=='mixed':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']

            label = Image.fromarray(label,'L')
            ## to resize to 320
            if self.base_size != 128:
                label = label.resize((self.base_size,self.base_size))
                image = resize(image,(self.base_size,self.base_size,14))


            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.int) # int = torch.int32
            #print(image.shape) # (640, 640, 14)
            #print(label.shape) # (128, 128)

            # for albumentations to work correctly, the channels must be at the last dimension
            if self.augmentation:
                sample = self.augmentation(image=image, mask=label) # random crop, ...
                image, label = sample['image'].transpose(2,0,1), sample['mask']
                #print('shape1',image.shape) # (14, 640, 640)
                #print(label.shape)
                image = image[[1,2,3,4,5,6,7,10,11,12,13],:,:]
            else:
                image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:] #12= DEM, 13=slope


            '''
            # add NDVI, GNDVI, BRightness
            ndvi =  (image[[3],:,:] - image[[2],:,:]) / (image[[3],:,:] + image[[2],:,:]) # NIR - R
            image = np.concatenate((image,ndvi))
            gndvi = (image[[3],:,:] - image[[1],:,:]) / (image[[3],:,:] + image[[1],:,:]) # NIR - G
            image = np.concatenate((image,gndvi))
            br = 1 / (image[[2],:,:] +image[[1],:,:] +image[[0],:,:] ) # 1 / (R + G + B)
            image = np.concatenate((image,br))
            #print('before',type(image))#<class 'numpy.ndarray'>
            '''
            
            #print('shape2',image.shape) # (12, 320, 320)
            
            #print(image[1,10,10])
            #normalization does not change the values by much
            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]      
            #print(image[1,10,10])  
            '''
            # remove 3 channels 
            image1 = image[:4,:,:] # first 4 channels ( B, G, R, IR)
            image2 = image[7:,:,:] # last 7 channels
            #print(image1.shape,image2.shape) # (4, 320, 320) (7, 320, 320)
            image = np.concatenate((image1,image2))
            '''
            # pretrained
            #image = image[:4,:,:]
            #print('shape3',image.shape) # (11, 320, 320)
        
            #print(type(label)) #<class 'numpy.ndarray'>
            return image.copy(), label.copy()#, name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']

            ## to resize to 320
            if self.base_size != 128:
                image = resize(image,(self.base_size,self.base_size,14))
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:]
            
            label_shape=(image.shape[-2],image.shape[-1])
            
            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), np.zeros(label_shape)#, name

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [ -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819] # add for L4S
        self.STD = [ 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913] #
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        #print(kwargs) # {'crop_size': 128, 'base_size': 128, 'scale': True, 'augment': True, 'flip': True, 'rotate': True, 'blur': False, 'data_dir': '/srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/', 'list_path': './dataloaders/voc_splits/70_labeled_L4S.txt', 'set': 'labeled', 'mean': [-0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819], 'std': [0.8775, 0.886, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913], 'ignore_index': 0}
        self.dataset = LandslideDataSet(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)



if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    
    train_dataset = LandslideDataSet(data_dir='/srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/', list_path='./voc_splits/70_labeled_L4S.txt',set='labeled')
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,pin_memory=True)

    for test_images, test_labels in train_loader:
        simple_image = test_images[0]
        print(simple_image.shape)
        simple_label = test_labels[0]
        print(simple_label.shape)