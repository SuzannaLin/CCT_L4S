import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py
from PIL import Image
from skimage.transform import resize

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='label'):
        self.list_path = list_path
        self.mean = [ -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.0823, 0.0516, 0.3338, 0.7819]
        #self.mean = [1.3656, 1.1936, 1.1444, 1.1096, 1.1315, 1.1459, 1.1293, 1.1034, 1.0854, 1.2597, 0.9031]
        self.std = [ 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.8848, 0.9232, 0.9018, 1.2913]
        #self.std = [0.2348, 0.3511, 0.6504, 0.5251, 0.5077, 0.5297, 0.5623, 0.5997, 0.7323, 0.6452, 0.7878]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        new_size = 320
        #print(datafiles)
        if self.set=='labeled':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']
            #print(type(image)) # <class 'numpy.ndarray'>

            ## to resize to 320
            label = Image.fromarray(label,'L')
            label = label.resize((new_size,new_size))
            image = resize(image,(new_size,new_size,14))

            #print('before transpose',image.shape) # (128, 128, 14)
            image = np.asarray(image, np.float32) 
            #print(image.shape) # (128, 128, 14)
            label = np.asarray(label, np.uint8)
            image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:]
            #print(image.shape) # (11, 128, 128)
            #print('before normalization',image[0][1][10:15])
            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]
            #print('after normalization',image[0][1][10:15])
            return image.copy(), label.copy(), name

        else:
            print('no landslide')
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']

            ## to resize to 320
            image = resize(image,(new_size,new_size,14))
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))[[1,2,3,4,5,6,7,10,11,12,13],:,:]
            
            label_shape=(image.shape[-2],image.shape[-1])
            
            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), np.zeros(label_shape),name # torch.Size([11, 128, 128])


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms
    set = 'labeled'
    train_dataset = LandslideDataSet(data_dir='/srvgentjkd98p2/K/Projects/Satellite_Photogrammetry/LandSlide/', list_path='./voc_splits/labeled_L4S.txt',set=set)
    train_loader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=True,pin_memory=True)
    '''    
    channels_sum,channel_squared_sum = 0,0
    num_batches = len(train_loader)
    for data,_,_ in train_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std)'''

    image ,mask, id= next(iter(train_loader))
    print(image[0].shape)
    print(mask[0].shape)
    image = image[0]
    print(image.shape) # torch.Size([11, 320, 320])

    u_v = image.unique()
    print(u_v.amax(),u_v.amin())
    print(u_v)

    print('mask max and min:',mask.unique().amax().item(),mask.unique().amin().item()) # tensor(1, dtype=torch.uint8) tensor(0, dtype=torch.uint8)


    '''    
    mean1 = [-0.3074, -0.1277, -0.0625]
    std1 = [ 0.8775, 0.8860, 0.8869]
    mean2 = [1.3656, 1.1936, 1.1444]
    std2 =  [ 0.2348, 0.3511, 0.6504]
    print('before DeNorm',image[0][1][10:15])
    restore_transform = transforms.Compose([
            DeNormalize(mean1,std1)
    ])
    imageDenorm = restore_transform(image[0:3,:,:])
    print('after deNorm',imageDenorm[0][1][10:15])
    
    restore_transform2 = transforms.Compose([transforms.ToPILImage()])
    plt.imshow(restore_transform2(imageDenorm))
    plt.show()

    image = restore_transform2(image)
    #image = np.transpose(image.numpy())/7
    print(np.max(image),np.min(image))    
    plt.imshow(image)
    #print(image.shape)
    print(type(image))
    plt.show()

    print(mask.shape) # torch.Size([2, 128, 128]) # BS
    mask = mask[0]
    print(mask.unique()) # tensor([0., 1.])

    plt.imshow(mask)
    plt.show()'''


    '''
    sys.exit()

    for images, labels, names in train_loader:
        #print('image shape', images[0].shape)
        print(labels[0].shape)
        #print(names[0])


    
    channels_sum,channel_squared_sum = 0,0
    num_batches = len(train_loader)
    for data,_,_,_ in train_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std) 
    '''
    ''' 
   count_all = 0
    count_landslide = 0
 
    for images, labels, names in train_loader:
        if len(labels[0].unique()) > 1:
            count_landslide += 1
    print(count_landslide)

    train = int(8*math.floor(count_landslide*0.7/8.))
    val = count_landslide-train
    print(train, val)
    '''

    '''count_all = 0 
    count_t = 0
    count_v = 0
    train = 1560


    with open('./voc_splits/70_labeled_L4S_selection.txt','w+') as t:
        with open ('./voc_splits/30_labeled_L4S_selection.txt','w+') as v:
            for images, labels, names in train_loader:
                count_all += 1
                if len(labels[0].unique()) > 1:
                    lijn = names[0] + '\n'
                    #plt.imshow(labels[0])
                    #plt.show()
                    if count_t < train:
                        t.write(lijn)
                        count_t += 1
                    else:
                        v.write(lijn)
                        count_v += 1

    print(count_all)
    print(count_t, count_v)'''
