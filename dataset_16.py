import os
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms


__all__ = ['Dataset', 'ToTensor', 'GrayscaleNormalization', 'RandomFlip']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.png')))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, '*.png')))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index], cv2.IMREAD_GRAYSCALE) / 255.
        label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE) / 255.
        w,h = img.shape
        
        
        img_d = cv2.resize(img, (16,16), interpolation= cv2.INTER_NEAREST)
        label_d = cv2.resize(label, (16,16), interpolation= cv2.INTER_NEAREST)
        
        
        img = cv2.resize(img, (256, 256), interpolation= cv2.INTER_NEAREST)
        label = cv2.resize(label, (256, 256), interpolation= cv2.INTER_NEAREST)


        idd = self.imgs[index]
        idd = os.path.basename(idd).split('/')[-1]
        idd = idd[:-4]
        ret = {
            'img': img[:, :, np.newaxis],
            'label': label[:, :, np.newaxis],
            'img_d': img_d[:, :, np.newaxis],
            'label_d': label_d[:, :, np.newaxis],
            'id': idd
        }
        
        if self.transform:
            ret = self.transform(ret)
        
        return ret


class ToTensor:
    def __call__(self, data):
        img, label,img_d, label_d, idd  = data['img'], data['label'], data['img_d'], data['label_d'], data['id']
        
        img = img.transpose(2, 0, 1).astype(np.float32)  # torch  (C, H, W) .transpose
        label = label.transpose(2, 0, 1).astype(np.float32)
        img_d = img_d.transpose(2, 0, 1).astype(np.float32)  # torch  (C, H, W)
        label_d = label_d.transpose(2, 0, 1).astype(np.float32)
        
        ret = {
            'img': torch.from_numpy(img),
            'label': torch.from_numpy(label),
            'img_d': torch.from_numpy(img_d),
            'label_d': torch.from_numpy(label_d),
            'id': idd
        }
        return ret


class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        img, label,img_d, label_d, idd  = data['img'], data['label'], data['img_d'], data['label_d'], data['id']
        img = (img - self.mean) / self.std
        img_d = (img_d - self.mean) / self.std
        
        ret = {
            'img': img,
            'label': label,
            'img_d': img_d,
            'label_d': label_d,
            'id': idd
        }
        return ret
    
    
class RandomFlip:
    def __call__(self, data):
       
        img, label,img_d, label_d, idd  = data['img'], data['label'], data['img_d'], data['label_d'], data['id']
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
            
            img_d = np.fliplr(img_d)
            label_d = np.fliplr(label_d)
            
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            label = np.flipud(label)
            
            img_d = np.flipud(img_d)
            label_d = np.flipud(label_d)
            
        ret = {
            'img': img,
            'label': label,
            'img_d': img_d,
            'label_d': label_d,
            'id': idd
        }
        return ret       #print(idd)
        #idd = idd[64:] # for val set 6

