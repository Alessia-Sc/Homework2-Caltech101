from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.dataset={}
        classes_dict={}
        self.n_instances=0
        n_class=0

        data_idx=set(np.loadtxt('Caltech101/'+split+'.txt',dtype='str'))
        self.classes=os.listdir(root)
        self.classes.remove('BACKGROUND_Google')
  

        for dir in self.classes:
          classes_dict[dir]=n_class
          n_class+=1
          images = root+'/'+dir  
          for file in images:
              if dir+'/'+file in data_idx: 
                self.dataset[self.n_instances]=(pil_loader(root+'/'+dir+'/'+file), classes_dict[dir])
                self.n_instances+=1


        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
      
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.dataset[index]
        
        
        ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = self.n_instances # Provide a way to get the length (number of elements) of the dataset
        return length

    def __getsplit__(self, train_size = 0.5):
        images, labels = [], []
        shuffled_data = StratifiedShuffleSplit(1,train_size=train_size)

        for value in self.dataset.values():
            images.append(value[0])
            labels.append(value[1])

        for x, y in shuffled_data.split(images,labels):
            train_indexes = x
            val_indexes = y 

        return train_indexes, val_indexes
