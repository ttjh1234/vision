import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from collections import defaultdict as ddict
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import torchvision
import torchvision.transforms as transforms

def data_loader_cifar10(path=None):
    # Cifar10 Data Fetch & Preprocessing & Split 

    transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if path:
        trainset = torchvision.datasets.CIFAR10(path+'/train',train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10(path+'/test',train=False,download=False,transform=transform)

    else:
        trainset = torchvision.datasets.CIFAR10("../assets/data/cifar10/train",train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10("../assets/data/cifar10/test",train=False,download=False,transform=transform)
    train_dataset, valid_dataset= random_split(trainset, [45000, 5000],generator=torch.Generator().manual_seed(42))

    # label check
    label_dict=ddict(int)
    for i in train_dataset:
        label_dict[i[1]]+=1
        
    # 0 ~ 9 label exists,
    # Some detail, 
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
    valid_loader=DataLoader(valid_dataset,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=128,shuffle=False)

    return train_loader,valid_loader,test_loader, classes


def rescale_data(x,mean=None,std=None):
    
    '''
    Input x is (C,H,W) image and normalized vec. 
    Output is (H,W,C) image and denormalized vec.
    '''
    
    if not mean:
        mean=np.array((0.4914, 0.4822, 0.4465)).reshape(-1,1,1)
    
    if not std:
        std=np.array((0.2023, 0.1994, 0.2010)).reshape(-1,1,1)
    
    x=x*std+mean
    x=np.transpose(x,[1,2,0])
    return x

