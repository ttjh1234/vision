import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from collections import defaultdict as ddict
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def data_loader_cifar10(path=None):
    # Cifar10 Data Fetch & Preprocessing & Split 

    transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if path:
        train_set = torchvision.datasets.CIFAR10(path+'/train',train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10(path+'/test',train=False,download=False,transform=transform2)

    else:
        train_set = torchvision.datasets.CIFAR10("../assets/data/cifar10/train",train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10("../assets/data/cifar10/test",train=False,download=False,transform=transform2)
            
    # 0 ~ 9 label exists,
    # Some detail, 
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_loader=DataLoader(train_set,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=128,shuffle=False)

    return train_loader, test_loader, classes

def fetch_subset_cifar10(path=None,subset_percent=1,seed=42):
    # Cifar10 Data Fetch & Preprocessing & Split 

    transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if path:
        train_set = torchvision.datasets.CIFAR10(path+'/train',train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10(path+'/test',train=False,download=False,transform=transform2)

    else:
        train_set = torchvision.datasets.CIFAR10("../assets/data/cifar10/train",train=True,download=False,transform=transform)
        test_set = torchvision.datasets.CIFAR10("../assets/data/cifar10/test",train=False,download=False,transform=transform2)
            
    # 0 ~ 9 label exists,
    # Some detail, 
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    trainset_size = len(train_set)
    indices = list(range(trainset_size))
    split = int(np.floor(subset_percent * trainset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])
    
    train_loader=DataLoader(train_set,batch_size=128,sampler=train_sampler)
    test_loader=DataLoader(test_set,batch_size=128,shuffle=False)

    return train_loader, test_loader, classes



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


def data_loader_mnist(path=None):
    transform = transforms.Compose([transforms.RandomCrop(28,4),transforms.ToTensor(),transforms.Normalize((0.3081), (0.1307))])
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.3081), (0.1307))])

    if path:
        trainset=torchvision.datasets.MNIST(path+"/train",train=True,download=False,transform=transform)
        test_set=torchvision.datasets.MNIST(path+"/test",train=False,download=False,transform=transform2)
    else:
        trainset=torchvision.datasets.MNIST("../assets/data/mnist/train",train=True,download=False,transform=transform)
        test_set=torchvision.datasets.MNIST("../assets/data/mnist/test",train=False,download=False,transform=transform2)
    train_dataset, valid_dataset= random_split(trainset, [50000, 10000],generator=torch.Generator().manual_seed(42))

    # label check
    label_dict=ddict(int)
    for i in train_dataset:
        label_dict[i[1]]+=1

    # 0 ~ 9 label exists,    
    classes=sorted(list(label_dict.keys()))

    train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
    valid_loader=DataLoader(valid_dataset,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=128,shuffle=False)

    return train_loader,valid_loader, test_loader, classes

def cifar10_experiment_load(path=None):
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if path==None:
        train_data=np.load('../assets/data/experiment/cifar10_train_data.npy')
        train_label=np.load('../assets/data/experiment/cifar10_train_label.npy')
        valid_data=np.load('../assets/data/experiment/cifar10_valid_data.npy')
        valid_label=np.load('../assets/data/experiment/cifar10_valid_label.npy')
        test_data=np.load('../assets/data/experiment/cifar10_test_data.npy')
        test_label=np.load('../assets/data/experiment/cifar10_test_label.npy')
    
    else:
        train_data=np.load(path+'/cifar10_train_data.npy')
        train_label=np.load(path+'/cifar10_train_label.npy')
        valid_data=np.load(path+'/cifar10_valid_data.npy')
        valid_label=np.load(path+'/cifar10_valid_label.npy')
        test_data=np.load(path+'/cifar10_test_data.npy')
        test_label=np.load(path+'/cifar10_test_label.npy')
    
    train_data=torch.FloatTensor(train_data)
    train_label=torch.LongTensor(train_label)
    valid_data=torch.FloatTensor(valid_data)
    valid_label=torch.LongTensor(valid_label)
    test_data=torch.FloatTensor(test_data)
    test_label=torch.LongTensor(test_label)
    
    train_ds=TensorDataset(train_data,train_label)
    valid_ds=TensorDataset(valid_data,valid_label)
    test_ds=TensorDataset(test_data,test_label)

    train_loader=DataLoader(train_ds,batch_size=128,shuffle=True)
    valid_loader=DataLoader(valid_ds,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_ds,batch_size=128,shuffle=False)

    return train_loader, valid_loader, test_loader, classes