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
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import hflip

def data_loader_cifar10(path=None,batch_size=128,transform_flag=1):
    # Cifar10 Data Fetch & Preprocessing & Split 

    if transform_flag==1:
        transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

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

    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader, test_loader, classes


def fetch_subset_cifar10(path=None,subset_percent=1,seed=42):
    # Cifar10 Data Fetch & Preprocessing & Split 

    transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

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

def get_mean_std(train_set):

    # calculate mean over each channel (r,g,b)
    mean_r = train_set[:,0,:,:].mean()
    mean_g = train_set[:,1,:,:].mean()
    mean_b = train_set[:,2,:,:].mean()
    mean=np.array((mean_r,mean_g,mean_b)).reshape(-1,1,1)

    # calculate std over each channel (r,g,b)
    std_r = train_set[:,0,:,:].std()
    std_g = train_set[:,1,:,:].std()
    std_b = train_set[:,2,:,:].std()
    std=np.array((std_r, std_g, std_b)).reshape(-1,1,1)
    return mean, std


def rescale_data(x,mean=None,std=None):
    
    '''
    Input x is (C,H,W) image and normalized vec. 
    Output is (H,W,C) image and denormalized vec.
    '''
    
    if not mean:
        mean=np.array((0.4914, 0.4822, 0.4465)).reshape(-1,1,1)
    
    if not std:
        std=np.array((0.247, 0.243, 0.261)).reshape(-1,1,1)
    
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


class experiment_cifar10(Dataset):
    """ KD Experiment Dataset """
    
    def __init__(self, path, train=True, usage='proposed',transform=None):
        
        self.usage=usage
        self.transform=transform
        self.classes=['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        if train==True:
            self.img=np.load(path+'/train_img.npy')
            self.label=np.load(path+'/train_label.npy')
            if self.usage=='attkd':
                self.ig=np.load(path+'/train_scaled_ig.npy')
        
        else:
            self.img=np.load(path+'/test_img.npy')
            self.label=np.load(path+'/test_label.npy')
            if self.usage=='attkd':
                self.ig=np.load(path+'/test_scaled_ig.npy')
        
        if usage=='proposed':
            self.data=np.concatenate([self.img,self.ig],axis=1)
        else:
            self.data=self.img
            
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image=torch.tensor(self.data[idx],dtype=torch.float)
        label=torch.tensor(self.label[idx],dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
                
        return image, label 


class experiment_cifar10_2(Dataset):
    """ KD Experiment Dataset """
    
    def __init__(self, path, train=True, usage='proposed',transform=None):
        
        self.usage=usage
        self.transform=transform
        self.classes=['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        if train==True:
            self.img=np.load(path+'/train_img.npy')
            self.label=np.load(path+'/train_label.npy')
            if self.usage=='attkd':
                self.ig=np.load(path+'/train_ig.npy')
        
        else:
            self.img=np.load(path+'/test_img.npy')
            self.label=np.load(path+'/test_label.npy')
            if self.usage=='attkd':
                self.ig=np.load(path+'/test_ig.npy')
        
        if usage=='proposed':
            self.data=np.concatenate([self.img,self.ig],axis=1)
        else:
            self.data=self.img
            
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image=torch.tensor(self.data[idx],dtype=torch.float)
        label=torch.tensor(self.label[idx],dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
                
        return image, label 




class CifarRandomCrop(nn.Module):
    def __init__(self, size, padding=None, fill=-99, padding_mode="constant", usage='proposed'):
        super().__init__()

        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.output_size = (size, size)
        else:
            assert len(size) == 2
            self.output_size = size

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.fill_value=[-1.989,-1.984,-1.711]
        self.usage=usage
    
    def channelwise_image_pad(self,img,fill_value):
        img[0,:,:]=torch.where(img[0,:,:]==-99,fill_value[0],img[0,:,:])       
        img[1,:,:]=torch.where(img[1,:,:]==-99,fill_value[1],img[1,:,:])
        img[2,:,:]=torch.where(img[2,:,:]==-99,fill_value[2],img[2,:,:])
        
        return img
    
    def forward(self, data):
        
        if self.usage=='proposed':
            img=data[:3,:,:]
            ig=data[3:,:,:]     
            
            if self.padding is not None:
                if isinstance(self.padding,int):
                    padding=[self.padding]*4
                img = F.pad(img, padding, self.padding_mode, self.fill)
                img = self.channelwise_image_pad(img,self.fill_value)
                ig = F.pad(ig, padding, self.padding_mode, 0)

            h, w = img.shape[1], img.shape[2]
    
            new_h, new_w = self.output_size

            top = torch.randint(0, h - new_h, size=(1,)).item()
            left = torch.randint(0, w - new_w, size=(1,)).item()

            image = img[:,top: top + new_h,
                        left: left + new_w]
            
            ig = ig[:,top: top + new_h,
                        left: left + new_w]
            
            crop_data=torch.cat([image,ig],dim=0)
        else:
            img=data
            
            if self.padding is not None:
                if isinstance(self.padding,int):
                    padding=[self.padding]*4
                img = F.pad(img, padding, self.padding_mode, self.fill)
                img = self.channelwise_image_pad(img,self.fill_value)

            h, w = img.shape[1], img.shape[2]
    
            new_h, new_w = self.output_size

            top = torch.randint(0, h - new_h, size=(1,)).item()
            left = torch.randint(0, w - new_w, size=(1,)).item()

            crop_data = img[:,top: top + new_h,
                        left: left + new_w]
            
        return crop_data


class CifarRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5, usage='proposed'):
        super().__init__()

        assert (p>=0) & (p<=1)
        self.usage=usage
        self.p=p
    
    def forward(self, data):
        if self.usage=='proposed':
            if torch.rand(1)<self.p:
                img=data[:3,:,:]
                ig=data[3:,:,:]
                flip_img=hflip(img)
                flip_ig=hflip(ig)
                data=torch.cat([flip_img,flip_ig],dim=0)
        else:
            if torch.rand(1)<self.p:
                img=data
                data=hflip(img)
        return data