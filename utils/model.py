import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict as ddict
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import json
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torchvision import models

# Cifar10 Data Fetch & Preprocessing & Split 

transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

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


# Model Define 

class image_classification(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.class_num=class_num
        self.conv1=nn.Conv2d(3,16,3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,64,3,stride=1,padding=1)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,self.class_num)

    def forward(self, src):
        # src : B, 3, 32, 32
        out=F.tanh(self.conv1(src)) 
        out=F.max_pool2d(out,2) # B, 16, 16, 16
        out=F.tanh(self.conv2(out))
        out=F.max_pool2d(out,2) # B, 64, 8, 8
        out=F.tanh(self.conv3(out)) 
        out=F.max_pool2d(out,2) # B, 128, 4, 4
        
        out=torch.flatten(out,1)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def train(model, iterator, optimizer, criterion, clip, device):
    
    model.train()
    
    epoch_loss = 0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        
        src = batch[0].to(device)
        target = batch[1].to(device)
        
        output = model(src)
                    
        loss = criterion(output, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def accuracy_score(model,train,valid,test,device):
    model.eval()
    
    total_train=0
    train_acc=0
    total_valid=0
    valid_acc=0
    total_test=0
    test_acc=0
        
    with torch.no_grad():
        for _, batch in enumerate(train):
            total_train+=batch[0].shape[0]
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            output = torch.argmax(output,dim=1)
            train_acc+=torch.sum(torch.where(output==trg,1,0))
    
    train_accuracy=train_acc/total_train

    with torch.no_grad():
        for _, batch in enumerate(valid):
            total_valid+=batch[0].shape[0]
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            output = torch.argmax(output,dim=1)
            valid_acc+=torch.sum(torch.where(output==trg,1,0))
    
    valid_accuracy=valid_acc/total_valid
    
    with torch.no_grad():
        for _, batch in enumerate(test):
            total_test+=batch[0].shape[0]
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            output = torch.argmax(output,dim=1)
            test_acc+=torch.sum(torch.where(output==trg,1,0))
    
    test_accuracy=test_acc/total_test

    print("--------------------------------------------")
    print("Train Accuracy : ",train_accuracy)
    print("Valid Accuracy : ",valid_accuracy)
    print("Test Accuracy : ",test_accuracy)
    print("--------------------------------------------")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




model=image_classification(10).to('cuda')
resnet=models.resnet18(pretrained=False)
num_classes=10
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,num_classes)



resnet=ResNet18()
print(resnet)
resnet.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


optimizer = torch.optim.Adam(resnet.parameters(),lr=0.00005)
criterion=nn.CrossEntropyLoss()

N_EPOCHS = 200
CLIP = 1
best_valid_loss = float('inf')
patient=0

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train(resnet, train_loader, optimizer, criterion, CLIP, device='cuda')
    valid_loss = evaluate(resnet, valid_loader, criterion, device='cuda')
    
    scheduler.step()
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    
    if valid_loss < best_valid_loss:
        patient=0
        best_valid_loss = valid_loss
        torch.save(resnet.state_dict(), '../assets/model/resnet18.pt')
    else:
        patient+=1
        if patient>=200:
            break

resnet.load_state_dict(torch.load('../assets/model/resnet18.pt'))
test_loss = evaluate(resnet, test_loader, criterion,'cuda')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score(model,train_loader,valid_loader,test_loader,device='cuda')







