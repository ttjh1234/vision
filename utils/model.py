import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from XAI.Integrated_gradient.implement.ig_image_pt import *
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import argparse

try:
    from vision.utils.data import *
    from vision.utils.neptune_arg import *
except:
    from data import *
    from neptune_arg import *

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self,block, num_blocks, in_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # conv1 in_channel : Cifar10 : 3, Mnist : 1
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3,
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
    
def ResNet18(in_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2],in_channel=in_channel)

def ResNet34(in_channel):
    return ResNet(BasicBlock, [3, 4, 6, 3],in_channel=in_channel)

def ResNet50(in_channel):
    return ResNet(Bottleneck, [3, 4, 6, 3],in_channel=in_channel)

def ResNet101(in_channel):
    return ResNet(Bottleneck, [3, 4, 23, 3],in_channel=in_channel)

def ResNet152(in_channel):
    return ResNet(Bottleneck, [3, 8, 63, 3],in_channel=in_channel)


class sae_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(28*28,100)
        self.act=nn.SELU()
        self.linear2=nn.Linear(100,30)
    
    def forward(self,x):
        # Input : B, C, H, W -> B, 1, 28,28
        batch_size=x.shape[0]
        
        x=x.view(batch_size,-1)
        out=self.act(self.linear1(x))
        out=self.act(self.linear2(out))
        
        return out

class sae_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(30,100)
        self.linear2=nn.Linear(100,28*28)
        self.act1=nn.SELU()
        
    
    def forward(self,x):
        # Input : B, C, H, W -> B, 1, 28,28
        batch_size=x.shape[0]
        out=self.act1(self.linear1(x))
        out=self.linear2(out)
        out=out.view(batch_size,28,28)
        return out



class stacked_autoencoder(nn.Module):
    def __init__(self,encoder, decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self, x):
        out=self.encoder(x)
        out=self.decoder(out)
        
        return out

class ChannelwisePool(nn.Module):
    def __init__(self,mode='max'):
        super().__init__()
        self.mode=mode
    
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        if self.mode=='max':
            result =  torch.max(input,dim=2).values
        else:
            result = torch.mean(input,dim=2)
        return result.view(n,1,w,h)

class ig_featuremap(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.class_num=class_num
        self.conv1=nn.Conv2d(1,16,3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,64,3,stride=1,padding=1)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.fc1=nn.Linear(1152,512)
        self.fc2=nn.Linear(512,self.class_num)

    def forward(self, src):
        # src : B, 1, 28, 28
        out=F.relu(self.conv1(src)) 
        out=F.max_pool2d(out,2) # B, 16, 14, 14
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2) # B, 64, 7, 7
        out=F.relu(self.conv3(out))
        out=F.max_pool2d(out,2) # B, 128, 3, 3
        
        out=torch.flatten(out,1)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        
        return out


class ig_featureplus(nn.Module):
    def __init__(self,class_num,featext):
        super().__init__()
        self.class_num=class_num
        self.conv1=nn.Conv2d(1,16,3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,64,3,stride=1,padding=1)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.fc1=nn.Linear(1152,512)
        self.fc2=nn.Linear(512,self.class_num)
        self.featext=featext

    def forward(self, src):
        # src : B, 1, 28, 28
        feat=self.featext(src)
        
        out=F.relu(self.conv1(src)) 
        out=F.max_pool2d(out,2) # B, 16, 14, 14
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2) # B, 64, 7, 7
        out=F.relu(self.conv3(out))
        out=F.max_pool2d(out,2) # B, 128, 3, 3
                
        weight=F.sigmoid(feat)     
        out=out+weight*out
        out=torch.flatten(out,1)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        
        return out



class ig_featureplus_depth(nn.Module):
    def __init__(self,class_num,featext,depthpool):
        super().__init__()
        self.class_num=class_num
        self.conv1=nn.Conv2d(1,16,3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,64,3,stride=1,padding=1)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.fc1=nn.Linear(1152,512)
        self.fc2=nn.Linear(512,self.class_num)
        self.featext=featext
        self.depthpool=depthpool

    def forward(self, src):
        # src : B, 1, 28, 28
        feat=self.featext(src)
        
        out=F.relu(self.conv1(src)) 
        out=F.max_pool2d(out,2) # B, 16, 14, 14
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2) # B, 64, 7, 7
        out=F.relu(self.conv3(out))
        out=F.max_pool2d(out,2) # B, 128, 3, 3
        
        depth=self.depthpool(feat)
        depth_size=depth.size()
        depth_weight=F.softmax(depth.view(depth.size()[0],1,-1)/torch.sqrt(depth.max()),dim=2).view(depth_size)
        weight=F.sigmoid((1+feat)*depth_weight)
        out=out+weight*out
        out=torch.flatten(out,1)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        
        return out







def train(model, iterator, optimizer, criterion, device, run_flag=0):
    
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
        if run_flag==1:
            run["train/train_iter_loss"].log(loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device, run_flag=0):
    
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if run_flag==1:
                run["valid/valid_iter_loss"].log(loss.item())

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

def sae_train(model, iterator, optimizer, criterion, device, run_flag=0):
    
    model.train()
    
    epoch_loss = 0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        batch=torch.squeeze(batch[0],dim=1)
        src = batch.to(device)
        target = batch.to(device)
        
        output = model(src)
                    
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if run_flag==1:
            run["train/train_iter_loss"].log(loss.item())

    return epoch_loss / len(iterator)

def sae_evaluate(model, iterator, criterion, device, run_flag=0):
    
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            batch=torch.squeeze(batch[0],dim=1)
            src = batch.to(device)
            trg = batch.to(device)
            output = model(src)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if run_flag==1:
                run["valid/valid_iter_loss"].log(loss.item())

    return epoch_loss / len(iterator)



if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(description='Model Train')
    parser.add_argument('--dataset',required=True, help='Dataset you want to fit')
    parser.add_argument('--experiment',required=True, default='0',help='Experiment Number')
    parser.add_argument('--run',required=True, default='0',help='If run neptune')
    args=parser.parse_args()
    
    data_set=args.dataset
    experiment=args.experiment
    run_flag=int(args.run)
    
    if run_flag==1:
        path='../assets/neptune/neptune_args.txt'
        neptune_key=fetch_neptune_key(path)

        run = neptune.init(
            project=neptune_key['project'],
            api_token=neptune_key['api_token'],
        )

    if data_set=='cifar10':    
        # Cifar10 Data Fetch & Preprocessing & Split 
        train_loader,valid_loader,test_loader,classes=data_loader_cifar10()

        # Model
        resnet=ResNet18(3)
        resnet.to('cuda')

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        N_EPOCHS = 200
        CLIP = 1
        best_valid_loss = float('inf')
        patient=0

        for epoch in tqdm(range(N_EPOCHS)):
            
            start_time = time.time()
            
            train_loss = train(resnet, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
            valid_loss = evaluate(resnet, valid_loader, criterion, device='cuda',run_flag=run_flag)
            
            scheduler.step()
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            if run_flag==1:
                run["train/train_epoch_loss"].log(train_loss)
                run["valid/valid_epoch_loss"].log(valid_loss)
            
            
            if valid_loss < best_valid_loss:
                patient=0
                best_valid_loss = valid_loss
                torch.save(resnet.state_dict(), '../assets/model/resnet18-{}.pt'.format(experiment))
            else:
                patient+=1
                if patient>=200:
                    break

        run.stop()
        resnet.load_state_dict(torch.load('../assets/model/resnet18-{}.pt'.format(experiment)))
        test_loss = evaluate(resnet, test_loader, criterion,'cuda')
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        accuracy_score(resnet,train_loader,valid_loader,test_loader,device='cuda')

    elif data_set=='mnist':

        # Mnist Data Fetch & Preprocessing & Split 
        train_loader,valid_loader,test_loader,classes=data_loader_mnist()

        # Model
        resnet=ResNet18(1)
        resnet.to('cuda')

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        N_EPOCHS = 200
        CLIP = 1
        best_valid_loss = float('inf')
        patient=0

        for epoch in tqdm(range(N_EPOCHS)):
            
            start_time = time.time()
            
            train_loss = train(resnet, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
            valid_loss = evaluate(resnet, valid_loader, criterion, device='cuda',run_flag=run_flag)
            
            scheduler.step()
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            #print(f'\tTrain Loss: {train_loss:.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f}')
            run["train/train_epoch_loss"].log(train_loss)
            run["valid/valid_epoch_loss"].log(valid_loss)
            
            if valid_loss < best_valid_loss:
                patient=0
                best_valid_loss = valid_loss
                torch.save(resnet.state_dict(), '../assets/model/mnist_resnet18-{}.pt'.format(experiment))
            else:
                patient+=1
                if patient>=200:
                    break

        run.stop()
        resnet.load_state_dict(torch.load('../assets/model/mnist_resnet18-{}.pt'.format(experiment)))
        test_loss = evaluate(resnet, test_loader, criterion,'cuda')
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        accuracy_score(resnet,train_loader,valid_loader,test_loader,device='cuda')

    else:
       raise NotImplementedError
