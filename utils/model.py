import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import argparse
from torch.nn import init

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


class image_classification_feature(image_classification):
    def forward(self, src):
        out1=F.tanh(self.conv1(src))
        out2=F.max_pool2d(out1,2) # B, 16, 16, 16
        out2=F.tanh(self.conv2(out2))
        out3=F.max_pool2d(out2,2) # B, 64, 8, 8
        out3=F.tanh(self.conv3(out3)) 
        out4=F.max_pool2d(out3,2) # B, 128, 4, 4
        
        out4=torch.flatten(out4,1)
        out=F.relu(self.fc1(out4))
        out=self.fc2(out)
        
        return out, [out1,out2,out3]

class image_classificationAT(image_classification):
    def forward(self, src):
        out1=F.tanh(self.conv1(src))
        out2=F.max_pool2d(out1,2) # B, 16, 16, 16
        out2=F.tanh(self.conv2(out2))
        out3=F.max_pool2d(out2,2) # B, 64, 8, 8
        out3=F.tanh(self.conv3(out3)) 
        out4=F.max_pool2d(out3,2) # B, 128, 4, 4
        
        out4=torch.flatten(out4,1)
        out=F.relu(self.fc1(out4))
        out=self.fc2(out)
        
        return out, [g.pow(2).mean(1) for g in (out1,out2,out3)]

    
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


class ResNetAT(ResNet):
    '''
    Attention maps of ResNet
    
    Overloaded ResNet model to return attention maps.
    '''
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out, [g.pow(2).mean(1) for g in (out1,out2,out3,out4)]

class ResNetFeature(ResNet):
    '''
    Attention maps of ResNet
    
    Overloaded ResNet model to return attention maps.
    '''
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out, [out1,out2,out3,out4]

def ResNet10(in_channel):
    return ResNet(BasicBlock, [1, 1, 1, 1],in_channel=in_channel)

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

def ResNet18AT(in_channel):
    return ResNetAT(BasicBlock, [2, 2, 2, 2],in_channel=in_channel)

def ResNet34AT(in_channel):
    return ResNetAT(BasicBlock, [3, 4, 6, 3],in_channel=in_channel)

def ResNet50AT(in_channel):
    return ResNetAT(Bottleneck, [3, 4, 6, 3],in_channel=in_channel)

def ResNet101AT(in_channel):
    return ResNetAT(Bottleneck, [3, 4, 23, 3],in_channel=in_channel)

def ResNet152AT(in_channel):
    return ResNetAT(Bottleneck, [3, 8, 63, 3],in_channel=in_channel)

def ResNet10Feature(in_channel):
    return ResNetFeature(BasicBlock, [1, 1, 1, 1],in_channel=in_channel)


def ResNet18Feature(in_channel):
    return ResNetFeature(BasicBlock, [2, 2, 2, 2],in_channel=in_channel)

def ResNet34Feature(in_channel):
    return ResNetFeature(BasicBlock, [3, 4, 6, 3],in_channel=in_channel)

def ResNet50Feature(in_channel):
    return ResNetFeature(Bottleneck, [3, 4, 6, 3],in_channel=in_channel)

def ResNet101Feature(in_channel):
    return ResNetFeature(Bottleneck, [3, 4, 23, 3],in_channel=in_channel)

def ResNet152Feature(in_channel):
    return ResNetFeature(Bottleneck, [3, 8, 63, 3],in_channel=in_channel)


class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WRNBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = WRNBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class WideResNetFeature(WideResNet):
    '''
    Attention maps of ResNet
    
    Overloaded ResNet model to return attention maps.
    '''
    def forward(self,x):
        out = self.conv1(x)
        out1 = self.block1(out)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.relu(self.bn1(out3))
        out4 = F.avg_pool2d(out4, 8)
        out = out4.view(-1, self.nChannels)

        return self.fc(out), [out1,out2,out3]


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, cardinality, depth, num_classes, widen_factor=4, dropRate=0):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(1024, num_classes)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)


def train(model, iterator, optimizer, criterion, device, run=None):
    
    model.train()
    
    epoch_loss = 0
    acc_temp=0
    total_sample=0
    
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
        
        label=target.cpu().numpy()
        pred=np.argmax(output.detach().cpu().numpy(),axis=1)
        acc_temp+=np.sum(pred==label)
        total_sample+=label.shape[0]
        
        if run!=None:
            run["train/train_iter_loss"].log(loss.item())

    accuracy=acc_temp / total_sample
    
    return epoch_loss / len(iterator), accuracy

def evaluate(model, iterator, criterion, device, run=None):
    
    model.eval()
    epoch_loss = 0
    acc_temp=0
    total_sample=0

    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            label=trg.cpu().numpy()
            pred=np.argmax(output.detach().cpu().numpy(),axis=1)
            acc_temp+=np.sum(pred==label)
            total_sample+=label.shape[0]
            
            if run!=None:
                run["valid/valid_iter_loss"].log(loss.item())
    
    accuracy=acc_temp / total_sample
    
    return epoch_loss / len(iterator), accuracy

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
    
    return train_accuracy.item(), valid_accuracy.item(), test_accuracy.item()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
            
            train_loss = train(resnet, train_loader, optimizer, criterion, device='cuda',run=run)
            valid_loss = evaluate(resnet, valid_loader, criterion, device='cuda',run=run)
            
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
            
            train_loss = train(resnet, train_loader, optimizer, criterion, device='cuda',run=run)
            valid_loss = evaluate(resnet, valid_loader, criterion, device='cuda',run=run)
            
            scheduler.step()
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            #print(f'\tTrain Loss: {train_loss:.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f}')
            if run_flag==1:
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
