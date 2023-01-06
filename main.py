import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
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
from XAI.Integrated_gradient.implement.ig_image_pt import *
from vision.utils.data import *
from vision.utils.model import *
from vision.utils.vision import *

# CIFAR10

train_loader, valid_loader , test_loader, classes = data_loader_cifar10('./assets/data/cifar10')
resnet=ResNet18()
resnet.load_state_dict(torch.load('./assets/model/resnet18.pt'))
resnet.to('cuda')
sm=nn.Softmax(dim=1)

baseline=torch.zeros(3,32,32).to('cuda')

labelname=ddict(int)

'''
for n,i in enumerate(test_loader):
    for j in range(i[0].shape[0]):
        test_image=i[0][j]
        test_label=i[1][j]
        labelname[test_label.item()]+=1

        ig=integrated_gradients(baseline,test_image.to('cuda'),model=resnet,label=test_label.item())
        ig=ig.to('cpu').detach().numpy()
        img=test_image.to('cpu').numpy()
        vi3=scaled_attribute(ig)
        img2=rescale_data(img)

        print("-----------------------------------------------------")
        print("Label : ",test_label.item(),',',classes[test_label.item()])
        prob=sm(resnet(test_image.unsqueeze(0).to('cuda'))).detach().to('cpu').numpy()
        print("Predicted Class : ",np.argmax(prob),f'\t prob: {prob[0,np.argmax(prob)]:.3f}')
        print("-----------------------------------------------------")
        
        sub_path=classes[test_label]+'/'+str(labelname[test_label.item()])+".png"
        save_results("./assets/result/cifar10/"+sub_path,img2,vi3)
'''        

for i in test_loader:
    break

test_image=i[0][11]
test_label=i[1][11]

ig=integrated_gradients(baseline,test_image.to('cuda'),model=resnet,label=test_label.item())
ig=ig.to('cpu').detach().numpy()
img=test_image.to('cpu').numpy()
vi3=scaled_attribute(ig)
img2=rescale_data(img)

print("Label : ",test_label.item(),',',classes[test_label])
prob=sm(resnet(test_image.unsqueeze(0).to('cuda'))).detach().to('cpu').numpy()
print("Predicted Class : ",np.argmax(prob),f'\t prob: {prob[0,np.argmax(prob)]:.3f}')

plot_attribute_map(img2,vi3)
plot_image_overlay(img2,vi3)

igact=np.where(vi3>=np.quantile(vi3,0.90),1.0,0.0)
plt.imshow(igact)

save_results("./assets/result/"+str(11)+".png",img2,vi3)


# MNIST

train_loader, valid_loader , test_loader, classes = data_loader_mnist('./assets/data/mnist')
resnet=ResNet18(1)
resnet.load_state_dict(torch.load('./assets/model/mnist_resnet18.pt'))
resnet.to('cuda')
sm=nn.Softmax(dim=1)

mnist_mean=np.array(0.3081) 
mnist_std=np.array(0.1307)

baseline=torch.zeros(1,28,28).to('cuda')

for i in test_loader:
    break


test_image=i[0][8]
test_label=i[1][8]

ig=integrated_gradients(baseline,test_image.to('cuda'),model=resnet,label=test_label.item())
ig=ig.to('cpu').detach().numpy()
img=test_image.to('cpu').numpy()
vi3=scaled_attribute(ig)
img2=rescale_data(img,mnist_mean,mnist_std)

print("Label : ",test_label.item(),',',classes[test_label])
prob=sm(resnet(test_image.unsqueeze(0).to('cuda'))).detach().to('cpu').numpy()
print("Predicted Class : ",np.argmax(prob),f'\t prob: {prob[0,np.argmax(prob)]:.3f}')


plot_attribute_map(img2,vi3)
plot_image_overlay(img2,vi3)

igact=np.where(vi3>=np.quantile(vi3,0.90),1.0,0.0)
plt.imshow(igact)

# save attribute map
# save att map & label of each instance
# all dataset : train, valid, test
# save ig attribute before rescaling.

baseline=torch.zeros(1,32,32).to('cuda')
train_ig=np.zeros((0,1,32,32))
train_scale_ig=np.zeros((0,32,32,1))
train_ig_label=np.zeros((0,1))


for i in tqdm(train_loader):
    for j in range(i[0].shape[0]):
        train_image=i[0][j]
        train_label=i[1][j]
        
        ig=integrated_gradients(baseline,train_image.to('cuda'),model=resnet,label=train_label.item())
        ig=ig.to('cpu').detach().numpy()
        img=train_image.to('cpu').numpy()
        vi3=scaled_attribute(ig)
        
        train_label=train_label.numpy().reshape(1,1)
        train_ig=np.concatenate([train_ig,ig.reshape(1,1,32,32)],axis=0) 
        train_scale_ig=np.concatenate([train_scale_ig,vi3.reshape(1,32,32,1)],axis=0)
        train_ig_label=np.concatenate([train_ig_label,train_label],axis=0)

 













