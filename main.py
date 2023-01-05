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


train_loader, valid_loader , test_loader, classes = data_loader_cifar10('./assets/data/cifar10')
resnet=ResNet18()
resnet.load_state_dict(torch.load('./assets/model/resnet18.pt'))
resnet.to('cuda')

baseline=torch.zeros(3,32,32).to('cuda')

for i in train_loader:
    break

test_image=i[0][12]
test_label=i[1][12]

ig=integrated_gradients(baseline,test_image.to('cuda'),model=resnet,label=test_label.item())
ig=ig.to('cpu').detach().numpy()

test_label
classes[test_label]

vi3=scaled_attribute(ig)


img=test_image.to('cpu').numpy()

img2=rescale_data(img)

resnet(test_image.unsqueeze(0).to('cuda'))

plot_attribute_map(img2,vi3)
plot_image_overlay(img2,vi3)





