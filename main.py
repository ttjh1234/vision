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
resnet=ResNet18(3)
resnet.load_state_dict(torch.load('./assets/model/resnet18-1.pt'))
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
resnet.load_state_dict(torch.load('./assets/model/mnist_resnet18-0.pt'))
resnet.to('cuda')
sm=nn.Softmax(dim=1)

mnist_mean=np.array(0.3081) 
mnist_std=np.array(0.1307)

baseline=torch.zeros(1,28,28).to('cuda')

for i in test_loader:
    break


test_image=i[0][2]
test_label=i[1][2]

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

baseline=torch.zeros(1,28,28).to('cuda')
train_ig=np.zeros((0,1,28,28))
train_scale_ig=np.zeros((0,28,28,1))
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
        train_ig=np.concatenate([train_ig,ig.reshape(1,1,28,28)],axis=0) 
        train_scale_ig=np.concatenate([train_scale_ig,vi3.reshape(1,28,28,1)],axis=0)
        train_ig_label=np.concatenate([train_ig_label,train_label],axis=0)

baseline=torch.zeros(1,28,28).to('cuda')
valid_ig=np.zeros((0,1,28,28))
valid_scale_ig=np.zeros((0,28,28,1))
valid_ig_label=np.zeros((0,1))


for i in tqdm(valid_loader):
    for j in range(i[0].shape[0]):
        valid_image=i[0][j]
        valid_label=i[1][j]
        
        ig=integrated_gradients(baseline,valid_image.to('cuda'),model=resnet,label=valid_label.item())
        ig=ig.to('cpu').detach().numpy()
        img=valid_image.to('cpu').numpy()
        vi3=scaled_attribute(ig)
        
        valid_label=valid_label.numpy().reshape(1,1)
        valid_ig=np.concatenate([valid_ig,ig.reshape(1,1,28,28)],axis=0) 
        valid_scale_ig=np.concatenate([valid_scale_ig,vi3.reshape(1,28,28,1)],axis=0)
        valid_ig_label=np.concatenate([valid_ig_label,valid_label],axis=0)

baseline=torch.zeros(1,28,28).to('cuda')
test_ig=np.zeros((0,1,28,28))
test_scale_ig=np.zeros((0,28,28,1))
test_ig_label=np.zeros((0,1))


for i in tqdm(test_loader):
    for j in range(i[0].shape[0]):
        test_image=i[0][j]
        test_label=i[1][j]
        
        ig=integrated_gradients(baseline,test_image.to('cuda'),model=resnet,label=test_label.item())
        ig=ig.to('cpu').detach().numpy()
        img=test_image.to('cpu').numpy()
        vi3=scaled_attribute(ig)
        
        test_label=test_label.numpy().reshape(1,1)
        test_ig=np.concatenate([test_ig,ig.reshape(1,1,28,28)],axis=0) 
        test_scale_ig=np.concatenate([test_scale_ig,vi3.reshape(1,28,28,1)],axis=0)
        test_ig_label=np.concatenate([test_ig_label,test_label],axis=0)



ig_train=np.load('./assets/ig/ig_train.npy')
ig_valid=np.load('./assets/ig/ig_valid.npy')

ig_train2=torch.Tensor(ig_train)
ig_valid2=torch.Tensor(ig_valid)

train_loader=DataLoader(ig_train2,batch_size=128,shuffle=True)
valid_loader=DataLoader(ig_valid2,batch_size=128,shuffle=True)

ig_train2.shape
ig_valid2.shape


for i in train_loader:
    break

enc=sae_encoder()
dec=sae_decoder()
sae=stacked_autoencoder(enc,dec)
device='cuda'
sae.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(sae.parameters(), lr=0.01,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

N_EPOCHS = 500
CLIP = 1
best_valid_loss = float('inf')
patient=0
run_flag=1
path='./assets/neptune/neptune_args.txt'
neptune_key=fetch_neptune_key(path)

run = neptune.init(
    project=neptune_key['project'],
    api_token=neptune_key['api_token'],
)
experiment=4

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = sae_train(sae, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
    valid_loss = sae_evaluate(sae, valid_loader, criterion, device='cuda',run_flag=run_flag)
    
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
        torch.save(sae.state_dict(), './assets/model/sae-{}.pt'.format(experiment))
    else:
        patient+=1
        if patient>=200:
            break

run.stop()

sae.load_state_dict(torch.load('./assets/model/sae-{}.pt'.format(experiment)))

for i in train_loader:
    break


origin_img=i[:5]
origin_img.shape
pred=sae(origin_img.unsqueeze(1).to(device))

origin_img[0].shape

plt.imshow(origin_img[0].to('cpu').numpy().transpose([1,2,0]))
plt.imshow(pred[[0]].to('cpu').detach().numpy().transpose([1,2,0]))

ig_train=np.load('./assets/ig/ig_train.npy')
ig_valid=np.load('./assets/ig/ig_valid.npy')

ig_scale_train=np.load('./assets/ig/ig_scale_train.npy')
ig_scale_valid=np.load('./assets/ig/ig_scale_valid.npy')

ig_train2=ig_train.reshape(50000,-1)
ig_valid2=ig_valid.reshape(10000,-1)

ig_scale_train2=ig_scale_train.reshape(50000,-1)
ig_scale_valid2=ig_scale_valid.reshape(10000,-1)

ig_label_train=np.load('./assets/ig/ig_label_train.npy')
    
ig_model=ig_featuremap(10)
ig_model.to('cuda')

train_loader,valid_loader,test_loader=mnist_ig_data_loader()

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




##### Experiment #####
'''

Ig data fitting
Very Simple Model, But Performace is good.
Train Accuracy :  tensor(1.0000, device='cuda:0')
Valid Accuracy :  tensor(0.9958, device='cuda:0')
Test Accuracy :  tensor(0.9973, device='cuda:0')

'''


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ig_model.parameters(), lr=0.01,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

N_EPOCHS = 100
CLIP = 1
best_valid_loss = float('inf')
patient=0
run_flag=1
path='./assets/neptune/neptune_args.txt'
neptune_key=fetch_neptune_key(path)

run = neptune.init(
    project=neptune_key['project'],
    api_token=neptune_key['api_token'],
)
experiment=1

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train(ig_model, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
    valid_loss = evaluate(ig_model, valid_loader, criterion, device='cuda',run_flag=run_flag)
    
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
        torch.save(ig_model.state_dict(), './assets/model/igmodel-{}.pt'.format(experiment))
    else:
        patient+=1
        if patient>=10:
            break

run.stop()
ig_model.load_state_dict(torch.load('./assets/model/igmodel-{}.pt'.format(experiment)))
test_loss = evaluate(ig_model, test_loader, criterion,'cuda')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score(ig_model,train_loader,valid_loader,test_loader,device='cuda')


##### Mnist Simple Model Performance Check #####

'''
Train Accuracy :  tensor(0.9976, device='cuda:0')
Valid Accuracy :  tensor(0.9936, device='cuda:0')
Test Accuracy :  tensor(0.9947, device='cuda:0')
'''

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ig_model.parameters(), lr=0.01,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

N_EPOCHS = 100
CLIP = 1
best_valid_loss = float('inf')
patient=0
run_flag=1
path='./assets/neptune/neptune_args.txt'
neptune_key=fetch_neptune_key(path)

run = neptune.init(
    project=neptune_key['project'],
    api_token=neptune_key['api_token'],
)
experiment=1

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train(ig_model, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
    valid_loss = evaluate(ig_model, valid_loader, criterion, device='cuda',run_flag=run_flag)
    
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
        torch.save(ig_model.state_dict(), './assets/model/mnist_simple-{}.pt'.format(experiment))
    else:
        patient+=1
        if patient>=10:
            break

run.stop()
ig_model.load_state_dict(torch.load('./assets/model/mnist_simple-{}.pt'.format(experiment)))
test_loss = evaluate(ig_model, test_loader, criterion,'cuda')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score(ig_model,train_loader,valid_loader,test_loader,device='cuda')


##### Experiment #####

'''
origin_image -> simple model -> + => classifier
            |_  IG_model      _|
'''

ig_model=ig_featuremap(10)
ig_model.load_state_dict(torch.load('./assets/model/igmodel-1.pt'))

for para in ig_model.parameters():
    para.requires_grad = False

feat_ext=nn.Sequential(ig_model.conv1,nn.MaxPool2d(2),ig_model.conv2,nn.MaxPool2d(2),ig_model.conv3,nn.MaxPool2d(2))

test_model=ig_featureplus(10,feat_ext)

for i in train_loader:
    break

r=feat_ext(i[0].to('cuda'))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


N_EPOCHS = 200
CLIP = 1
best_valid_loss = float('inf')
patient=0
run_flag=1
path='./assets/neptune/neptune_args.txt'
neptune_key=fetch_neptune_key(path)

run = neptune.init(
    project=neptune_key['project'],
    api_token=neptune_key['api_token'],
)
experiment=3

test_model.to('cuda')

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train(test_model, train_loader, optimizer, criterion, device='cuda',run_flag=run_flag)
    valid_loss = evaluate(test_model, valid_loader, criterion, device='cuda',run_flag=run_flag)
    
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
        torch.save(test_model.state_dict(), './assets/model/experiment_plus-{}.pt'.format(experiment))
    else:
        patient+=1
        if patient>=10:
            break

run.stop()
test_model.load_state_dict(torch.load('./assets/model/experiment_plus-{}.pt'.format(experiment)))
test_loss = evaluate(test_model, test_loader, criterion,'cuda')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score(test_model,train_loader,valid_loader,test_loader,device='cuda')


##### Experiment #####
# check each layer activate map.

plt.figure(figsize=(12,12))
for i, filter in enumerate(ig_model.conv1.weight.detach().to('cpu').numpy()):
    plt.subplot(4,4,i+1)
    plt.imshow(filter[0,:,:],cmap='gray')
    plt.axis('off')
plt.show()

ig_model.conv2.weight.detach().to('cpu').numpy().shape

plt.figure(figsize=(12,12))
for i, filter in enumerate(ig_model.conv2.weight.detach().to('cpu').numpy()):
    plt.subplot(8,8,i+1)
    plt.imshow(filter[3,:,:],cmap='gray')
    plt.axis('off')
plt.show()

train_loader2,valid_loader2,test_loader2=mnist_ig_data_loader()

for i in train_loader2:
    break

test_image=i[0][0]

results=[ig_model.conv1(test_image.unsqueeze(0))]
results.append(ig_model.conv2(F.max_pool2d(results[-1],2)))
results.append(ig_model.conv3(F.max_pool2d(results[-1],2)))

outputs=results
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    #print(f"Saving layer {num_layer} feature maps...")
    #plt.savefig(f"../outputs/layer_{num_layer}.png")
    plt.show()
    plt.close()


## Exper###

# (Channel)Depthwise Pooling -> Input B,128,3,3 -> B,1,3,3     ---- 1
# Featurewise Max Pooling -> Input B,128,3,3     ---- 2
# We want weight (prob) B,128,3,3,  

origin=i[0][0]
origin=origin.numpy()
mnist_mean=np.array(0.3081) 
mnist_std=np.array(0.1307)
origin2=rescale_data(origin,mnist_mean,mnist_std)

plt.imshow(origin2)




