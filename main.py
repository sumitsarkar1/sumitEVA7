'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import resnet
from utils import progress_bar

epochs = None
batch_size = 128

criterion = nn.CrossEntropyLoss()
optimizer = None
scheduler = None
testloader = None
trainloader = None

def getModel():
    net = resnet.ResNet18()
    return net

def setNumberofepochs(epochs_):
    epochs = epochs_
    print("Number of epochs set to = ", epochs)
    return epochs

def setBatchSize(batch_size_):
    batch_size = batch_size_
    print("Batch Size set to = ", batch_size)
    return batch_size

def getBatchSize():
    print(batch_size)
    return batch_size

def setOptimizer(net):
    optimizer_ = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
    optimizer = optimizer_
    return optimizer

def setScheduler(optimizer_):
    optimizer = optimizer_
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return scheduler


import utils
from utils import Cifar10Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = utils.getTrainTransforms()
test_transforms = utils.getTestTransforms()

class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

trainset = Cifar10Dataset(root='../data', train=True,download=True, transform=train_transforms) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,shuffle=True, **args().kwargs)


testset = Cifar10Dataset(root='../data', train=False,download=True, transform=test_transforms) 
testloader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,shuffle=True, **args().kwargs)


from tqdm import tqdm

train_losses = []
test_losses = []

def train(net,epoch, optimizer):
  net.train()
  pbar = tqdm(trainloader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):    
    # get samples
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()
    # Predict
    y_pred = net(data)
    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Update pbar-tqdm    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        
#Testing
def test(net, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total)) 



def getTrainLoss():
    train_losses_cpu = []
    for loss in train_losses:
        train_losses_cpu.append(loss.cpu().data.numpy())  
    
    return train_losses_cpu

def getTestLoss():
    return test_losses






