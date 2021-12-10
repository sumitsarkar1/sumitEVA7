import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import argparse

import sumitEVA7
from sumitEVA7.models import resnet18_64

criterion = nn.CrossEntropyLoss()

def getModel():
    net = resnet.ResNet18()
    return net

def setOptimizer(net, lr):
    lr = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    return optimizer

def setScheduler(optimizer, epochs, max_lr, steps_per_epoch, pct_start, div_factor, final_div_factor):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,steps_per_epoch=steps_per_epoch,
            epochs=epochs, pct_start=pct_start, div_factor=div_factor,final_div_factor=final_div_factor ) 
    return scheduler


class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 512 
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 4, 'pin_memory': True} if self.use_cuda else {}
        
def getTrainTestLoader(args):
    #data_dir = '/home/sumit/eva7/assignment10/IMagenet/tiny-imagenet-200/'
    data_dir = '/home/rogbot/eva7_/assignment10/IMagenet/tiny-imagenet-200'
    num_workers = {'train' : 4,'val'   : 0,'test'  : 0}
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            ])
        }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=num_workers[x])
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    return dataloaders['train'], dataloaders['test']


from tqdm import tqdm

train_losses = []
train_losses_per_epoch = []
test_losses = []
train_acc = []
test_acc = []
train_acc_per_epoch = []
lrs = []


def train(net,optimizer, trainloader, device):
    grad_clip = 0.1
    net.train()
    #pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    #lrs = []
    
    train_loss_epoch = 0
    correct_epoch = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):    
        # get samples
        data, target = data.to(device), target.to(device)
        # Init
        optimizer.zero_grad()
        # Predict
        y_pred = net(data)
        # Calculate loss
        print("predicted = ", y_pred.shape, "labels = ",target.shape)
        loss = criterion(y_pred, target)
        train_losses.append(loss)
        #train_loss_epoch += loss.item()

        # Backpropagation
        loss.backward()        
        optimizer.step()
        
        # Update pbar-tqdm    
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
     
    train_loss_epoch /= len(trainloader.dataset)
    train_losses_per_epoch.append(train_loss_epoch)
    train_acc_per_epoch.append(100. * correct / len(trainloader.dataset))


       
#Testing
def test(net, testloader, device):
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
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    test_acc.append(100. * correct / len(testloader.dataset))


def getTrainLoss():
    return train_losses_per_epoch

def getTestLoss():
    return test_losses

def getTrainAcc():
    return train_acc_per_epoch

def getTestAcc():
    return test_acc

def getlrVals():
    return lrs
