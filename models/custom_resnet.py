import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
      
        # RESIDUAL BLOCK 1
        self.residualblock1 = nn.Sequential(
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 
        
        # RESIDUAL BLOCK 2
        self.residualblock2 = nn.Sequential(
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) 
        
        # PREP LAYER
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 
        
        # LAYER 1        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 
        
        # LAYER 2        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),padding=0,bias=False),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        
        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) 
        
        # MAX POOL WITH KERNEL=4
        self.maxpool2d = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, padding=1),
        ) 

        # FULLY CONNECTED LAYER
        self.FC = nn.Sequential(
            nn.Linear(512, 10,bias=False),
        )
        

    def forward(self, x):
        
        x = self.preplayer(x)
        
        x = self.layer1(x)   
        
        identity1 = x
        x = self.residualblock1(x)
        x = x + identity1
        
        x = self.layer2(x)
        
        x = self.layer3(x)  
        
        identity2 = x
        x = self.residualblock2(x)
        x = x + identity2 
        
        x = self.maxpool2d(x)
        
        x = torch.flatten(x, 1)
        
        x = self.FC(x)

        return F.log_softmax(x, dim=1)
