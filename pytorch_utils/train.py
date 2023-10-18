import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import sys 
from dataset import ImageDataset
from torch.utils.data import DataLoader

path = os.path.abspath("..")
sys.path.append(path)

from utils import create_mean_and_std_for_images


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG16(nn.Module):
    def __init__(self, num_classes = 5):
        super(VGG16, self).__init__()  
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU())                               
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 64, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer_4 = nn.Sequential(
            nn.Conv2d(128, 128, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_5 = nn.Sequential(
            nn.Conv2d(128, 256, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer_6 = nn.Sequential(
            nn.Conv2d(256, 256, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer_7 = nn.Sequential(
            nn.Conv2d(256,256, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_8 = nn.Sequential(
            nn.Conv2d(256, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer_9 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer_10 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_11 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer_12 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer_13 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1,kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(4096,num_classes),
            nn.Softmax(dim=1))
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.layer_10(out)
        out = self.layer_11(out)
        out = self.layer_12(out)
        out = self.layer_13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)        
        out = self.fc2(out)   
        out = self.fc3(out)     
        return out    
   

model = VGG16()
print(model)
        
#load the dataset, loss and optimizer 

#normalized values found in the function

mean, std = create_mean_and_std_for_images('../fruits.csv')
normalize = transforms.Normalize(mean=mean,
                                 std=std)

transform_method = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_dataset = ImageDataset('../fruits_train.csv', "./dataset", transform=transform_method)

valid_dataset = ImageDataset("../fruits_valid.csv", "./dataset", transform=transform_method)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)   

valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

num_classes = 5
num_epochs = 30
batch_size = 10
learning_rate = 0.0001

model = VGG16(num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# training on the neural network


total_step = len(train_loader)

for epoch in range(num_epochs):
    for i , (images, labels) in enumerate(train_loader):
        #move on the gpu
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        output = model(images)
        loss = criterion(output, labels)

        #Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total)) 

        
torch.save(model.state_dict(), 'model_v4.pth')
        
        
        
        
        
        
        

                