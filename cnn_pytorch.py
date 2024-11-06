import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

trans=transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



train_data=CIFAR10(root='./cifar',train=True,download=True,transform=trans)
test_data=CIFAR10(root='./cifar',train=False,download=True,transform=trans)

train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(train_data,batch_size=32,shuffle=False)

class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel,self).__init__()  

         # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)  # Input channels = 3 (RGB), output channels = 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)  # Output channels = 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,padding=1) # Output channels = 128
        self.fc1 = nn.Linear(128 * 4 * 4,256)  # Fully connected layer (128 channels, 4x4 feature map)
        self.fc2 = nn.Linear(256,128)  # Output layer (10 classes for CIFAR-10)
        self.fc3=nn.Linear(128,10)
        self.pool = nn.MaxPool2d(2, 2)  # Max Pooling layer with 2x2 kernel

    def forward(self,x):
       x=self.conv1(x)
       x=F.relu(x)
       x=self.pool(x)
       x=self.pool(F.relu(self.conv2(x)))
       x=self.pool(F.relu(self.conv3(x)))
       x=x.view(-1,128*4*4)
       x=torch.relu(self.fc1(x))
       x=torch.relu(self.fc2(x))
       x=self.fc3(x)

       return x


model=cnnmodel()
criterian = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Training the model
num_epochs=10
for epoch in range(num_epochs):
  
    running_loss = 0.0
    
    for image,label in train_loader:
        output=model(image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}] , loss:{running_loss/len(train_loader)}')   
    
model.eval()
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients during testing
    for image, label in test_loader:
        output = model(image)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        running_loss+=loss.item()
        correct += (predicted == label).sum().item()
    # print(f'Epoch [{epoch+1}/{num_epochs}] , loss:{running_loss/len(train_loader)}')    
    print(f"Test Accuracy: {100 * (correct / total)}%")

# plt.plot(loss)
# plt.show()