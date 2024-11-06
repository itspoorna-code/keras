import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10,ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision import models


trans = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

dataset  =  ImageFolder(root=r"C:\Users\student\Downloads\archive",transform=trans)

# split a dataset into train & validation set
train_ratio = 0.8
tran_size=int(train_ratio*len(dataset))
test_size=len(dataset)-tran_size
train_data,test_data= random_split(dataset,[tran_size,test_size])

print(train_data[0][0][0].shape)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False)

class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel,self).__init__()
        self.conv1 =  nn.Conv2d(3,32,3,stride=2)
        self.conv2  = nn.Conv2d(32,64,3)
        self.conv3  = nn.Conv2d(64,128,3,stride=2)

        self.fc1  = nn.Linear(128*7*7,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 15)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x =  self.pool(F.relu(self.conv1(x)))
        x =  self.pool(F.relu(self.conv2(x)))
        x =  self.pool(F.relu(self.conv3(x)))
        x =  x.view(-1,128*7*7)
        x =  torch.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x =  self.fc3(x)
        return x

model = cnnmodel()
criterian =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)


# Train the model 
num_epoch =  10
for epoch in range(num_epoch):

    runing_loss = 0.0

    for image, label in train_loader:

        output =  model(image)
        loss  =  criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        runing_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epoch}] , loss:{runing_loss/len(train_loader)}')

model.eval()
total = 0.0
correct = 0.0
with torch.no_grad():
    for image,label in test_loader:
        output =  model(image)
        _,predicted =  torch.max(output, 1)
        total  += label.size(0)
        runing_loss+=loss.item()
        correct += (predicted ==  label).sum().item()
    print(f'Accuracy: {100*(correct/total)}')

# plt.plot(loss)
# plt.show()
