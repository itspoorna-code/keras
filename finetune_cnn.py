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
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                       transforms.RandomHorizontalFlip(0.5),
                       transforms.RandomRotation(10)
                       ])

dataset  =  ImageFolder(root=r"C:\Users\student\Downloads\archive\PlantVillage",transform=trans)
train_data,val_data=random_split(dataset,[0.8,0.2])


train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(val_data,batch_size=32,shuffle=False)

# Load the pre-trained ResNet18 model
model=models.resnet18(pretrained=True)


for param in model.parameters():
    param.requires_grad = False  # Freeze all layers


in_features=model.fc.in_features
model.fc=nn.Linear(512,15)

# Unfreeze the last fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# Train the model 
num_epoch =  10
for epoch in range(num_epoch):

    runing_loss = 0.0

    for image, label in train_loader:

        output =  model(image)
        loss  =  criterion(output,label)
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


if accuracy>best_val_accuracy:
    best_val_accuracy=accuracy
    torch.save(model.state_dict(),'best_model.pth')
