import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import gradio  as gr


#load pretrained model &  finetune it on the target dataset
model=models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,15)

trans = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                       transforms.RandomHorizontalFlip(0.5),
                       transforms.RandomRotation(10)
                       ])

class_names=['c1','c2','c3','c4','c5','c6','c7','c8','c9']
model.load_state_dict(torch.load('best_model.pth',weights_only=True))
def predict(image):
    image=trans(image).unsqueeze(0)


        # Get model predictions
    with torch.no_grad():
        output = model(image)
         # Convert output to predicted class label
        _, predicted = torch.max(output, 1)
        predicted_class=predicted.item()
        # Return the class name
        return predicted_class 

# setup gradio interface

interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(type="pil"), 
                         outputs="text",
                         title="Ciphar dataset predictions",
                         description="Upload an Image to get its class prediction"
                         )

# Launch the Gradio interface
interface.launch(share=True)

