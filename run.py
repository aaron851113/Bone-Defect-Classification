import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import resnet as ResNet 
from  convnet import ConvNet as convnet
import argparse
import glob
import os
from PIL import Image

parser = argparse.ArgumentParser("sota")
parser.add_argument('--model', type=str, default='convnet', help='select cnn model')
parser.add_argument('--weight_path', type=str, default='./weights/', help='batch size')

args = parser.parse_args()

# create mkdir
try: 
    os.mkdir('./defect/run_Fine')
except: 
    print('-> [run_Fine]  folder is already exist')

try: 
    os.mkdir('./defect/run_Broke')
except: 
    print('-> [run_Broke] folder is already exist')

# setting GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader and transform datasets
testdir = './defect/run/'
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# select model 
"""
resnet18
convnet
"""
model_name = args.model

# Setting CNN model 
if model_name == 'resnet18':
    model = ResNet.resnet18(num_classes=2)
elif model_name == 'convnet':
    model = convnet()
assert model != None

# Loading model weight
weight_path = args.weight_path + model_name + '.ckpt'
print(weight_path)
model.load_state_dict(torch.load(weight_path))
model.requires_grad_(False)
model.to(device)

# Read images list 
img_list = sorted(glob.glob(testdir+'*.tif'))

# Test the model
model.eval()
for img_path in img_list:
    # img name
    img_name = img_path.split('/')[-1]
    
    # PIL 讀取圖檔
    img = Image.open(img_path, mode='r')
    img = img.convert('RGB')
    # resize & normalize 
    img = transform(img) # PIL->tensor
    img = normalize(img)
    # [3, 224 ,224] -> [1, 3, 224, 224]
    img = img.view(1,3,224,224)
    
    # put img into GPU
    img = img.to(device)
    
    # predict image label
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    if predicted.data == 0:
        label = 'Broke'
        new_img_path = "./defect/run_Broke/"+img_name
        
        
    elif predicted.data == 1:
        label = 'Fine'
        new_img_path = "./defect/run_Fine/"+img_name
    
    # replace img
    os.replace(img_path, new_img_path)
    
    # print out result 
    print(img_name, ':', label)
    