import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import resnet as ResNet 
from  convnet import ConvNet as convnet
import argparse

parser = argparse.ArgumentParser("sota")
parser.add_argument('--model', type=str, default=None, help='select cnn model')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
args = parser.parse_args()

# select model 
"""
resnet18
convnet
"""
model_name = args.model

# setting GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# DataLoader and transform datasets
traindir = './defect/Train/'
testdir = './defect/Test/'
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
trainset = torchvision.datasets.ImageFolder( traindir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]))

testset = torchvision.datasets.ImageFolder( testdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8) #batch8代表一次只丟8張圖進去訓練
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


# Setting CNN model 
if model_name == 'resnet18':
    model = ResNet.resnet18(num_classes=2)
elif model_name == 'convnet':
    model = convnet()
assert model != None

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
best_acc = 0
best_epoch = 0
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        valid_acc = 100*correct / total
        if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = i+1

        print ("Epoch [{}/{}], Train Loss: {:.4f}, Valid Acc: {:.2f}, Best Acc: {:.2f}".format(epoch+1, num_epochs, loss.item(), valid_acc, best_acc))

# Save the model checkpoint
weight_path = './weights/'+model_name+'.ckpt'
torch.save(model.state_dict(),weight_path)
