import torch        
from torch import nn  
from torch.utils.data import DataLoader  
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
import cv2
from torchvision import models
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder


def imgprocess(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(100,100))
    frame = cv2.GaussianBlur(frame,(5,5),0)
  
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    
    img = cv2.dilate(skinMask,kernel)

    img = cv2.erode(img,kernel)

    edges = cv2.Canny(img, 30, 70)  # canny边缘检测 


    return edges  


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # image = Image.open(self.filenames[idx]).convert('RGB')
        image = imgprocess(self.filenames[idx])
        image = self.transform(image)
#         image = image[..., None]
#         image = np.tile(image, (1, 1, 3))
        return image, self.labels[idx]

    
transformer_ImageNet = transforms.Compose([
  #   transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def split_Train_Val_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)     # data_dir精确到分类目录的上一级 即 'Train_Data'
    character = [[] for i in range(len(dataset.classes))]
    #print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    #print(dataset.samples)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):   # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        #print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    #print(len(train_inputs))
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, transformer_ImageNet),
                                  batch_size=8, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, transformer_ImageNet),
                                  batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for iteration, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        pred = model(X)   #模型预测
        loss = loss_fn(pred, y)   #计算损失函数  # y进行了广播
        
        # 后向传播 （Backpropagation）：固定的3个步骤
        optimizer.zero_grad()   # 优化器置零
        loss.backward()        # 后向传播
        optimizer.step()       # 参数更新

        if iteration % 20 == 0:
            loss, current = loss.item(), iteration * len(X)
            print("loss: %.4f, current:%5d/size:%5d" %(loss, current, size))


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("Test Result\n  Accuracy: %.1f,  Average loss:%.8fd \n" %(100*correct, test_loss))



data_dir = './data/Dataset'
train_dataloader, val_dataloader = split_Train_Val_Data(data_dir, [0.8, 0.2])
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)
model = resnet50
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

loss_fn = nn.CrossEntropyLoss()   # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    #定义优化器
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 200
model = model.to(device)        # 十分重要，如果不加这一步，无法读取之前已经训练的模型
#model = cnn_model.to(device)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model)
print("Done!")


torch.save(model.state_dict(), "Gesture-digits.pth")
print("Saved PyTorch Model State to Gesture-digits.pth")