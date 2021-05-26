import cv2
from torchvision import models
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def imgprocess(img):
    # frame = cv2.imread(path)
    frame = cv2.resize(img,(100,100))
    frame = cv2.GaussianBlur(frame,(5,5),0)
    # downsize it to reduce processing time
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    
    img = cv2.dilate(skinMask,kernel)   # 膨胀操作

    img = cv2.erode(img,kernel)  # 腐蚀操作

    edges = cv2.Canny(img, 30, 70)  # canny边缘检测 

#    cv2.imshow('wsdwe',edges)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return edges  


transformer_ImageNet = transforms.Compose([
  #   transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def img_preprocess(img_path):
    img = imgprocess(img_path)
    img = transformer_ImageNet(img)
    return img   



classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


resnet50 = models.resnet50(pretrained=True)  
resnet50.fc = nn.Linear(2048, 10)
# resnet50.conv1 = nn.Conv2d(1, resnet50.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
model = resnet50
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


model.load_state_dict(torch.load("D:/DUT/code/Python/Gesture-digits/Gesture-digits.pth"))

 
model = model.to(device)
model.eval()
# x, y = test_data[0][0], test_data[0][1]    # 此时，x.shape= [1, 28, 28], y 表示标签
# x = x[None, ...].to(device)     # [1, 28, 28] -> [1, 1, 28, 28] 
# # print(x.shape)

# ## --------------------------------------------- test ---------------------
# for i in range(10):
    
#     img_path = 'D:/DUT/code/Python/Gesture-digits/Examples/example_' + str(i) + '.JPG'
#     image = img_preprocess(img_path)
#     # print(type(image))
#     # print(image.shape)
#     # print(image.shape)
#     # plt.imshow(image)
#     image = image[None, ...].to(device)
#     # dog = dog[None, ...].to(device)

#     # print(cat.shape)

#     with torch.no_grad():
#         pred = model(image)
#         predicted, actual = classes[pred[0].argmax(0)], str(i)
#         print("Predicted: %s, Actual: %s" %(predicted, actual))
#         plt.figure()
#         plt.xlabel(f'Predicted: {predicted}')
#         plt.imshow(cv2.imread(img_path))
#         plt.show()
    
        # print(type(x[0][0].cpu()))
    # with torch.no_grad():
    #     pred = model(cat)
    #     predicted, actual = classes[pred[0].argmax(0)], 'cat'
    #     print("Predicted: %s, Actual: %s" %(predicted, actual))
    #     plt.figure()
    #     plt.imshow(cat[0].permute(1, 2, 0).cpu())
    #     # print(type(x[0][0].cpu()))

# --------------------------------------------- test -----------------------


cv2.namedWindow("camera",1)

# 开启ip摄像头
# video= "http://admin:admin@10.5.14.249:8081" #此处@后的ipv4 地址需要改为app提供的地址

# 开启电脑内置摄像头
video = 0

cap =cv2.VideoCapture(video)

while True:

    # Start Camera, while true, camera will run

    ret, image_np = cap.read()

    # Set height and width of webcam
    height = 600
    width = 1000
     
    # Set camera resolution and create a break function by pressing 'q'
    # cv2.imshow('object detection', cv2.resize(image_np, (width, height)))
    cv2.imshow('phone', image_np)
    # image = cv2.resize(image_np, (100, 100))
    
    image = img_preprocess(image_np)
    # print(type(image))
    # print(image.shape)
    # print(image.shape)
    # plt.imshow(image)
    image = image[None, ...].to(device)
    # dog = dog[None, ...].to(device)

    # print(cat.shape)

    with torch.no_grad():
        pred = model(image)
        predicted  = classes[pred[0].argmax(0)] 
        print("Predicted: %s" %predicted)
        # plt.figure()
        # plt.xlabel(f'Predicted: {predicted}')
        # plt.imshow(cv2.imread(img_path))
        # plt.show()



    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# Clean up

cap.release()
cv2.destroyAllWindows()

