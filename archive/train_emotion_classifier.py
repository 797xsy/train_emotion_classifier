# 人脸表情识别
# Python + PyTorch + GPU
# 用卷积神经网络（CNN）训练一个表情分类器
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models # type: ignore
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------
# 1️⃣ 设置数据集路径（改成你自己的路径）
# ----------------------------------------------------------
train_dir = r"C:\Users\rog1\Desktop\人脸表情识别数据集\archive\train"
test_dir = r"C:\Users\rog1\Desktop\人脸表情识别数据集\archive\test"

# ----------------------------------------------------------
# 2️⃣ 定义图像预处理方式（transforms）
# transforms.Resize((48,48)) 表示把所有图片缩放到 48x48 像素
# transforms.ToTensor() 把图片转为 PyTorch 张量
# transforms.Normalize() 让模型更稳定
# ----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),   # 保留
    transforms.RandomRotation(10),       # 改成 ±10°
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------------------------------------
# 3️⃣ 使用 ImageFolder 自动加载数据
# 它会根据子文件夹名字自动生成标签
# ----------------------------------------------------------
train_dataset = datasets.ImageFolder(train_dir,transform=transform)
test_dataset = datasets.ImageFolder(test_dir,transform=transform)

# ----------------------------------------------------------
# 4️⃣ 创建数据加载器（DataLoader）
# batch_size = 64 表示每次训练读取 64 张图片
# shuffle=True 表示打乱训练数据顺序
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

print("类别列表",train_dataset.classes)
print("训练样本数量",len(train_dataset))
print("测试样本数量",len(test_dataset))

# ----------------------------------------------------------
# 5️⃣ 检查是否可以使用 GPU（CUDA）
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:",{device})

# ----------------------------------------------------------
# 6️⃣ 定义一个简单的 CNN 模型（类似 LeNet）
# nn.Sequential 用来快速搭建网络结构
# ----------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),  #展平成一维向量
            nn.Linear(64*12*12,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.network(x)
# ----------------------------------------------------------
# 7️⃣ 实例化模型
# len(train_dataset.classes) 会自动获取类别数量
# ----------------------------------------------------------
model = SimpleCNN(num_classes = len(train_dataset.classes)).to(device)

# ----------------------------------------------------------
# 8️⃣ 定义损失函数与优化器
# CrossEntropyLoss：多分类常用的损失函数
# Adam：一种常用的优化算法
# ----------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# ----------------------------------------------------------
# 9️⃣ 开始训练循环
# ----------------------------------------------------------
num_epochs = 45
train_losses = []  # 记录每轮训练的平均损失
train_accuracies = []  # 记录每轮训练的准确率

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

# ----------------------------------------------------------
# 🔟 绘制训练过程曲线
# ----------------------------------------------------------
plt.figure(figsize=(10,4))

# 子图1: 损失曲线
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# 子图2: 准确率曲线
plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Training Accuracy", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()
