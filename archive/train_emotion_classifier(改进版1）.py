# 人脸表情识别
# 环境: Python + PyTorch + GPU
# 功能: 使用卷积神经网络（CNN）训练一个表情分类器
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models # type: ignore
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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


#定义一个残差块来提升准确率
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += x  # 残差连接：输出 + 输入
        return self.relu(out)


# ----------------------------------------------------------
# 6️⃣ 定义一个加了残差块的的 CNN 模型
# ----------------------------------------------------------
class DeepResCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepResCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24

            # Block 2 + 残差块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12

            # Block 3 + 残差块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# ----------------------------------------------------------
# 7️⃣ 实例化模型
# len(train_dataset.classes) 会自动获取类别数量
# ----------------------------------------------------------
model = DeepResCNN(num_classes=len(train_dataset.classes)).to(device)

# ----------------------------------------------------------
# 8️⃣ 定义损失函数与优化器
# CrossEntropyLoss：多分类常用的损失函数
# Adam：一种常用的优化算法
# ----------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0002)

# ----------------------------------------------------------
# 9️⃣ 开始训练循环
# ----------------------------------------------------------
num_epochs = 25
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
