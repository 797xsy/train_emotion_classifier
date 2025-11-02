# 人脸表情识别 
# 工具 pytorch + GPU + python
# 用卷积神经网络(CNN)实现一个表情分类

#导入必要的库
import os  #用于操作系统的交互
import torch  #pytorch主包，包含张量和计算图功能
import torch.nn as nn  #神经网络模块，包含常用的层
import torch.optim as optim  #优化器模块，包含SGD、Adam等优化算法
from torchvision import datasets,transforms,models  #torchvision包含数据集、图像变换和预训练模型等
from torch.utils.data import DataLoader   #Dataloader用于吧Dataset变成可迭代的数据加载器
import matplotlib.pyplot as plt  #用于绘图
from PIL import Image  #用于图像读取

# 1. 设置数据集的路径
train_dir = r"C:\Users\rog1\Desktop\人脸表情识别数据集\archive\train"
test_dir = r"C:\Users\rog1\Desktop\人脸表情识别数据集\archive\train"

# 2. 定义图像预处理的方式
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])


# 3. 使用ImageFolder 自动加载数据
train_dataset = datasets.ImageFolder(train_dir,transform=transform)
test_dataset = datasets.ImageFolder(test_dir,transform=transform)


# 4. 创建数据加载器
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

# 5. 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：",{device})


# 6. 定义一个简单的CNN模型
class SimperCNN(nn.Module):
    def __init__(self,nums_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 , 32 , kernel_size=3 , stride=1 , padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d( 32 , 64 , kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(64*12*12,128),
            nn.ReLU(),
            nn.Linear(128,nums_classes)
        )
    
    def forward(self,x):
        return self.network(x)
# 7. 实例化模型
model = SimperCNN(nums_classes = len(train_dataset.classes)).to(device)

# 8. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr =0.001)  
 
# 9. 开始循环训练
num_epochs = 45
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0 
    total = 0 
    for images , labels in train_loader:
        images , labels = images.to(device) , labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output , labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predict = torch.max(output,1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Tain_acc: {train_acc:.2f}%")

# 10. 绘制训练过程曲线
plt.figure(figsize=(10,4))                       # 创建一个图形窗口，设置大小

# 子图1: 损失曲线
plt.subplot(1,2,1)                               # 在一行两列的子图中选择第 1 个位置
plt.plot(train_losses, label="Training Loss")    # 绘制损失曲线（X 轴为 epoch 索引，Y 轴为损失）
plt.xlabel("Epoch")                             # X 轴标签
plt.ylabel("Loss")                              # Y 轴标签
plt.title("Training Loss Curve")                # 子图标题
plt.legend()                                      # 显示图例

# 子图2: 准确率曲线
plt.subplot(1,2,2)                               # 选择第 2 个子图位置
plt.plot(train_accuracies, label="Training Accuracy")  # 绘制准确率曲线
plt.xlabel("Epoch")                             # X 轴标签
plt.ylabel("Accuracy (%)")                      # Y 轴标签
plt.title("Training Accuracy Curve")            # 子图标题
plt.legend()                                      # 显示图例

plt.tight_layout()                                # 自动调整子图间距，防止重叠
plt.show()                                        # 显示图形