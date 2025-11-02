# 😊 人脸表情识别（Face Emotion Recognition）

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-谢诗怡-pink.svg)](#-作者信息)

---

作者：谢诗怡 | 东北林业大学  
这是一个使用 PyTorch实现的人脸表情识别项目，能够自动识别人类面部的多种表情（如高兴、愤怒、惊讶、悲伤等）。  
该项目为深度学习入门实践作品，包含完整的数据集、模型训练代码与结果展示。

---

## 📁 项目结构

train_emotion_classifier/
│
├── # 人脸表情识别 （独立完成版）.py # 主程序代码
├── Figure_1（独立完成版）.png # 模型训练结果图
├── 人脸表情识别数据集.zip # 数据集（含人脸表情图片）
└── README.md # 项目说明文件


## 🚀 环境要求

运行本项目需要以下依赖环境：

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- torchvision  
- matplotlib  
- numpy  
- OpenCV（cv2）

安装依赖：
```bash
pip install torch torchvision matplotlib numpy opencv-python
🧠 项目简介
本项目基于 卷积神经网络（CNN） 架构，实现了人脸表情的自动识别。
模型结构参考 LeNet-5，适合深度学习初学者的实验与展示。

主要特点：

使用 CNN 自动提取面部特征

支持 GPU 加速训练

自动绘制训练准确率与损失曲线

可扩展至更多表情类别

🏃‍♀️ 运行步骤
下载或克隆本仓库：

bash
复制代码
git clone https://github.com/797xsy/train_emotion_classifier.git
解压 人脸表情识别数据集.zip；

运行主程序：

bash
复制代码
python "# 人脸表情识别 （独立完成版）.py"
程序会自动开始训练并绘制结果图。


✨ 项目亮点
💡 自建数据集：人脸表情图片按类别分布

⚙️ 模型轻量化：运行速度快、效果稳定

🎓 学术用途：可作为深度学习入门实验作品

📬 作者信息
👩‍💻 作者：谢诗怡
🏫 学校：东北林业大学
📧 邮箱：769867455@qq.com


🌱 本项目作为个人学习实践项目，仅供教学与研究使用。

—— END ——
