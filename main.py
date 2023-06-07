# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
# 加载 ResNet 模型
model = models.resnet18(pretrained=True)
model.eval()

PATH = 'data\cifar_net.pth'
# 设置类别名称映射表
# 定义 CIFAR-10 中各个类别的名称
class_names = ['飞机', '汽车', '鸟', '猫', '鹿',
               '狗', '青蛙', '马', '船', '卡车']

# name2label = {label_name: i for i, label_name in enumerate(label_names)}  # 将类别名称映射为类别索引
net = models.resnet18(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
device = torch.device("cpu")
net.load_state_dict(torch.load(PATH))
net.eval()
net.to(device)

from PIL import Image
import torchvision.transforms.functional as TF



# # 定义用于处理图像的转换器
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

col1, col2, col3 = st.columns(3)

uploaded_files = []

# 显示三个 file uploader，并将其加载到 uploaded_files 列表中
with col1:
    uploaded_file1 = st.file_uploader('Choose an image file 1', type=['jpg', 'jpeg', 'png'])
    if uploaded_file1 is not None:
        uploaded_files.append(uploaded_file1)

with col2:
    uploaded_file2 = st.file_uploader('Choose an image file 2', type=['jpg', 'jpeg', 'png'])
    if uploaded_file2 is not None:
        uploaded_files.append(uploaded_file2)

with col3:
    uploaded_file3 = st.file_uploader('Choose an image file 3', type=['jpg', 'jpeg', 'png'])
    if uploaded_file3 is not None:
        uploaded_files.append(uploaded_file3)

# 对每个已上传的文件进行分类并显示结果
for uploaded_file in uploaded_files:
    # 将上传的图像转换为 PyTorch 所需的张量形式
    img = Image.open(uploaded_file).convert('RGB')
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = img_tensor.unsqueeze(0)  # 增加一个维度作为批次数

    # 在 GPU 上进行推理，获取预测结果
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = net(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # 显示预测结果
    st.write('上传的图像：')
    st.image(img, use_column_width=True, caption=' ')

    st.write('预测结果：')
    st.write(class_names[predicted[0]])



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
