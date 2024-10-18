import os

import pandas as pd
# 将数据集分割成数据集和训练集
from sklearn.model_selection import train_test_split
import shutil

import os
HOME = os.getcwd()
print(HOME)

# 数据预处理
train_img = 'E:/repository/yolov5/yolov5-6.0/car_data/training_images'
test_img = 'E:/repository/yolov5/yolov5-6.0/car_data/test_images'
train_csv = 'E:/repository/yolov5/yolov5-6.0/car_data/train_solution_bounding_boxes (1).csv'
test_csv = 'E:/repository/yolov5/yolov5-6.0/car_data/sample_solution_bounding_boxes.csv'

# 创建名为images_dir的目录
images_dir = 'E:/repository/yolov5/yolov5-6.0/images'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir,'val'),exist_ok=True)

labels_dir = 'E:/repository/yolov5/yolov5-6.0/labels'
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(os.path.join(labels_dir,'train'),exist_ok=True)
os.makedirs(os.path.join(labels_dir,'val'),exist_ok=True)

# 将边界框的坐标从原始图像尺寸转换为YOLO格式
def convert_to_yolo(row,image_width,image_height):
    class_id = 0
    x_center = (row['xmin'] + row['xmax']) / 2 / image_width
    y_center = (row['ymin'] + row['ymax']) / 2 / image_height
    width = (row['xmax'] - row['xmin']) / image_width
    height = (row['ymax'] - row['ymin']) / image_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

df = pd.read_csv(train_csv)
df.columns = ['image','xmin','ymin','xmax','ymax']
# 使用dropna函数删除任何在'image'列中有缺失值的行
#使用fillna函数的ffill方法填充剩余的缺失值。ffill代表“向前填充”，意味着用前一个非空值来填充当前的空值。
df = df.dropna(subset=['image']).fillna(method='ffill')

train_images,val_images = train_test_split(df['image'].unique(),test_size=0.2,random_state=42)

import cv2
for image_name in df['image'].unique():
    image_path = os.path.join(train_img, image_name)

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or could not be opened:{image_path}")
        images_heigth, images_width,_ = img.shape
    except FileNotFoundError as e:
        print(f"Error provessing {image_name}:{e}")
        continue
    group = df[df['image'] == image_name]

    label_subdir = 'train' if image_name in train_images else 'val'
    image_subdir = label_subdir
    # 构建标签文件导入
    label_path = os.path.join(labels_dir,label_subdir,os.path.splitext(image_name)[0] + '.txt')
    # 转换成yolo并写入标签文件
    with open(label_path,'w') as f:
        for _,row in group.iterrows():
            yolo_line = convert_to_yolo(row,images_width,images_heigth)
            f.write(yolo_line + '\n')
    # 将原始图像复制到相应的训练集或验证集中
    shutil.copy(image_path,os.path.join(images_dir,image_subdir,image_name))

data_yaml = 'E:/repository/yolov5/yolov5-6.0/data.yaml'
with open(data_yaml,'w') as f:
    f.write(f"""
train: E:/repository/yolov5/yolov5-6.0/images/train
val: E:/repository/yolov5/yolov5-6.0/images/val
test: {test_img}
nc: 1
names: ['car']
""")

import matplotlib.pyplot as plt
import os
from PIL import Image
import random

# 设置你的图片和目录

result_dir = 'E:/repository/yolov5/yolov5-6.0/runs/detect/exp9'
all_image_files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg','.jpeg','.png'))]

# 从所有图片中随机选择5张
random_image = random.sample(all_image_files,5)
image_paths = [os.path.join(result_dir,img) for img in random_image]


# 遍历图片路径列表，并显示图片
for path in image_paths:
    img = Image.open(path)
    plt.figure(figsize=(10,10))

    plt.imshow(img)
    plt.axis('off')
    plt.show()