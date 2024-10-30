# import os
# import shutil

# # 图片路径
# image_dir = './images'

# # 创建训练集和测试集文件夹
# train_dir = 'CUB_200_2011/train/'
# test_dir = 'CUB_200_2011/test/'

# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # 读取train_test_split.txt文件内容
# with open('train_test_split.txt', 'r') as file:
#     lines = file.readlines()

# # 获取所有文件夹及其图片路径并排序
# all_image_paths = []
# for folder in sorted(os.listdir(image_dir)):
#     folder_path = os.path.join(image_dir, folder)
#     print(folder_path)
#     if os.path.isdir(folder_path):
#         for file in sorted(os.listdir(folder_path)):
#             if file.endswith('.jpg'):
#                 all_image_paths.append(os.path.join(folder_path, file))

# num_train = 0
# num_test = 0

# # 解析文件内容并移动图片
# for i, line in enumerate(lines):
#     print(i)
#     label = int(line.strip().split()[1])
#     print(len(all_image_paths))
#     image_path = all_image_paths[i]
    
#     if label == 0:
#         shutil.move(image_path, os.path.join(train_dir, os.path.basename(image_path)))
#         num_train += 1
#     else:
#         shutil.move(image_path, os.path.join(test_dir, os.path.basename(image_path)))
#         num_test += 1

# print("图片已成功按照训练集和测试集划分, train: %d, test: %d" % (num_train, num_test))

import os
import shutil

# 图片路径
image_dir = './images'

# 创建训练集和测试集文件夹
train_dir = 'CUB_200_2011/train/'
test_dir = 'CUB_200_2011/test/'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 读取train_test_split.txt文件内容
with open('train_test_split.txt', 'r') as file:
    lines = file.readlines()

# 获取所有文件夹及其图片路径并排序
all_image_paths = []
for folder in sorted(os.listdir(image_dir)):
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):
        for file in sorted(os.listdir(folder_path)):
            if file.endswith('.jpg'):
                all_image_paths.append(os.path.join(folder_path, file))

num_train = 0
num_test = 0

# 解析文件内容并移动图片
for i, line in enumerate(lines):
    label = int(line.strip().split()[1])
    image_path = all_image_paths[i]
    
    # 获取原始文件夹路径
    relative_path = os.path.relpath(image_path, image_dir)
    if label == 0:
        destination_path = os.path.join(train_dir, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(image_path, destination_path)
        num_train += 1
    else:
        destination_path = os.path.join(test_dir, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(image_path, destination_path)
        num_test += 1

print("图片已成功按照训练集和测试集划分, train: %d, test: %d" % (num_train, num_test))
