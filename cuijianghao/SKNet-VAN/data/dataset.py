import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import imageio  # 引入imageio包
import pickle as pk


# 得到某个文件夹中文件的全部绝对地址
def get_all_path(path):
    path_list = []
    for fpath, dirs, fs in os.walk(path):
        for file in fs:
            name, category = os.path.splitext(file)
            if category == '.png':
                path_list.append(file)
    return path_list


class MyData(Dataset):
    def __init__(self, train, path='C:/Users/cuiji/Desktop/Deep_Learning/SKNet/data/'):  # 修改为CIFAR-100的路径
        if train:
            self.path = path + 'train/'
        else:
            self.path = path + 'test/'

        self.datas = []
        self.labels = []
        datalist = get_all_path(self.path)
        for i in datalist:
            img_path = self.path + i
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image not loaded: {img_path}")
            else:
                img = img.transpose((2, 0, 1))
                self.datas.append(img.astype(np.float32))
                label = int(i.split('_')[0])  # 从文件名获取类别编号
                self.labels.append(label)
        self.size = len(self.datas)

    def __getitem__(self, index):
        data = torch.from_numpy(self.datas[index])
        label = torch.tensor(self.labels[index])
        return data, label

    def __len__(self):
        return self.size

test = MyData(train=False)