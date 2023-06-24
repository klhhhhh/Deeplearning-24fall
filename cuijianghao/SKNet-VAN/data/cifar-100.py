import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_as_image(data_dict, directory):
    for i, img_flat in enumerate(data_dict[b'data']):
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        Image.fromarray(img).save(os.path.join(directory, str(data_dict[b'fine_labels'][i]) + '_' + str(i) + '.png'))

def main():
    path = r"C:\Users\cuiji\Desktop\Deep_Learning\SKNet\data\cifar-100-python"  # 将此路径替换为你的数据集解压路径
    train_dir = "train/"
    test_dir = "test/"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with open(os.path.join(path, 'train'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        save_as_image(data_dict, train_dir)

    with open(os.path.join(path, 'test'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        save_as_image(data_dict, test_dir)

if __name__ == '__main__':
    main()