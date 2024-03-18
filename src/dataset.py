import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import math

class Mydataset(Dataset):
    def __init__(self, root, total_images_per_class=1000, ratio = 0.8, mode = 'train'):
        self.root = root
        self.num_classes = 20

        if mode == "train":
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class * ratio)
        else:
            self.offset = int(total_images_per_class * ratio)
            self.num_images_per_class = math.ceil(total_images_per_class * (1 - ratio))

        self.num_samples = self.num_images_per_class * self.num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        self.data_path = []
        for file in sorted(os.listdir(self.root)):
            self.data_path.append(os.path.join(self.root, file))

        num_label = int(index / self.num_images_per_class)
        image = np.load(self.data_path[num_label]).astype(np.float32)[self.offset + index % self.num_images_per_class]
        image = image.reshape((1, 28, 28))
        image = torch.tensor(image)

        #image = image.reshape((1, 28, 28))
        return image, num_label

if __name__ == '__main__':
    Test = Mydataset(root='/home/minhtran/Documents/MDEEP_LEARNING/Project-QuickDraw/data', mode='train')
