import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset, transfor):
        self.data = dataset[0].astype(np.float32)
        self.data_LIDAR = dataset[1].astype(np.float32)
        self.transformer = transfor
        self.labels = []
        for n in dataset[2]:
            self.labels += [int(n)]

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transformer == None:
            img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
            img_LIDAR = torch.from_numpy(np.asarray(self.data_LIDAR[index,:,:,:]))
            return img, img_LIDAR, label
        elif len(self.transformer) == 2:
            img = torch.from_numpy(np.asarray(self.transformer[1](self.transformer[0](self.data[index,:,:,:]))))
            img_LIDAR = torch.from_numpy(np.asarray(self.transformer[1](self.transformer[0](self.data_LIDAR[index,:,:,:]))))
            return img, img_LIDAR, label
        else:
            img = torch.from_numpy(np.asarray(self.transformer[0](self.data[index,:,:,:])))
            img_LIDAR = torch.from_numpy(np.asarray(self.transformer[0](self.data_LIDAR[index,:,:,:])))
            return img, img_LIDAR, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels


