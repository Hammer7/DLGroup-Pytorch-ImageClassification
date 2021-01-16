import numpy as np
from torch.utils import data

class DataGenerator(data.Dataset):
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.subset =subset
        self.datasize = len(self.dataset.labels_dict[subset])

    def __len__(self):
        return self.datasize

    def __getitem__(self, i):
        images, labels = self.dataset.load_single(self.subset, i)
        norm_images = (images - self.dataset.mean) / self.dataset.stddev

        return norm_images, labels
