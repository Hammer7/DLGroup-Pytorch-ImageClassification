import os, re

import numpy as np
from PIL import Image
import cv2 as cv 
from sklearn.model_selection import train_test_split

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def load_resize_image(fullpath, target_size):
    img = Image.open(fullpath)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size is not None:
        img = img.resize(target_size)
    img = np.asarray(img)
    return img

class Dataset():
    TRAIN = 'Train'
    VAL = 'Val'
    TEST = 'Test'

    def __init__(self):
        self.path = '/content/drive/MyDrive/DLGroup/weather_dataset'
        self._indices = {}

    def list_images(self):
        l_images = []
        l_labels = []
        for subfolder in sorted_alphanumeric(os.listdir(self.path)):
            label = int(subfolder.split('_', 1)[0])

            class_path = os.path.join(self.path, subfolder)
            class_images = [os.path.join(class_path, p) for p in os.listdir(class_path)]
            l_images.extend(class_images)
            l_labels.extend([label] * len(class_images))

        return np.array(l_images, dtype=object), np.array(l_labels)

    def prepare_dataset(self):
        rs = 42
        tvt = (.8, .1, .1)

        self.image_paths, self.labels = self.list_images()
        total_size = len(self.labels)
        indices = np.arange(total_size)
        trainval_idx, test_idx =  train_test_split(indices, test_size = tvt[2], random_state = rs)
        test_size2 = tvt[1] / (1. - tvt[2])
        train_idx, val_idx =  train_test_split(trainval_idx, test_size = test_size2, random_state = rs)

        self._indices[Dataset.TRAIN] = train_idx
        self._indices[Dataset.VAL] = val_idx
        self._indices[Dataset.TEST] = test_idx

        self.images_dict = {}
        self.labels_dict = {}

        self.images_dict[Dataset.TRAIN], self.labels_dict[Dataset.TRAIN] = self.preload_image_labels(Dataset.TRAIN)
        self.images_dict[Dataset.VAL], self.labels_dict[Dataset.VAL] = self.preload_image_labels(Dataset.VAL)
        self.images_dict[Dataset.TEST], self.labels_dict[Dataset.TEST] = self.preload_image_labels(Dataset.TEST)

        self.mean = np.mean(self.images_dict[Dataset.TRAIN], axis=(0, 1, 2)) # mean for data centering
        self.stddev = np.std(self.images_dict[Dataset.TRAIN], axis=(0, 1, 2)) # std for data normalization

    def preload_image_labels(self, subset):
        if subset not in self._indices:
            raise ValueError(f'{subset} set is not found in indices dictionary')
        
        self.image_size = (128, 128)
        indices = self._indices[subset]
        #indices = indices[:32]

        image_dim = (len(indices), ) + self.image_size + (3, )
        images_array = np.zeros(image_dim, dtype=np.float32)
        for i, filename in enumerate(self.image_paths[indices]):
            images_array[i] = load_resize_image(filename, (128, 128))
            print(f'loading {subset} image... {i} - {os.path.basename(filename)}')

        return images_array, self.labels[indices]

    def load_batch(self, subset, start, end):
        return self.images_dict[subset][start:end], self.labels_dict[subset][start:end]


    def load_single(self, subset, i):
        return self.images_dict[subset][i], self.labels_dict[subset][i]