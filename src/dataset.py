import os
import sys
sys.path.append('..')
import paths
import numpy as np
import cv2
from torch.utils.data import Dataset


class CelebA(Dataset):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    def __init__(self, dataset_type=TRAIN):
        self.master_csv_file_path = os.path.join(paths.data_folder_path, 'master.csv')
        with open(self.master_csv_file_path) as master_csv_file:
            rows = master_csv_file.read().splitlines()
        self.images_folder_path = os.path.join(paths.data_folder_path, 'Img', 'img_celeba')
        self.instances = []
        header = True
        for row in rows:
            if header:
                header = False
            else:
                elements = row.split(',')
                if int(elements[1]) == dataset_type:
                    # Every instance: (image_file_name, dataset_type, identity, anchor_x, anchor_y, width,height)
                    self.instances.append([elements[0], int(elements[2]), int(elements[3]), int(elements[4]), int(elements[5]), int(elements[6])])
        self.num_instances = len(self.instances)

    def __len__(self):
        return self.num_instances

    @staticmethod
    def bring_image_channels_first(image):
        return np.transpose(image, axes=(2, 0, 1))

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.images_folder_path, self.instances[idx][0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        print(image.shape)
        image = CelebA.bring_image_channels_first(image)
        print(image.shape)
        return CelebA.bring_image_channels_first(image), np.asarray(self.instances[idx][2:])

    def get_instance(self, idx):
        return self.instances[idx]

    def shuffle(self):
        np.random.shuffle(self.instances)
