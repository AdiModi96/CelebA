import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import paths


class CelebA(Dataset):
    ANCHORS = None
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    NORMALIZED_IMAGE_SHAPE = 800

    def __init__(self, dataset_type=TRAIN):
        self.master_csv_file_path = os.path.join(paths.data_folder_path, 'master.csv')

        # Checking if master_csv_file_path exists
        if not os.path.isfile(self.master_csv_file_path):
            print('Master CSV file doesn\'t exist.')
            print('Quitting...')

        # Checking if images_folder_path exists
        self.images_folder_path = os.path.join(paths.data_folder_path, 'Img', 'img_celeba')
        if not os.path.isdir(self.images_folder_path):
            print('Images folder doesn\'t exist.')
            print('Quitting...')

        with open(self.master_csv_file_path) as master_csv_file:
            rows = master_csv_file.read().splitlines()

        self.instances = []
        self.unique_labels_and_images = {}
        header = True
        for row in rows:
            if header:
                header = False
            else:
                elements = row.split(',')
                if int(elements[1]) == dataset_type:
                    # Every instance: (image_file_name, identity, bbox_tl_x, bbox_tl_y, bbox_br_x, bbox_br_y)
                    self.instances.append([elements[0], int(elements[2]), float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])])

                    if elements[2] in self.unique_labels_and_images.keys():
                        self.unique_labels_and_images[int(elements[2])].append(elements[0])
                    else:
                        self.unique_labels_and_images[int(elements[2])] = [elements[0]]

        self.num_instances = len(self.instances)
        self.unique_labels = sorted(self.unique_labels_and_images.keys())
        self.num_unique_labels = len(self.unique_labels)

    @staticmethod
    def bring_image_channels_first(image):
        return np.transpose(np.asarray(image), axes=(2, 0, 1))

    @staticmethod
    def normalize_image_shape_and_bbox(image, bbox):
        width = image.shape[1]
        height = image.shape[0]
        image_shape = max(width, height)
        if height < width:
            padding_space = width - height
            top_padding = padding_space // 2
            bottom_padding = padding_space - top_padding
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(1, 1, 1))
            bbox[1] += top_padding
            bbox[3] += top_padding
        elif width < height:
            padding_space = height - width
            left_padding = padding_space // 2
            right_padding = padding_space - left_padding
            image = cv2.copyMakeBorder(image, 0, 0, left_padding, right_padding, borderType=cv2.BORDER_CONSTANT, value=(1, 1, 1))
            bbox[0] += left_padding
            bbox[2] += left_padding

        if image_shape > CelebA.NORMALIZED_IMAGE_SHAPE:
            scaling_factor = image_shape / CelebA.NORMALIZED_IMAGE_SHAPE
            image = cv2.resize(image, (CelebA.NORMALIZED_IMAGE_SHAPE, CelebA.NORMALIZED_IMAGE_SHAPE))
            bbox = bbox / scaling_factor
        elif image_shape < CelebA.NORMALIZED_IMAGE_SHAPE:
            anchor_idx = np.random.randint(low=0, high=CelebA.NORMALIZED_IMAGE_SHAPE - image_shape, size=2)
            top_padding = anchor_idx[1]
            bottom_padding = CelebA.NORMALIZED_IMAGE_SHAPE - (top_padding + image_shape)
            left_padding = anchor_idx[0]
            right_padding = CelebA.NORMALIZED_IMAGE_SHAPE - (bottom_padding + image_shape)

            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, borderType=cv2.BORDER_CONSTANT, value=(1, 1, 1))
            bbox[0] += left_padding
            bbox[1] += right_padding
            bbox[2] += left_padding
            bbox[3] += right_padding

        return image, bbox

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        idx = idx % self.num_instances
        image = cv2.imread(os.path.join(self.images_folder_path, self.instances[idx][0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        label = self.instances[idx][1]
        bbox = np.asarray(self.instances[idx][2:])
        normalized_image, normalized_bbox = CelebA.normalize_image_shape_and_bbox(image, bbox=bbox)
        return normalized_image, label, normalized_bbox

    def get_instance(self, idx):
        return self.instances[idx]

    def shuffle(self):
        np.random.shuffle(self.instances)


db = CelebA()
