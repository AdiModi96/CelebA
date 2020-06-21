import numpy as np
from dataset import CelebA
from matplotlib import pyplot as plt
import cv2


def test_instance(db=None):
    instance_idx = 39
    image, label, (bbox_tl_x, bbox_tl_y, bbox_br_x, bbox_br_y) = db[instance_idx]
    print('Instance Index: {}'.format(instance_idx))
    print('Label: {}'.format(label))
    print('Image Shape (Width × Height): ({} × {})'.format(image.shape[1], image.shape[0]))
    print('Facial Shape (Width × Height): ({} × {})'.format(bbox_br_x - bbox_tl_x, bbox_br_y - bbox_tl_y))
    print('Facial Area: {}'.format((bbox_br_x - bbox_tl_x) * (bbox_br_y - bbox_tl_y)))

    plt.figure(num='Dataset Tester', figsize=(10, 10))
    plt.title('Identity: {}'.format(label))
    image = cv2.rectangle(image, (int(bbox_tl_x), int(bbox_tl_y)), (int(bbox_br_x), int(bbox_br_y)), color=(1, 0, 0), thickness=2)
    plt.imshow(image, cmap='gray')
    plt.show()


db = CelebA(dataset_type=CelebA.VALIDATION)
# db.shuffle()
test_instance(db)
