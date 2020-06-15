import numpy as np
from torch.utils.data import DataLoader
from dataset import CelebA
import matplotlib.pyplot as plt
import cv2


def test_instance(db=None):

    instance_idx = np.random.randint(0, len(db), 1)[0]
    image, (anchor_x, anchor_y, width, height) = db[instance_idx]
    point_tl = (anchor_x, anchor_y)
    point_br = (anchor_x + width, anchor_y + height)

    plt.figure(num='Dataset Tester', figsize=(10, 10))
    plt.title('Instance Index: {}'.format(instance_idx))
    image = cv2.rectangle(image, point_tl, point_br, color=(255, 0, 0), thickness=2)
    plt.imshow(image, cmap='gray')
    plt.show()

db = CelebA(dataset_type=CelebA.VALIDATION)
db.shuffle()
image, (anchor_x, anchor_y, width, height) = db[0]
# test_instance(db)
