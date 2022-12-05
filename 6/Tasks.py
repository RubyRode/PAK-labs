import matplotlib.pyplot as plt
import keyboard
import cv2
import os
import numpy as np

path = r"/labs/6/archive/nails_segmentation"
dictionary = {}


class Segmentation:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        # assert len(self._images) == len(self._labels)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        images, labels = cv2.imread(self._images)

for filename in os.listdir(path + r"\images"):
    image_path = os.path.join(path + r"\images", filename)
    label_path = os.path.join(path + r"\labels", filename)

    if filename in os.listdir(path + r"\labels"):
        dictionary[image_path] = label_path

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_bgr[:, :, ::-1]
    plt.imshow(img_rgb)
    plt.show()

    keyboard.wait('right')

    label_bgr = cv2.imread(label_path)
    label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
    label_rgb = label_bgr[:, :, ::-1]
    plt.imshow(label_rgb)
    plt.show()

    keyboard.wait('right')

    label_gray = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(label_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_out = img_rgb.copy()
    image_out = cv2.drawContours(image_out, contours, -1, (0, 255, 0), 2)

    plt.imshow(image_out)
    plt.show()

    keyboard.wait('right')



