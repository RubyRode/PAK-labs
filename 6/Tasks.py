import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os

path = r"archive/nails_segmentation"
dictionary = {}

for filename in os.listdir(path + r"/images"):
    image_path = os.path.join(path + r"/images", filename)
    label_path = os.path.join(path + r"/labels", filename)

    if filename in os.listdir(path + r"/labels"):
        dictionary[image_path] = label_path

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_bgr[:, :, ::-1]

    label_bgr = cv2.imread(label_path)
    label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
    label_rgb = label_bgr[:, :, ::-1]

    res = np.hstack((label_rgb, img_rgb))
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.imshow("res", res)
    cv2.waitKey(0)


    label_gray = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(label_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_out = img_rgb.copy()
    image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    image_out = cv2.drawContours(image_out, contours, -1, (0, 255, 0), 2)
    cv2.imshow("out", image_out)

    cv2.waitKey(0)
