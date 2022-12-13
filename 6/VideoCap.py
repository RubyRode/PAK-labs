import cv2
import numpy as np


cv2.namedWindow("RESULT")
cv2.namedWindow("SETTINGS")

cap = cv2.VideoCapture(0)
cv2.createTrackbar('h1', "SETTINGS", 0, 255, lambda x:x)
cv2.createTrackbar('s1', "SETTINGS", 0, 255, lambda x:x)
cv2.createTrackbar('v1', "SETTINGS", 0, 255, lambda x:x)
cv2.createTrackbar('h2', "SETTINGS", 255, 255, lambda x:x)
cv2.createTrackbar('s2', "SETTINGS", 255, 255, lambda x:x)
cv2.createTrackbar('v2', "SETTINGS", 255, 255, lambda x:x)
crange = [0, 0, 0, 0, 0, 0]
ret, frame1 = cap.read()
while True:

    ret, frame2 = cap.read()
    key = cv2.waitKey(20) & 0xff
    if ret:

        hsv_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        h1 = cv2.getTrackbarPos('h1', "SETTINGS")
        s1 = cv2.getTrackbarPos('s1', "SETTINGS")
        v1 = cv2.getTrackbarPos('v1', "SETTINGS")
        h2 = cv2.getTrackbarPos('h2', "SETTINGS")
        s2 = cv2.getTrackbarPos('s2', "SETTINGS")
        v2 = cv2.getTrackbarPos('v2', "SETTINGS")

        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)
        thresh = cv2.inRange(hsv_2, h_min, h_max)



        cv2.imshow('Frame', thresh)

    frame1 = frame2
    if key == 27 | (not ret):
        cv2.destroyAllWindows()
        cap.release()
        break
