import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening Video File.")
try:
    while True:
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        key = cv2.waitKey(20) & 0xff

        # image_out = np.zeros_like(frame)
        image_out = frame.copy()
        for i in range(3):
            image_out[:, :, i] = frame[:, :, 1]

        cv2.imshow('Frame', image_out)

        if key == 27 | (not ret):
            cv2.destroyAllWindows()
            cap.release()
            break
except:
    print("Video has ended.")
