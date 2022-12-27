import pandas as pd
import numpy as np
import cv2

vid = cv2.VideoCapture(0)
cv2.namedWindow("SETTINGS")
cv2.createTrackbar('fil_par', "SETTINGS", 50, 255, lambda x: x)
cv2.createTrackbar('matrix_size', "SETTINGS", 1, 25, lambda x: x)
cv2.createTrackbar('iters', "SETTINGS", 7, 10, lambda x: x)
cv2.createTrackbar('accx', "SETTINGS", 700, 800, lambda x: x)
cv2.createTrackbar('accy', "SETTINGS", 700, 800, lambda x: x)
cv2.createTrackbar('pre_dilate', "SETTINGS", 0, 30, lambda x: x)
cv2.createTrackbar('thresh', "SETTINGS", 0, 255, lambda x: x)

prev_frame = None
ret = True
while ret:

    ret, cur_frame = vid.read()

    if not ret:
        break

    # Filter an image
    par = cv2.getTrackbarPos('fil_par', "SETTINGS")
    cont_kernel = np.array([
        [0, -1, 0],
        [-1, par / 10, -1],
        [0, -1, 0]
    ])
    filtered_cur = cv2.filter2D(cur_frame, -1, cont_kernel)

    # convert frame to gray colors
    cur_gray = cv2.cvtColor(filtered_cur, cv2.COLOR_RGB2GRAY)



    # blur frame
    cur_blurred = cv2.GaussianBlur(cur_gray, (21, 21), 0)
    # cur_blurred = cv2.blur(cur_blurred, (5, 5))
    if prev_frame is None:
        prev_frame = cur_blurred
        prev_fr_blurred = cur_blurred.copy()
    else:
        pass

    diff1 = cv2.absdiff(cur_blurred, prev_fr_blurred)
    tmp = cv2.getTrackbarPos('thresh', "SETTINGS")
    thresh1 = cv2.threshold(diff1, tmp, 255, cv2.THRESH_BINARY)[1]
    dil1 = cv2.dilate(thresh1.copy(), np.ones((5, 5), 'uint8'), 3)

    accx = cv2.getTrackbarPos('accx', "SETTINGS")
    accy = cv2.getTrackbarPos('accy', "SETTINGS")
    grounds = cv2.Canny(cur_blurred.copy(), accx, accy, apertureSize=5)

    pre_dilate = cv2.getTrackbarPos("pre_dilate", "SETTINGS")

    dilated = cv2.dilate(grounds.copy(), np.ones((pre_dilate + 10, pre_dilate + 10), 'uint8'), 7)
    prev_dilated = cv2.dilate(prev_frame.copy(), np.ones((pre_dilate, pre_dilate), 'uint8'), 7)

    diff = cv2.absdiff(prev_dilated, dilated)
    mx_s = cv2.getTrackbarPos('matrix_size', "SETTINGS")
    iter = cv2.getTrackbarPos('iterations', "SETTINGS")
    th_dil = cv2.dilate(diff, np.ones((mx_s, mx_s), 'uint8'), iter)
    invert = cv2.bitwise_or(th_dil, dil1)

    cv2.imshow("diff", invert)

    # find diff frame
    # diff = cv2.absdiff(prev_frame, cur_blurred)

    prev_frame = dilated
    prev_fr_blurred = cur_blurred
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
