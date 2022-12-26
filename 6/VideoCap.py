import pandas as pd
import numpy as np
import cv2

vid = cv2.VideoCapture("test3.mp4")
cv2.namedWindow("SETTINGS")
cv2.createTrackbar('thresh', "SETTINGS", 0, 255, lambda x: x)
cv2.createTrackbar('area_size', "SETTINGS", 50, 10000, lambda x: x)
cv2.createTrackbar('fil_par', "SETTINGS", 50, 255, lambda x: x)
cv2.createTrackbar('matrix_size', "SETTINGS", 1, 25, lambda x: x)
cv2.createTrackbar('iters', "SETTINGS", 7, 10, lambda x: x)
cv2.createTrackbar('accx', "SETTINGS", 700, 800, lambda x: x)
cv2.createTrackbar('accy', "SETTINGS", 700, 800, lambda x: x)
# cv2.createTrackbar('apper', "SETTINGS", 3, 6, lambda x: x if x % 2 == 1 else x + 1)

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

    if prev_frame is None:
        prev_frame = cur_gray
    else:
        pass

    # blur frame
    cur_blurred = cv2.GaussianBlur(cur_gray, (21, 21), 0)
    cur_blurred = cv2.blur(cur_blurred, (5, 5))

    accx = cv2.getTrackbarPos('accx', "SETTINGS")
    accy = cv2.getTrackbarPos('accy', "SETTINGS")
    # apper = cv2.getTrackbarPos('apper', "SETTINGS")
    grounds = cv2.Canny(cur_blurred.copy(), accx, accy, apertureSize=5)
    dilated = cv2.dilate(grounds.copy(), None, 2)
    prev_dilated = cv2.dilate(prev_frame.copy(), None, 2)

    diff = cv2.absdiff(dilated, prev_dilated)
    thresh_ = cv2.getTrackbarPos('thresh', "SETTINGS")
    thresh = cv2.threshold(diff.copy(), thresh_, 255, cv2.THRESH_BINARY)[1]
    mx_s = cv2.getTrackbarPos('matrix_size', "SETTINGS")
    iter = cv2.getTrackbarPos('iterations', "SETTINGS")
    th_dil = cv2.dilate(thresh, np.ones((mx_s, mx_s), 'uint8'), iter)

    cnts = cv2.findContours(th_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    area = cv2.getTrackbarPos('area_size', "SETTINGS")
    for contour in cnts:
        if cv2.contourArea(contour) > area:
            cv2.drawContours(cur_frame, contour, 3, (0, 0, 255), 3)
    cv2.imshow("diff", cur_frame)

    # find diff frame
    diff = cv2.absdiff(prev_frame, cur_blurred)

    prev_frame = grounds
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
