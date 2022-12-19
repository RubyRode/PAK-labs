import pandas as pd
import numpy as np
import cv2

vid = cv2.VideoCapture(0)
ret, prev_frame = vid.read()
prev_frame1 = prev_frame.copy()
# prepare the first frame
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
prev_blured = cv2.GaussianBlur(prev_gray, (21, 21), 0)
cv2.namedWindow("SETTINGS")
cv2.createTrackbar('thresh_sta', "SETTINGS", 0, 255, lambda x: x)
cv2.createTrackbar('thresh_mov', "SETTINGS", 0, 255, lambda x: x)
# cv2.createTrackbar('fil_par', "SETTINGS", 0, 255, lambda x: x)
cv2.createTrackbar('matrix_size', "SETTINGS", 0, 25, lambda x: x)
cv2.createTrackbar('iters', "SETTINGS", 0, 10, lambda x: x)

while ret:
    ret, cur_frame = vid.read()
    if not ret:
        break

    # par = cv2.getTrackbarPos('fil_par', "SETTINGS")
    cont_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    filtered_cur = cv2.filter2D(cur_frame, -1,  cont_kernel)

    # convert frame to gray colors
    cur_gray = cv2.cvtColor(filtered_cur, cv2.COLOR_RGB2GRAY)

    # blur frame
    cur_blurred = cv2.GaussianBlur(cur_gray, (21, 21), 0)

    # find diff frame
    diff = cv2.absdiff(prev_blured, cur_blurred)
    # threshed frame
    thresh_sta = cv2.getTrackbarPos('thresh_sta', "SETTINGS")
    thresh_mov = cv2.getTrackbarPos('thresh_mov', "SETTINGS")

    threshed_sta = cv2.threshold(diff, thresh_sta, 255, cv2.THRESH_BINARY)[1]
    threshed_mov = cv2.threshold(diff, thresh_mov, 255, cv2.THRESH_BINARY)[1]
    # dilated frame
    dilated_sta = cv2.dilate(threshed_sta, None, iterations=2)
    iters = cv2.getTrackbarPos('iters', "SETTINGS")
    mat_size = cv2.getTrackbarPos('matrix_size', "SETTINGS")
    dilated_mov = cv2.dilate(threshed_mov, np.ones((mat_size, mat_size), 'uint8'), iterations=iters)
    # find contour
    cnts_sta = cv2.findContours(dilated_sta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_mov = cv2.findContours(dilated_mov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # draw contours
    cv2.drawContours(cur_frame, cnts_sta, -1, (0, 255, 0), 3)
    cv2.drawContours(cur_frame, cnts_mov, -1, (0, 0, 255), 3)

    prev_blured = cur_blurred

    cv2.imshow("cur", cur_frame)
    cv2.imshow("filtered", filtered_cur)

    key = cv2.waitKey(20)
    if key & 0xFF == ord("q"):
        break


vid.release()
cv2.destroyAllWindows()
