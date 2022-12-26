import numpy as np
import cv2


def switch(gm):
    """for switching the game state"""
    return not gm


vid = cv2.VideoCapture(0)
cv2.namedWindow("RESULT")
cv2.namedWindow("RESULT2")
cv2.createTrackbar('thresh', "RESULT", 15, 255, lambda x: x)
cv2.createTrackbar('canny_x', "RESULT", 20, 255, lambda x: x)
cv2.createTrackbar('canny_y', "RESULT", 20, 255, lambda x: x)
cv2.createTrackbar('area_sta', "RESULT2", 100, 1000, lambda x: x)
cv2.createTrackbar('area_mov', "RESULT2", 100, 1000, lambda x: x)
g_b_m_size = 21
# read first frame and prepare for future changes
ret, prev_frame = vid.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (g_b_m_size, g_b_m_size), 0)

game_state = False
while ret:
    # read current frame and prepare it for future changes
    ret, cur_frame = vid.read()
    orig = cur_frame.copy()
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.GaussianBlur(cur_frame, (g_b_m_size, g_b_m_size), 0)

    # find the difference between two frames and threshold them => motion capture mask
    diff = cv2.absdiff(cur_frame, prev_frame)
    threshold = cv2.getTrackbarPos('thresh', "RESULT")
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # find the mask for boundaries of the static objects => static objects mask
    canny_x = cv2.getTrackbarPos('canny_x', "RESULT")
    canny_y = cv2.getTrackbarPos('canny_y', "RESULT")
    static = cv2.Canny(cur_frame.copy(), canny_x, canny_y)

    # dilate the masks
    mov_erode = cv2.erode(thresh, None, 2)
    mov_dilate = cv2.dilate(mov_erode, np.ones((3, 3), 'uint8'), 3)
    sta_dilate = cv2.dilate(static, np.ones((3, 3), 'uint8'), 2)

    if game_state:
        # find contours from static objects mask
        cnts_sta, _ = cv2.findContours(sta_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        area_sta = cv2.getTrackbarPos('area_sta', "RESULT2")

        # find contours with the perimeter more than area_sta
        cont_size_bool = []
        for i in range(0, len(cnts_sta)):
            if cv2.arcLength(cnts_sta[i], True) > area_sta:
                cont_size_bool.append(cnts_sta[i])

        # merging all found contours into one big contour
        if len(cont_size_bool) != 0:
            res_cont = np.vstack(cont_size_bool)
            cv2.drawContours(orig, res_cont, -1, (0, 255, 0), 2)

        # same process as above but for motion capture mask
        cnts_mov, _ = cv2.findContours(mov_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        area_mov = cv2.getTrackbarPos('area_mov', "RESULT2")

        cont_size_bool = []
        for i in range(0, len(cnts_mov)):
            if cv2.arcLength(cnts_mov[i], True) > area_mov:
                cont_size_bool.append(cnts_mov[i])
        if len(cont_size_bool) != 0:
            res_cont = np.vstack(cont_size_bool)
            cv2.drawContours(orig, res_cont, -1, (0, 0, 255), 2)

    # merging motion capture and static object masks
    res = np.hstack((sta_dilate, mov_dilate))

    # change frames
    prev_frame = cur_frame

    # show the results: two masks and the original video
    cv2.imshow("RESULT", res)
    cv2.imshow("RESULT2", orig)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    if key & 0xFF == ord("n"):
        game_state = switch(game_state)

vid.release()
cv2.destroyAllWindows()
