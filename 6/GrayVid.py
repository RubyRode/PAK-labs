import cv2


cap = cv2.VideoCapture("video_2022-12-06_06-07-29.mp4")
if not cap.isOpened():
    print("Error opening Video File.")
try:
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        key = cv2.waitKey(20) & 0xff
        cv2.imshow('Frame', frame)
        if key == 27 | (not ret):
            cv2.destroyAllWindows()
            cap.release()
            break
except:
    print("Video has ended.")