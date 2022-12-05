import cv2


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    key = cv2.waitKey(20) & 0xff
    cv2.imshow('Frame', frame)

    if key == 27 | (not ret):
        cv2.destroyAllWindows()
        cap.release()
        break
