import numpy as np
import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    isTrue , frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_gray = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
    canny = cv.Canny(blur_gray,125,175)
    cv.imshow('Canny',canny)

    corners = cv.goodFeaturesToTrack(blur_gray, 100, 0.05, 50)
    corners = np.rint(corners).astype(int)

    for corner in corners:
        x,y = corner.ravel()
        cv.circle(frame,(x,y),5,(255,0,0),-1)

    # ret, corners = cv.findChessboardCorners(gray,(7,6),None)

    # if ret == True :
    #     frame = cv.drawChessboardCorners(frame, (7,6), corners, ret)

    cv.imshow('Frame',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break




capture.release()
cv.destroyAllWindows()