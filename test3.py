import cv2 as cv
import numpy as np

def nothing(X):
    pass

cv.nameWindow("Tracking")
cv.createTracker("LH", "Tracking", 0, 255, nothing)
cv.createTracker("LS", "Tracking", 0, 255, nothing)
cv.createTracker("LV", "Tracking", 0, 255, nothing)
cv.createTracker("UH", "Tracking", 255, 255, nothing)
cv.createTracker("US", "Tracking", 255, 255, nothing)
cv.createTracker("UV", "Tracking", 255, 255, nothing)

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: break

    hsvframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lh = cv.getTrackbarPos("LH", "Tracking")
    ls = cv.getTrackbarPos("LS", "Tracking")
    lv = cv.getTrackbarPos("LV", "Tracking")

    uh = cv.getTrackbarPos("UH", "Tracking")
    us = cv.getTrackbarPos("US", "Tracking")
    uv = cv.getTrackbarPos("UV", "Tracking")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, 255, 255])

    mask = cv.inRange(hsvframe, lower, upper)

    blobs = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', blobs)
    cv.imshow('mask', mask)
    cv.imshow('hsv', hsvframe)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
