import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture(0)

font = cv.FONT_HERSHEY_SIMPLEX

params = cv.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 220

params.filterByInertia = True
params.minInertiaRatio = 0.5

params.filterByArea = True
params.minArea = 500
params.maxArea = 1000000000

params.filterByConvexity = True
params.minConvexity = 0.3

params.filterByCircularity = True
params.minCircularity = 0.6

params.filterByColor = True
params.blobColor = 255

def findGreen(frame):

    blurframe = cv.GaussianBlur(frame, (15,15), 0)
    hsvframe = cv.cvtColor(blurframe, cv.COLOR_BGR2HSV)

    lower = np.array([42, 40, 20]) 
    upper = np.array([60, 255, 255])

    mask = cv.inRange(hsvframe, lower, upper)
    mask = cv.erode(mask, None, iterations=3)
    mask = cv.dilate(mask, None, iterations=6)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 10000):
            x, y, w, h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Tracking Target"
            cv.putText(frame, text, (x, y), font, 1.2, (0, 255, 0))

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)

    blobCount = len(keypoints)
    text = "Count=" + str(blobCount) 
    cv.putText(frame, text, (5,25), font, 1, (0, 0, 255), 2)

    blank = np.zeros((1, 1))

    blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    centroids = cv.drawKeypoints(frame, keypoints, frame, (0, 0, 255), flags=0)

    cv.imshow('Blobs', blobs)
    cv.imshow('Mask', mask)

def findBlue(frame):

    blurframe = cv.GaussianBlur(frame, (15,15), 0)
    hsvframe = cv.cvtColor(blurframe, cv.COLOR_BGR2HSV)

    lower = np.array([22, 40, 10]) 
    upper = np.array([60, 255, 255])

    mask = cv.inRange(hsvframe, lower, upper)
    mask = cv.erode(mask, None, iterations=3)
    mask = cv.dilate(mask, None, iterations=6)

    detector = cv.SimpleBlobDetector_create(params)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 10000):
            x, y, w, h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Target"
            cv.putText(frame, text, (x, y), font, 1, (0, 255, 0))

    keypoints = detector.detect(mask)

    blobCount = len(keypoints)
    text = "Count=" + str(blobCount) 
    cv.putText(frame, text, (5,25), font, 1, (0, 0, 255), 2)

    blank = np.zeros((1, 1))

    blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    centroids = cv.drawKeypoints(frame, keypoints, frame, (0, 0, 255), flags=0)

    cv.imshow('Blobs', blobs)
    cv.imshow('Mask', mask)



while(1):

    ret, frame = videoCapture.read()
    if not ret: 
        break

    findGreen(frame)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
 