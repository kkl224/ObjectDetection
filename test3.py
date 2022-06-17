import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Capturing video through webcam
videoCapture = cv.VideoCapture(0)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.thresholdStep = 10
params.minThreshold = 10
params.maxThreshold = 220
params.minRepeatability = 2

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Filter by Area.
params.filterByArea = True
params.minArea = 200
params.minArea = 20000

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2

params.filterByColor = True

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: 
        break

    hsvframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower = np.array([60, 70, 30]) #80, 40, 30
    upper = np.array([105, 255, 255]) 

    mask = cv.inRange(hsvframe, lower, upper)

    blue = cv.bitwise_and(frame, frame, mask=mask)

    lower2 = np.array([50, 0, 0])
    upper2 = np.array([55, 50, 50])

    ## mask of blue color
    mask2 = cv.inRange(blue, lower2, upper2)

    noblue = cv.bitwise_and(blue, blue, mask2=mask2)

    #print(blue.dtype, blue.shape)
    #print(mask.dtype, mask.shape)

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(blue)

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles.
    blobs = cv.drawKeypoints(blue, keypoints, blank, (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(blobs)
    cv.imshow('Blobs', blobs)
    cv.imshow('NoBlue', noblue)
    #cv.imshow('Blue', blue)
    #cv.imshow('frame', mask)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break