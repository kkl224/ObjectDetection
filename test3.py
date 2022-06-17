import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change threshold
params.minThreshold = 10
params.maxThreshold = 220

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0.3

# Filter by Area.
params.filterByArea = True
params.minArea = 200
params.maxArea = 20000
#params.minRepeatability = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.4

params.filterByColor = True
params.blobColor = 255

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: 
        break

    hsvframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower = np.array([70, 80, 50]) #80, 40, 30
    upper = np.array([140, 255, 255]) 

    mask = cv.inRange(hsvframe, lower, upper)

    #blue = cv.bitwise_and(frame, frame, mask=mask)

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mask)
    if len(keypoints) != 0:
        print(len(keypoints), 'blob found!')

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles.
    blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(blobs)
    cv.imshow('Blobs', blobs)
    cv.imshow('Mask', mask)
    #cv.imshow('Blue', blue)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break