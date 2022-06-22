import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Capturing video through webcam
videoCapture = cv.VideoCapture(0)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()
# Change Threshold
params.minThreshold = 10
params.maxThreshold = 220
# Filter by Inertia Ratio
params.filterByInertia = True
params.minInertiaRatio = 0.5
# Filter by Area
params.filterByArea = True
params.minArea = 500
params.maxArea = 1000000000
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
# Filter by Color
params.filterByColor = True
params.blobColor = 255

# Font to write text overlay
font = cv.FONT_HERSHEY_SIMPLEX

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: 
        break

    # get vcap property 
    width  = int(videoCapture.get(3))   # int(float `width`)
    height = int(videoCapture.get(4))  # int(float `height`)
    
    cv.line(frame, (0, 0), (width, height), (0, 255, 0), 2)
    cv.line(frame, (0, height), (width, 0), (0, 255, 0), 2)

    blurframe = cv.GaussianBlur(frame, (11,11), 0)
    #blurframe = cv.medianBlur(frame, 9)

    hsvframe = cv.cvtColor(blurframe, cv.COLOR_BGR2HSV)

    lower = np.array([80, 90, 10]) #80, 40, 30
    upper = np.array([130, 255, 255]) 

    mask = cv.inRange(hsvframe, lower, upper)
    mask = cv.erode(mask, None, iterations=4)
    mask = cv.dilate(mask, None, iterations=8)

    #blue = cv.bitwise_and(frame, frame, mask=mask)

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mask)

    if keypoints:

        # Get the number of blobs found
        blobCount = len(keypoints)
        text = "Count=" + str(blobCount) 
        cv.putText(frame, text, (5,50), font, 2, (0, 0, 255), 2)

        # Write X position of first blob
        blob_x = keypoints[0].pt[0]
        text1 = "X=" + "{:.2f}".format(blob_x )
        cv.putText(frame, text1, (5,100), font, 2, (0, 0, 255), 2)

        # Write Y position of first blob
        blob_y = keypoints[0].pt[1]
        text2 = "Y=" + "{:.2f}".format(blob_y)
        cv.putText(frame, text2, (5,150), font, 2, (0, 0, 255), 2)

        # Get distance of largest circle
        my_circle = sorted(keypoints, key=(lambda x: x.size), reverse=True)[0]
        p = my_circle.size / 2.0    # perceived width, in pixels
        w = 0.25          # approx. actual width, in meters (pre-computed)
        f = 1461             # camera focal length, in pixels (pre-computed)
        d = f * w / p
        cv.putText(frame, "Distance=%.3fm" % d, (5,200), font, 2, (0, 0, 255), 2)

    else: 
        
        text = "Count=0"
        cv.putText(frame, text, (5,25), font, 2, (0, 0, 255), 2)

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles
    blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    centroids = cv.drawKeypoints(frame, keypoints, frame, (0, 0, 255), flags=0)

    #plt.imshow(blobs)
    #plt.imshow(centroids)
    cv.imshow('Blobs', blobs)
    cv.imshow('Mask', mask)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break