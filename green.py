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
params.minCircularity = 0.6
# Filter by Color
params.filterByColor = True
params.blobColor = 255

# Font to write text overlay
font = cv.FONT_HERSHEY_SIMPLEX

# Constant for focal length, in pixels (must be changed per camera)
FOCAL_LENGTH = 1460
BALLOON_WIDTH = 0.33

SCALE = 0.5

i = 1

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret:
        break

    frame = cv.resize(frame, None, fx=SCALE, fy=SCALE)

    # get vcap property
    width  = int(videoCapture.get(3))   # int(float `width`)
    height = int(videoCapture.get(4))  # int(float `height`)

    #cv.line(frame, (0, 0), (width, height), (255, 0, 0), 2)
    #cv.line(frame, (0, height), (width, 0), (255, 0, 0), 2)

    blurframe = cv.GaussianBlur(frame, (13,13), 0)
    #blurframe = cv.medianBlur(frame, 9)
    #cv.imshow('Blir', blurframe)

    hsvframe = cv.cvtColor(blurframe, cv.COLOR_BGR2HSV)
    #cv.imshow('HSV', hsvframe)

    #lower = np.array([30, 40, 90])
    #upper = np.array([70, 255, 255])

    lower = np.array([30, 40, 10])
    upper = np.array([60, 255, 255])

    mask = cv.inRange(hsvframe, lower, upper)
    #cv.imshow('Mask1', mask)
    mask = cv.erode(mask, None, iterations=4)
    #cv.imshow('Mask2', mask)
    mask = cv.dilate(mask, None, iterations=6)
    cv.imshow('Mask3', mask)

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Draw bounding boxes around contours
    box_width = 0
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 10000):
            x, y, w, h = cv.boundingRect(contour)
            box_width = w
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Tracking Target"
            cv.putText(frame, text, (x, y), font, 1.2, (0, 255, 0))

    # Detect blobs
    keypoints = detector.detect(mask)

    # Get the number of blobs found
    blobCount = len(keypoints)
    text = "Count = " + str(blobCount)
    cv.putText(frame, text, (5,50), font, 2, (0, 0, 255), 2)
    print(blobCount, "found")

    if box_width:
        p = box_width                    # perceived width, in pixels
        w = BALLOON_WIDTH                # approx. actual width, in meters (pre-computed)
        f = FOCAL_LENGTH * SCALE         # camera focal length, in pixels (pre-computed)
        d = f * w / p
        cv.putText(frame, "Distance=%.3fm" % d, (5,100), font, 2, (0, 0, 255), 2)

    """
    if keypoints:

        # Write X position of first blob
        blob_x = keypoints[0].pt[0]
        text1 = "X=" + "{:.2f}".format(blob_x )
        #cv.putText(frame, text1, (5,100), font, 1, (0, 0, 255), 2)

        # Write Y position of first blob
        blob_y = keypoints[0].pt[1]
        text2 = "Y=" + "{:.2f}".format(blob_y)
        #cv.putText(frame, text2, (5,150), font, 1, (0, 0, 255), 2)

        # Get distance of largest circle
        my_circle = sorted(keypoints, key=(lambda x: x.size), reverse=True)[0]
        if box_width:
            p = box_width      # perceived width, in pixels
            w = BALLOON_WIDTH                # approx. actual width, in meters (pre-computed)
            f = FOCAL_LENGTH        # camera focal length, in pixels (pre-computed)
            d = f * w / p
            cv.putText(frame, "Distance=%.3fm" % d, (5,50), font, 1, (0, 0, 255), 2)
    """

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles
    blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    centroids = cv.drawKeypoints(frame, keypoints, frame, (0, 0, 255), flags=0)

    #plt.imshow(blobs)
    #plt.imshow(centroids)
    #cv.imshow('Green', green)
    cv.imshow('Blobs', blobs)
    #cv.imshow('Mask', mask)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
