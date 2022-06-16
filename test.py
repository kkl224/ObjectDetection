# Standard imports
import cv2 as cv
import numpy as np

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 20000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

params.filterByColor = True
blueLower = (35, 140, 60)
blueUpper = (255, 255, 180)

# Start a while loop
while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: break

    mask = cv.inRange(frame, blueLower, blueUpper)
    mask = cv.erode(mask, None, iterations=0)
    mask = cv.dilate(mask, None, iterations=0)
    frame = cv.bitwise_and(frame,frame,mask = mask)

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(mask)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the resulting frame
    frame = cv.bitwise_and(frame,im_with_keypoints,mask = mask)

    cv.imshow('frame',frame)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
