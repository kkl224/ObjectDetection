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
#params.blobClor = 240

# Start a while loop
while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: break

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(frame)

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles.
    blobs = cv.drawKeypoints(frame, keypoints, blank, (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('frame',blobs)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
