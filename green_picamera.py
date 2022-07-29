from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
import time
#import matplotlib.pyplot as plt

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 220

params.filterByInertia = True
params.minInertiaRatio = 0.5

params.filterByArea = True              # TO-DO: retire area filtering?
params.minArea = 500
params.maxArea = 1000000000

params.filterByConvexity = True
params.minConvexity = 0.3

params.filterByCircularity = True
params.minCircularity = 0.6

params.filterByColor = True
params.blobColor = 255

# Font to write text overlay
font = cv.FONT_HERSHEY_SIMPLEX

# Constant for focal length, in pixels (must be changed per camera)
FOCAL_LENGTH = 618
BALLOON_WIDTH = 0.33

SCALE = 1.0

# start pi camera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
stream = PiRGBArray(camera)
time.sleep(0.1) # warm-up!

# For this program, we tell the Pi Camera to send us a continuous stream of frames.
# Above, we configured the camera to send us just the information we need (i.e. custom
# resolution and framerate) which will make this code faster. The rest of the algorithm
# is based on the file "green.py" found in this same directory.
for f in camera.capture_continuous(stream, format='bgr', use_video_port=True):
    # grab frame from buffer
    frame = f.array

    # blur the frame
    frame2 = cv.GaussianBlur(frame, (13,13), 0)
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

    #lower = np.array([30, 40, 90])
    #upper = np.array([70, 255, 255])

    lower = np.array([30, 40, 10])
    upper = np.array([60, 255, 255])

    # erode and dilate image
    mask = cv.inRange(frame2, lower, upper)
    mask = cv.erode(mask, None, iterations=4)
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

    stream.truncate()
    stream.seek(0)
