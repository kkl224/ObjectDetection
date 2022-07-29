# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
RES = (640, 480)
camera = PiCamera()
camera.resolution = RES
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=RES)

font = cv2.FONT_HERSHEY_SIMPLEX

params = cv2.SimpleBlobDetector_Params()

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

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        blurframe = cv2.GaussianBlur(image, (15,15), 0)
        hsvframe = cv2.cvtColor(blurframe, cv2.COLOR_BGR2HSV)

        lower = np.array([38, 40, 15]) 
        upper = np.array([60, 255, 255])

        mask = cv2.inRange(hsvframe, lower, upper)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=6)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if (area > 10000):
                        x, y, w, h = cv2.boundingRect(contour)
                        frame = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = "Target"
                        cv2.putText(image, text, (x, y), font, 1, (0, 255, 0))
	# show the frame
        cv2.imshow("Frame", image)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
        rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
        if cv.waitKey(33) == 27:
                break

