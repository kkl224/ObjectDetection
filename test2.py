import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

# Start a while loop
while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: break

    # Convert the imageFrame in BGR gray color space
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blur the imageFrame
    blurframe = cv.medianBlur(grayframe, 5)

    # Draw edges 
    # cannyframe = cv.Canny(blurframe,100,200)
    # cv.imshow("Canny", cannyframe)

    # Apply Hough transform on the blurred image
    circles = cv.HoughCircles(blurframe, cv.HOUGH_GRADIENT, 1.2, 100, param1=60, param2=60, minRadius=1, maxRadius=10)

    if circles is not None:
        # Draw circles that are detected.
        circles = np.uint16(np.round(circles))

        # loop over the (x, y) coordinates and radius of the circles
        for i in circles[0,:]:
            cv.circle(frame,(i[0], i[1]), i[2], (255,0,0), 2)
            cv.circle(frame,(i[0], i[1]), 2, (0,0,255), 3)
        
    # plt.imshow(frame)
    cv.imshow("Detected Ballon", frame)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break