import cv2 as cv
import numpy as np

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: 
        break

    hsvframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower = np.array([80, 40, 40])
    upper = np.array([130, 255, 180])

    mask = cv.inRange(hsvframe, lower, upper)

    blue = cv.bitwise_and(frame, frame, mask=mask)

    print(blue.dtype, blue.shape)
    print(mask.dtype, mask.shape)

    # Apply Hough transform on the image
    # circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 100, param1=10, param2=10, minRadius=1, maxRadius=300)

    #if circles is not None:

        # Draw circles that are detected.
        #circles = np.uint16(np.round(circles))

        # loop over the (x, y) coordinates and radius of the circles
        #for i in circles[0,:]:
            #cv.circle(frame,(i[0], i[1]), i[2], (255,0,0), 2)
            #cv.circle(frame,(i[0], i[1]), 2, (0,0,255), 3)
     
    #cv.imshow('frame', blue)
    cv.imshow('frame', mask)

    # Wait for Esc key to stop
    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break
