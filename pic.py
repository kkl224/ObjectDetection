import cv2
import numpy as np
while(1):
    image = cv2.imread("/Users/karenli/Documents/ObjectDetection/Image.png")                        
    cv2.imshow("Original", image)                           
    blur = cv2.GaussianBlur(image, (21,21), 0)   
    cv2.imshow("Blur", blur)           
    hsvframe = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsvframe)
    lower = np.array([29, 40, 10])
    upper = np.array([60, 255, 255])
    mask = cv2.inRange(hsvframe, lower, upper)
    cv2.imshow('Mask1', mask)
    mask = cv2.erode(mask, None, iterations=6)
    cv2.imshow('Mask2', mask)
    mask = cv2.dilate(mask, None, iterations=8)
    cv2.imshow('Mask3', mask)
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv2.destroyAllWindows()
        break            