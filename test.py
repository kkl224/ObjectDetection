from pickle import TRUE
from turtle import Turtle
import cv2 as cv
import numpy as np

# Capturing video through webcam
videoCapture = cv.VideoCapture(1)

# Start a while loop
while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: break

    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 10

    params.filterByCircularity = True
    params.minCircularity = 0.5

    params.filterByColor = True
