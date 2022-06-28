#!/usr/bin/env python

"""
A module for balloon detection using OpenCV, to be used in conjunction with OpenBlimp.
"""

import cv2 as cv
import numpy as np
import time
from threading import Thread

class BalloonDetector:
    """
    A simple balloon detector.
    """

    def __init__(self, lower_bound, upper_bound, focal_length, balloon_width, video_cap_index=0, disp_mode=False):
        """
        Initializes and configures a balloon detector.

        Parameters
        ----------
            lower_bound : tuple
                Lower bounds for the balloon's color (HSV).
            upper_bound : tuple
                Upper bounds for the balloon's color (HSV).
            focal_length : float
                The camera's focal length, in pixels (use OpenCV chessboard algorithms).
            balloon_width : float
                The balloon's average width (in meters).
            video_cap_index : int, optional
                The camera's device index (default is 0).
            disp_mode : bool, optional
                If True, enables support for a feedback window.
        """

        self.video_capture = cv.VideoCapture(video_cap_index)
        self.lb = lower_bound
        self.ub = upper_bound
        self.focal_length = focal_length
        self.balloon_width = balloon_width
        self.width = int(self.video_capture.get(3))
        self.height = int(self.video_capture.get(4))
        self.capture_thread = Thread(target=self.capture_loop)
        self.is_running = False
        self.disp_mode = disp_mode
        self.last_distance = None
        self.blobs = None

    def capture_loop(self):
        lower = self.lb
        upper = self.ub
        focal_length = self.focal_length
        balloon_width = self.balloon_width

        # Setup SimpleBlobDetector parameters, optimized for balloon detection
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

        font = cv.FONT_HERSHEY_SIMPLEX

        while self.is_running:
            # capture image
            ret, frame = self.video_capture.read()

            # blur and convert image
            blurframe = cv.GaussianBlur(frame, (15,15), 0)
            hsvframe = cv.cvtColor(blurframe, cv.COLOR_BGR2HSV)

            # acquire mask
            mask = cv.inRange(hsvframe, lower, upper)
            mask = cv.erode(mask, None, iterations=3)
            mask = cv.dilate(mask, None, iterations=6)

            comb = cv.bitwise_and(frame, frame, mask=mask)

            detector = cv.SimpleBlobDetector_create(params)

            # draw bounding boxes
            p = 1.0
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv.contourArea(contour)
                if(area > 10000):
                    x, y, w, h = cv.boundingRect(contour)
                    p = w
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if self.disp_mode:
                        text = "Target"
                        cv.putText(frame, text, (x, y), font, 1, (0, 255, 0))

            # detect blobs
            keypoints = detector.detect(mask)

            if self.disp_mode:
                blobCount = len(keypoints)
                text = "Count=" + str(blobCount) 
                cv.putText(frame, text, (5,25), font, 1, (0, 0, 255), 2)
            
            if keypoints:

                # Write X position of first blob 
                #blob_x = keypoints[0].pt[0]
                #text1 = "X=" + "{:.2f}".format(blob_x )
                #cv.putText(frame, text1, (5,100), font, 1, (0, 0, 255), 2)

                # Write Y position of first blob
                #blob_y = keypoints[0].pt[1]
                #text2 = "Y=" + "{:.2f}".format(blob_y)
                #cv.putText(frame, text2, (5,150), font, 1, (0, 0, 255), 2)

                # Get distance of largest circle
                #max_circle = sorted(keypoints, key=(lambda x: x.size), reverse=True)[0]

                #p = max_circle.size      # perceived width, in pixels
                w = balloon_width                # approx. actual width, in meters (pre-computed)
                f = focal_length        # camera focal length, in pixels (pre-computed)
                d = f * w / p

                self.last_distance = d

            # finish drawing to window, if necessary
            if self.disp_mode:
                blank = np.zeros((1, 1))

                # Draw detected blobs as red circles
                blobs = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                centroids = cv.drawKeypoints(frame, keypoints, frame, (0, 0, 255), flags=0)

                self.blobs = blobs

                #plt.imshow(blobs)
                #plt.imshow(centroids)
                #cv.imshow('Green', green)
                #cv.imshow('Blobs', blobs)
                #cv.imshow('Mask', mask)

    def start(self):
        """
        Starts the balloon detection thread.
        """

        self.is_running = True
        self.capture_thread.start()
    
    def stop(self):
        """
        Stops the balloon detection thread.
        """

        self.is_running = False
        self.capture_thread.join()

    def display(self, sleep_ms=5000):
        """
        Display a feedback window, showing the tracked object, with a time delay (disp_mode must be initialized to True).

        Parameters
        ----------
        sleep_ms : int, optional
            Time delay, in milliseconds.
        """

        if self.disp_mode:
            cv.imshow('Blobs', self.blobs)
            cv.waitKey(sleep_ms)
            return True
        else:
            return False

    def distance(self):
        """
        Gets the most recently observed distance from the camera to the balloon.
        """

        return self.last_distance

if __name__ == '__main__':
    lower = (22, 40, 10)
    upper = (60, 255, 255)
    bd = BalloonDetector(lower, upper, 655, 0.31, video_cap_index=0, disp_mode=True)
    bd.start()
    print("Started.")
    time.sleep(5)
    print("Distance = %.2fm" % bd.distance())
    bd.display()
    bd.stop()
    print("Stopped.")