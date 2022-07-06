import cv2 as cv

videoCapture = cv.VideoCapture(0)

while(1):

    # Reading the video from the webcam in image frames
    ret, frame = videoCapture.read()
    if not ret: 
        break

    # get vcap property 
    width  = int(videoCapture.get(3))   # int(float `width`)
    height = int(videoCapture.get(4))  # int(float `height`)

    print(width, height)

    if cv.waitKey(33) == 27:
        # De-allocate any associated memory usage
        cv.destroyAllWindows()
        videoCapture.release()
        break