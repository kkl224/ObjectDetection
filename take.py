import cv2
videoCapture = cv2.VideoCapture(0)
while(1):
    ret, image = videoCapture.read()
    if ret:
        cv2.imshow("Image", image)
        cv2.imwrite("Image.png", image)
        if cv2.waitKey(33) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            break            