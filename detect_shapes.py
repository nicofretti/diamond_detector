import argparse
import cv2
import numpy as np
import imutils


def detect_shape(contour):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    perimeter = cv2.arcLength(contour, True)
    # applying contour approximation (curve can be approximated as short line segments)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) < 1:
        return "unidentified"
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"
    elif len(approx) == 5:
        return "pentagon"
    elif len(approx) == 6:
        return "hexagon"
    return "circle"

if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(arg.parse_args())
    image = cv2.imread(args["image"])


    # Image operations
    elab = cv2.resize(image, (400, 400))
    ratio = image.shape[0] / float(elab.shape[0])
    elab = cv2.GaussianBlur(elab, (9, 9), 0)
    elab = cv2.cvtColor(elab, cv2.COLOR_BGR2GRAY)
    elab[elab <= 30] = 0
    # blur
    #elab = cv2.GaussianBlur(elab, (5, 5), 0)
    #elab = cv2.Canny(elab, 40, 150)
    #elab = cv2.Laplacian(elab, -1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    #elab = cv2.Laplacian(elab, cv2.CV_64F)
    #elab = cv2.Laplacian(elab, -1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    elab = cv2.threshold(elab, 45, 255, cv2.THRESH_TRIANGLE)[1]
    # make boders mode white
    #elab = cv2.Canny(elab, 100, 255)
    #elab[tmp == 255] = 255


    # Show the thresholded image
    cv2.imshow("Filtered", elab)
    cv2.waitKey(0)

    # Find contours
    cnts = cv2.findContours(elab.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for contour in cnts:
        # draw the poin in the center
        contour = contour.astype("float")
        contour *= ratio
        contour = contour.astype("int")
        cv2.drawContours(image, [contour], -1, (0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

