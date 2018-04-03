import cv2
import numpy as np

from matplotlib import  pyplot as plt
def count(sure_fg):
       pass

def main():
        cv2.namedWindow("win")
        img = cv2.imread(r'labelled/003.jpg')
        rows, cols = img.shape[0], img.shape[1]
        img = cv2.resize(img, (int(cols / 2), int(rows / 2)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        print('hi')

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        sure_fg = cv2.erode(opening, kernel, iterations=2)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)

        img[markers == -1] = [0, 255, 0]

        cv2.imshow("Results", img)
        sure_fg = cv2.erode(opening, kernel, iterations=3)
        cv2.imshow("Ex",sure_fg)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()