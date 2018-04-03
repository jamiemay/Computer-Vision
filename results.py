#Code is for 
#! -*- coding:utf-8 -*-
import sys
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

def main():
    #Input the images from the file you want analysed for comparison: img = Original img2 = Labelled
    cv2.namedWindow("Results")
    img = cv2.imread(r'originals/001.jpg')
    img2 = cv2.imread(r'labelled/001.jpg')

    #Converting the images into grayscale. 
    rows, cols= img.shape[0], img.shape[1]
    img = cv2.resize(img, (int(cols / 2), int(rows / 2)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Noise reduction.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Identifying the background area of the overall image.
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    # Determining each individual object. (Cells.) 
    sure_fg = cv2.erode(opening, kernel, iterations=1)

    # Remaining area is the border (seperation area) between the cells and the background.
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label the interior of each individual cell with a unique label. 
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Watershed algorithm identifies the boundaries of each cell.
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # Number of markers is the number of cells in the image. 
    numbers = markers.max()
	

    # Calculate the gray-scale variance for each cell and treat all cells that are greater than the maximum value of the (gray-scale variance * var_threshold) as diseased and marked as green.
    var_threshold = 0.2
    mess = []
    max_var = 0
    for i in range(2, numbers + 1):
        if gray[markers==i] != []:
            tmp = gray[markers==i]
            if tmp.size > 50:
                pix_var = tmp.var()
                mess.append((pix_var, i, tmp.min()))
                if pix_var > max_var:
                    max_var = pix_var
    disease = []
    for i in range(len(mess)):
        if mess[i][0] >= max_var * var_threshold:
            disease.append(mess[i][2])
            img[markers == mess[i][1]] = [0, 255, 0] #BGR
            
 	# Traverse all the cells once again to prevent any leaks. 
    # Taking the average of the darkest gray values for eatch diseased cell as described above, treat all cells that have a darker color than the average* pix_threshold as being diseased and marked in red.
    pix_threshold = 0.7
    for i in range(len(mess)):
        if mess[i][2] <= np.array(mess).mean() * pix_threshold:
            img[markers == mess[i][1]] = [0, 0, 255] #BGR
            
    # Map all borders in blue for easy viewing.
    img[markers == -1] = [255, 0, 0]#BGR


    #Show final results.
    print('Total Number Of Cells:' + str(numbers))
    (r, g, b) = cv2.split(img)
    img = cv2.merge([b, g, r])
    (r, g, b) = cv2.split(img2)
    img2 = cv2.merge([b, g, r])
    plt.figure('Result')
    plt.subplot(1,2,1)
    plt.imshow(img2)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.show()
    
    # Show grey level histogram.
    plt.figure('Gray Level Histogram')
    arr = gray.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)
    plt.show()

if __name__ == '__main__':
    main()