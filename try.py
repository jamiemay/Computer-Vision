#! -*- coding:utf-8 -*-
import sys
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(1000000)
cnt = 0

def main():
    img = cv2.imread(r'originals/001.jpg')
    img2 = cv2.imread(r'labelled/001.jpg')
    rows, cols= img.shape[0], img.shape[1]
    img = cv2.resize(img, (int(cols / 4), int(rows / 4)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    # Finding sure foreground area
    sure_fg = cv2.erode(opening, kernel, iterations=1)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero，分水岭算法识别每个细胞的边界
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    # plt.figure(1)
    # print gray
    # plt.imshow(gray)
    # plt.show()
    # #  给每个细胞标上不同标记
    # data = sure_fg.tolist()
    # data = [[int(i) for i in row] for row in data]
    # ass(data)
    # data = np.array(data)
    # max_index = data.max()
    # data[markers == -1] = max_index + 1
    # data = data.tolist()
    # fill(data, max_index)
    # data = np.array(data)
    # 统计标记的数量即细胞数
    numbers = markers.max()
    # index_var = []
    # for i in range(256, max_index + 1):
    #     if data[data == i].shape[0] != 0:
    #         index_var.append((gray[data == i].var(), i))
    # numbers = len(index_var)
    # 如果细胞的像素方差较大则视为有病。在原图中上色并统计个数
    print (markers)
    for i in range(2, numbers + 1):
        if gray[markers==i] != []:
            max_pix = gray[markers==i].max()
            min_pix =gray[markers==i].min()
            if max_pix - min_pix >1:
                img[markers == i] = [0, 255, 0]
    # sttresh = np.array([i[0] for i in index_var]).max()
    # count = 0
    # for i in index_var:
    #     if i[0] > sttresh * 0.4:
    #         count += 1
    #         img[data == i[1]] = [0, 255, 0]
    # print 'total:' + str(numbers)
    # print 'sick:' + str(count)
    # print 'porpotion:'
    # print float(count) / float(numbers)
    # img[markers == -1] = [255, 0, 0]
    # (r, g, b) = cv2.split(img)
    # img = cv2.merge([b, g, r])
    # (r, g, b) = cv2.split(img2)
    # img2 = cv2.merge([b, g, r])
    # plt.figure(2)
    # plt.subplot(1,2,1)
    # plt.imshow(img2)
    # plt.subplot(1,2,2)
    # plt.imshow(img)
    # plt.show()




if __name__ == '__main__':
    main()