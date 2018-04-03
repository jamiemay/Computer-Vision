import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import sys
sys.setrecursionlimit(1000000)
cnt = 0

#不明白
def judge(index, x, y, data):
    #Marking the pixels.
    #
    if data[x][y] != 255:
        return
    else:
        data[x][y] = index
        if x + 1 < len(data):
            judge(index, x + 1, y, data)
        if y + 1 < len(data[0]):
            judge(index, x, y + 1, data)
        if x - 1 >= 0:
            judge(index, x - 1, y, data)
        if y - 1 >= 0:
            judge(index, x, y - 1, data)

def ass(data):
    # Marking all the cells in the picture, each cell has 1 marker.
    m, n = len(data), len(data[0])
    index = 256
    for i in range(m):
        for j in range(n):
            judge(index, i, j, data)
            index += 1

#不明白
def fill(data, max_index):
    # Label all the cells.
    stack = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] >= 256 and data[i][j] <= max_index:
                stack.append((data[i][j], i, j))
    while len(stack) != 0:
        elem = stack.pop()
        if elem[1] + 1 < len(data):
            if data[elem[1] + 1][elem[2]] < 256:
                data[elem[1] + 1][elem[2]] = elem[0]
                stack.append((elem[0], elem[1] + 1, elem[2]))
        if elem[1] - 1 >= 0:
            if data[elem[1] - 1][elem[2]] < 256:
                data[elem[1] - 1][elem[2]] = elem[0]
                stack.append((elem[0], elem[1] - 1, elem[2]))
        if elem[2] + 1 < len(data[0]):
            if data[elem[1]][elem[2] + 1] < 256:
                data[elem[1]][elem[2] + 1] = elem[0]
                stack.append((elem[0], elem[1], elem[2] + 1))
        if elem[2] - 1 >= 0:
            if data[elem[1]][elem[2] - 1] < 256:
                data[elem[1]][elem[2] - 1] = elem[0]
                stack.append((elem[0], elem[1], elem[2] - 1))

def main():
    cv2.namedWindow("Result")
    #Change the value of the jpg file to read different image.
    img = cv2.imread(r'originals/002.jpg')
    rows, cols= img.shape[0], img.shape[1]
    img = cv2.resize(img, (int(cols / 2), int(rows / 2)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Noise Removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    #Finding background
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    #Finding Foreground
    sure_fg = cv2.erode(opening, kernel, iterations=1)

    #Finding Unknown Region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker Labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    #Add one to all labels so that the determined background is not 0, but 1.
    markers = markers + 1

    #Marking unknown region to 0, watershed to mark the edges of each cell.
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    
	#不明白
    #Differentiate each cell with a position in an array.
    data = sure_fg.tolist()
    data = [[int(i) for i in row] for row in data]
    ass(data)
    data = np.array(data)
    max_index = data.max()
    data[markers == -1] = max_index + 1
    data = data.tolist()
    fill(data, max_index)
    data = np.array(data)
    # Statistical markers = Number of Cells.
    index_var = []
    for i in range(256, max_index + 1):
        if data[data == i].shape[0] != 0:
            index_var.append((gray[data == i].var(), i))
    numbers = len(index_var)
    
    # Detecting the cells with higher variance values to find infected cells and marking them.
    sttresh = np.array([i[0] for i in index_var]).max()
    count = 0
    for i in index_var:
        if i[0] > sttresh * 0.4:
            count += 1
            img[data == i[1]] = [0, 255, 0]
            
    #想把这个放进results.py        
    print ('Total Number Of Cells:' + str(numbers))
    print ('Number of Infected Cells:' + str(count))
    print ('Proportion of Infection:')
    print (float(count) / float(numbers))
    img[markers == -1] = [255, 0, 0]
    cv2.imshow("Malaria Sample", img)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()