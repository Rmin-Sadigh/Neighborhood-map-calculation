import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
import csv

RgbCodes = dict()
RgbCodes[0] = np.array([255, 255, 255])
csvfile = open("data/RgbCodes.txt", "rt")
reader = csv.reader(csvfile, delimiter="\t", skipinitialspace=True)
for num in reader:
    RgbCodes[int(num[0])] = np.array([int(num[1]), int(num[2]), int(num[3])])
csvfile.close()

img = PIL.Image.open("data/UL_Crop.png")
imgRaster = np.asarray(img)
classifiedRaster = np.tile(0, (img.height, img.width))

for row in range(0, imgRaster.shape[0]):
    for col in range(0, imgRaster.shape[1]):
        for id, color in RgbCodes.items():
            if np.array_equal(imgRaster[row][col], color):
                classifiedRaster[row][col] = id
                break

plt.imshow(classifiedRaster)

rangeMatrix = np.tile(0, (8, 17, 17))
for i in range(0, 8):
    rangeMatrix[i, 8 - i - 1, 8 - i - 1 : 8 + i + 2] = 1
    rangeMatrix[i, 8 + i + 1, 8 - i - 1 : 8 + i + 2] = 1
    rangeMatrix[i, 8 - i - 1 : 8 + i + 2, 8 - i - 1] = 1
    rangeMatrix[i, 8 - i - 1 : 8 + i + 2, 8 + i + 1] = 1

KAvgTotal = np.empty([8])
for i in range(0, 8):
    KAvgTotal[i] = np.count_nonzero(classifiedRaster == i + 1) / classifiedRaster.size

KLDRes = np.zeros((8, 8, 8))

lCells = classifiedRaster != 0
for row in range(0, classifiedRaster.shape[0]):
    for col in range(0, classifiedRaster.shape[1]):
        if lCells[row, col]:
            for i in range(0, 8):
                baseImg = classifiedRaster[
                    max(0, row - i - 1) : 1
                    + min(row + i + 1, classifiedRaster.shape[0]),
                    max(0, col - i - 1) : 1
                    + min(col + i + 1, classifiedRaster.shape[1]),
                ]
                mask = rangeMatrix[
                    i,
                    max(8 - row, 8 - i - 1) : 9
                    + min(i + 1, classifiedRaster.shape[0] - row - 1),
                    max(8 - col, 8 - i - 1) : 9
                    + min(i + 1, classifiedRaster.shape[1] - col - 1),
                ]
                maskedRaster = np.multiply(baseImg, mask)
                for k in range(0, 8):
                    kCount = np.count_nonzero(maskedRaster == k + 1)
                    dTotal = np.count_nonzero(mask == 1)
                    KDAvg = kCount / dTotal
                    KLDRes[i, k, classifiedRaster[row, col] - 1] = KDAvg
