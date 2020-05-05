import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os
import PIL
import csv

os.system("cls")

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
img.close()

for row in trange(0, imgRaster.shape[0], desc="Reclassifying the image", unit=" rows"):
    for col in range(0, imgRaster.shape[1]):
        for id, color in RgbCodes.items():
            if np.array_equal(imgRaster[row][col], color):
                classifiedRaster[row][col] = id
                break

rangeMatrix = np.tile(0, (8, 17, 17))
for i in range(0, 8):
    rangeMatrix[i, 8 - i - 1, 8 - i - 1 : 8 + i + 2] = 1
    rangeMatrix[i, 8 + i + 1, 8 - i - 1 : 8 + i + 2] = 1
    rangeMatrix[i, 8 - i - 1 : 8 + i + 2, 8 - i - 1] = 1
    rangeMatrix[i, 8 - i - 1 : 8 + i + 2, 8 + i + 1] = 1

KAvgTotal = np.empty([8])
for i in range(0, 8):
    KAvgTotal[i] = np.count_nonzero(classifiedRaster == i + 1) / classifiedRaster.size

IKDLRes = np.zeros((classifiedRaster.shape[0], classifiedRaster.shape[1], 8, 8))
lCells = classifiedRaster != 0
for row in trange(
    0, classifiedRaster.shape[0], desc="Calculating IKDL matrixes", unit=" rows"
):
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
                    IKDLRes[row, col, k, i] = KDAvg / KAvgTotal[k]

sumMatrix = np.zeros([6, 8, 8])
for i in range(0, 6):
    cellPos = classifiedRaster == i + 1
    cellCount = 0
    for row in range(0, classifiedRaster.shape[0]):
        for col in range(0, classifiedRaster.shape[1]):
            if cellPos[row, col]:
                cellCount += 1
                sumMatrix[i] += IKDLRes[row, col, :, :]
    sumMatrix[i] /= cellCount

if not os.path.exists("data/outputs"):
    os.makedirs("data/outputs")
for l in trange(0, 6, desc="Printing plots", unit=" Plot"):
    for k in range(0, 8):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
        ax.set_title("Plot for l={} and k={}".format(l + 1, k + 1))
        ax.set_xlabel("Distance in pixels")
        ax.set_ylabel("Average Enrichment Factor")
        ax.plot(range(1, 9), sumMatrix[l, k, :])
        fig.savefig("data/outputs/fig#{}{}.png".format(l + 1, k + 1))
        plt.close(fig)

print("Output figures are saved in data/outputs directory")
