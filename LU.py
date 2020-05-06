# ─── LIBRARIES ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os
import PIL
import csv

# ─── CLEAN CONSOLE ──────────────────────────────────────────────────────────────
os.system("cls")
# ─── IMPORT LANDUSE COLORS ──────────────────────────────────────────────────────
RgbCodes = dict()
RgbCodes[0] = np.array([255, 255, 255])
csvfile = open("data/RgbCodes.txt", "rt")
reader = csv.reader(csvfile, delimiter="\t", skipinitialspace=True)
for num in reader:
    RgbCodes[int(num[0])] = np.array([int(num[1]), int(num[2]), int(num[3])])
csvfile.close()
# ─── IMPORT IMAGE ───────────────────────────────────────────────────────────────
img = PIL.Image.open("data/UL_Crop.png")
imgRaster = np.asarray(img)
classifiedRaster = np.tile(0, (img.height, img.width))
img.close()
# ─── CLASSIFYING THE IMAGE ──────────────────────────────────────────────────────
for row in trange(0, imgRaster.shape[0], desc="Reclassifying the image", unit=" rows"):
    for col in range(0, imgRaster.shape[1]):
        for id, color in RgbCodes.items():
            if np.array_equal(imgRaster[row][col], color):
                classifiedRaster[row][col] = id
                break
# ─── CREATING RANGE MATRIX TEMPLATE ─────────────────────────────────────────────
rangeMatrix = np.tile(0, (8, 17, 17))
for d in range(0, 8):
    rangeMatrix[d, 8 - d - 1, 8 - d - 1 : 8 + d + 2] = 1
    rangeMatrix[d, 8 + d + 1, 8 - d - 1 : 8 + d + 2] = 1
    rangeMatrix[d, 8 - d - 1 : 8 + d + 2, 8 - d - 1] = 1
    rangeMatrix[d, 8 - d - 1 : 8 + d + 2, 8 + d + 1] = 1
# ─── CALCULATING AVERAGE LANDUSE COVER ──────────────────────────────────────────
KAvgTotal = np.empty([8])
for k in range(0, 8):
    KAvgTotal[k] = np.count_nonzero(classifiedRaster == k + 1) / classifiedRaster.size
# ─── CALCULATING KD FOR EACH CELL ──────────────────────────────────────────────
IKDLRes = np.zeros((classifiedRaster.shape[0], classifiedRaster.shape[1], 8, 8))
lCells = classifiedRaster != 0
for row in trange(
    0, classifiedRaster.shape[0], desc="Calculating IKDL matrixes", unit=" rows"
):
    for col in range(0, classifiedRaster.shape[1]):
        if lCells[row, col]:
            for d in range(0, 8):
                baseImg = classifiedRaster[
                    max(0, row - d - 1) : 1
                    + min(row + d + 1, classifiedRaster.shape[0]),
                    max(0, col - d - 1) : 1
                    + min(col + d + 1, classifiedRaster.shape[1]),
                ]
                mask = rangeMatrix[
                    d,
                    max(8 - row, 8 - d - 1) : 9
                    + min(d + 1, classifiedRaster.shape[0] - row - 1),
                    max(8 - col, 8 - d - 1) : 9
                    + min(d + 1, classifiedRaster.shape[1] - col - 1),
                ]
                maskedRaster = np.multiply(baseImg, mask)
                for k in range(0, 8):
                    kCount = np.count_nonzero(maskedRaster == k + 1)
                    dTotal = np.count_nonzero(mask == 1)
                    KDAvg = kCount / dTotal
                    IKDLRes[row, col, k, d] = KDAvg / KAvgTotal[k]
# ─── CALCULATING AVERAGE EF FOR EACH L PER K ────────────────────────────────────
sumMatrix = np.zeros([6, 8, 8])
for l in range(0, 6):
    cellPos = classifiedRaster == l + 1
    cellCount = 0
    for row in range(0, classifiedRaster.shape[0]):
        for col in range(0, classifiedRaster.shape[1]):
            if cellPos[row, col]:
                cellCount += 1
                sumMatrix[l] += IKDLRes[row, col, :, :]
    sumMatrix[l] /= cellCount
# ─── PLOTTING KD FOR EACH L ─────────────────────────────────────────────────────
if not os.path.exists("data/outputs"):
    os.makedirs("data/outputs")
for l in trange(0, 6, desc="Printing plots", unit=" Plot"):
    for k in range(0, 8):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
        ax.plot(range(1, 9), sumMatrix[l, k, :])
        ax.set_title("Plot for L={} and K={}".format(l + 1, k + 1))
        ax.set_xlabel("Distance in pixels (d)")
        ax.set_ylabel(r"Average Enrichment Factor ($\overline{EF}$)")
        for i, txt in enumerate(sumMatrix[l, k, :]):
            ax.annotate(np.round(txt, 4), (i + 1, sumMatrix[l, k, i]))
        fig.savefig("data/outputs/fig#{}{}.png".format(l + 1, k + 1))
        plt.close(fig)

print("Output figures are saved in data/outputs directory")
