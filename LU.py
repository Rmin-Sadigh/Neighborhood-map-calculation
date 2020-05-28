# ─── LIBRARIES ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os
import PIL
import csv

# ─── CLEAN CONSOLE ──────────────────────────────────────────────────────────────
os.system("cls")

#
# ────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: R E C L A S S I F I C A T I O N :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────
#

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
classifiedRaster = np.zeros((img.height, img.width), dtype=np.int8)
img.close()
# ─── CLASSIFYING THE IMAGE ──────────────────────────────────────────────────────
for row in trange(imgRaster.shape[0], desc="Reclassifying the image", unit=" rows"):
    for col in range(imgRaster.shape[1]):
        for id, color in RgbCodes.items():
            if np.array_equal(imgRaster[row][col], color):
                classifiedRaster[row][col] = id
                break

#
# ────────────────────────────────────────────────────────────────────────────── II ──────────
#   :::::: P A R T   - A -   O F   P R O J E C T : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────
#

# ─── CREATING RANGE MATRIX TEMPLATE ─────────────────────────────────────────────
rangeMatrix = np.zeros((8, 17, 17), dtype=np.int8)
for d in range(8):
    rangeMatrix[d, 8 - d - 1, 8 - d - 1 : 8 + d + 2] = 1
    rangeMatrix[d, 8 + d + 1, 8 - d - 1 : 8 + d + 2] = 1
    rangeMatrix[d, 8 - d - 1 : 8 + d + 2, 8 - d - 1] = 1
    rangeMatrix[d, 8 - d - 1 : 8 + d + 2, 8 + d + 1] = 1
# ─── CALCULATING AVERAGE LANDUSE COVER ──────────────────────────────────────────
KAvgTotal = np.empty([8])
for k in range(8):
    KAvgTotal[k] = np.count_nonzero(classifiedRaster == k + 1) / classifiedRaster.size
# ─── CALCULATING KD FOR EACH CELL ──────────────────────────────────────────────
IKDLRes = np.zeros(
    (classifiedRaster.shape[0], classifiedRaster.shape[1], 8, 8), dtype=np.float32
)
lCells = classifiedRaster != 0
for row in trange(
    classifiedRaster.shape[0], desc="Calculating IKDL matrixes", unit=" rows"
):
    for col in range(classifiedRaster.shape[1]):
        if lCells[row, col]:
            for d in range(8):
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
                dTotal = np.count_nonzero(mask == 1)
                for k in range(8):
                    kCount = np.count_nonzero(maskedRaster == k + 1)
                    KDAvg = kCount / dTotal
                    IKDLRes[row, col, k, d] = KDAvg / KAvgTotal[k]
# ─── CALCULATING AVERAGE EF FOR EACH L PER K ────────────────────────────────────
sumMatrix = np.zeros([6, 8, 8], dtype=np.float64)
for row in range(classifiedRaster.shape[0]):
    for col in range(classifiedRaster.shape[1]):
        if classifiedRaster[row, col] in range(1,7):
            sumMatrix[classifiedRaster[row, col] - 1] += IKDLRes[row, col]
for l in range(6):
    sumMatrix[l] /= np.count_nonzero(classifiedRaster == l + 1)
# ─── PLOTTING KD FOR EACH L ─────────────────────────────────────────────────────
weightMatrix = np.zeros([6, 8, 8], dtype=np.float64)
if not os.path.exists("data/outputs"):
    os.makedirs("data/outputs")
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
for l in trange(6, desc="Printing plots", unit=" Plot"):
    for k in range(8):
        data = np.vstack((range(1, 9), sumMatrix[l, k]))
        zeroValues = [i for (i, v) in enumerate(data[1]) if v == 0]
        data = np.delete(data, zeroValues, 1)
        if data.shape[1] != 0:
            ax.set_xlabel("Distance in pixels (d)")
            ax.set_ylabel(
                r"Log of Average Enrichment Factor ($\log_2 {\overline{EF}}$)"
            )
            data[1] = np.log(data[1]) / np.log(2)
            for (i, v) in enumerate(data[1]):
                if v != 0:
                    weightMatrix[l, k, int(data[0, i]) - 1] = v
            ax.plot(data[0], data[1])
            ax.set_title("Plot for L={} and K={}".format(l + 1, k + 1))
            for i, txt in enumerate(data[1]):
                ax.annotate(np.round(txt, 4), (data[0, i], data[1, i]))
            fig.savefig("data/outputs/fig#{}{}.png".format(l + 1, k + 1))
            ax.clear()
plt.close(fig)

#
# ────────────────────────────────────────────────────────────────────────────── III ──────────
#   :::::: P A R T   - B -   O F   P R O J E C T : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────
#

# ─── NEIGHBORHOOD MAP ───────────────────────────────────────────────────────────
nbhRaster = np.zeros((img.height, img.width), dtype=np.float32)
for l in range(6):
    nbhRaster.fill(0)
    lCells = classifiedRaster == l + 1
    # ─── MAP CALCULATION ────────────────────────────────────────────────────────────
    for row in trange(
        nbhRaster.shape[0],
        desc="Calculating Neighborhood Map for landuse {}".format(l + 1),
        unit=" Row",
    ):
        for col in range(nbhRaster.shape[1]):
            if lCells[row, col]:
                for d in range(8):
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
                    for k in range(8):
                        kCount = np.count_nonzero(maskedRaster == k + 1)
                        nbhRaster[row, col] += kCount * weightMatrix[l, k, d]
    # ─── MAP NORMALZATION ───────────────────────────────────────────────────────────
    a = (np.max(nbhRaster) - np.min(nbhRaster)) / 0.8
    b = a / 10 - np.min(nbhRaster)
    for row in trange(
        nbhRaster.shape[0],
        desc="Normalizing Neighborhood Map for landuse {}".format(l + 1),
        unit=" Row",
    ):
        for col in range(nbhRaster.shape[1]):
            nbhRaster[row, col] = min(max(0, (nbhRaster[row, col] + b) / a), 1)
    # ─── PLOTTING MAP ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    ax.set_title("Neighborhood map for L={}".format(l + 1))
    img = ax.imshow(nbhRaster)
    cbar = ax.figure.colorbar(ax.imshow(nbhRaster))
    plt.axis("off")
    fig.savefig("data/outputs/N-map#{}".format(l + 1))
    plt.close(fig)
# ─── FINAL MESSAGE ──────────────────────────────────────────────────────────────
print("Output figures are saved in data/outputs directory")
