import sys
import os
from skimage.morphology import skeletonize,, thin
from skimage.transform import probabilistic_hough_line
from skimage.util import invert
from skimage import data, img_as_bool, io, color
import matplotlib.pyplot as plt

mainDir = os.path.dirname(__file__)

imgPath = sys.argv[1]

image = img_as_bool(
    color.rgb2gray(color.rgba2rgb(io.imread(mainDir + "input/" + imgPath)))
)

skeleton = skeletonize(image)
skeleton_lee = skeletonize(image, method="lee")
thinned = thin(image)

threshold = 15
line_length = 130
line_gap = 50

angle = probabilistic_hough_line(
    skeleton, threshold=threshold, line_length=line_length, line_gap=line_gap
)
angle_lee = probabilistic_hough_line(
    skeleton_lee, threshold=threshold, line_length=line_length, line_gap=line_gap
)
angle_thinned = probabilistic_hough_line(
    thinned, threshold=threshold, line_length=line_length, line_gap=line_gap
)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("original")
ax[0].axis("off")

ax[1].imshow(skeleton, cmap=plt.cm.gray)
for line in angle:
    p0, p1 = line
    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[1].set_title("skeletonize")
ax[1].axis("off")

ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
for line in angle_lee:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_title("skeletonize (Lee 94)")
ax[2].axis("off")

ax[3].imshow(thinned, cmap=plt.cm.gray)
for line in angle_thinned:
    p0, p1 = line
    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[3].set_title("thinned")
ax[3].axis("off")

fig.tight_layout()
plt.savefig(mainDir + "output/" + imgPath)