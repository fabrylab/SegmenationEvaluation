import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops,label

im =  plt.imread("/home/user/test_orientation.png")
im = 1 - im[:,:,0]
labeled = label(im)

plt.figure();plt.imshow(labeled)
for r in regionprops(labeled):
    angle = -r.orientation
    if angle < 0:  # this is to match clickpoints eelipse angles
        angle = np.pi - np.abs(angle)
    angle *= 180 / np.pi

    plt.text(r.centroid[1],r.centroid[0], str(angle))


