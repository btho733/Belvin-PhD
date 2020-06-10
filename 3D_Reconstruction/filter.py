"""
CONTRAST STABILISATION FILTER
#############################
This is a demo to show the operation of a novel 3D image filter developed to extract tissue structure from 3D stacks,
bypassing the noise created by lighting and staining differences. The artefacts appear as stripe/striations in the
orthogonal planes (planes perpendicular to the image plane).

HOW TO RUN AND INTERPRET THE OUTPUT
***********************************
Run the script and observe the images popping up. After observing one image, close it to allow the next image to pop up.
Images before filtering and those getting filtered (in multiple steps) are shown in sequence.
The original image (matplotlib image) and first order gradient image (opencv image) are displayed for each case.
First 2 images are the UNFILTERED IMAGES (See the VERTICAL STRIATIONS that need to be filtered out).
Those getting filtered (in multiple steps) follow the sequence.

Compare the final output images against the unfiltered ones to see the effect of filtering.
The tissue structure is kept intact while striation artefacts are filtered out.

DATA
*****
A 3D stack of images is loaded.
Notice that the image being displayed here is an orthogonal 2D section (X-Z plane).
Thus it is NOT a single slice (X-Y plane), but an orthogonal cross-section made from all the slices in the stack.
It is selected to perform filtering and display results.

APPLICATION & SIGNIFICANCE
**************************
The noise being removed here affects any gradient-based operation on 3D stacks negatively. For instance, it distorts
the orientation results computed using structure tensor. These effects are studied in greater detail during the
computation of atrial fiber orientation. See chapter 6 of my thesis for more info.

Author : Belvin Thomas (Part of PhD project)
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.tsa import movstat as st

img_dir = ".\\filterTestData"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*.tif')
files = glob.glob(data_path)
num = len(files)
sf = 4
data = np.ndarray((150, 180, 100))

# Loading all the images to form the 3D stack
for i in range(num):
    img = cv2.imread(files[i])
    data[:, :, i] = img[:, :, 1]
Ny, Nx, Nz = data.shape

# Slicing out one X-Z plane for filtering
yplane = np.squeeze(data[80, 50:, :])
Nx, Ny = yplane.shape

# Find x and y gradients
sobelx = cv2.Sobel(yplane, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(yplane, cv2.CV_64F, 0, 1)

# Display the Unfiltered images (First, the original image and then its gradient image)
plt.imshow(yplane, 'summer', aspect=100/130)
plt.title('Original image(Before Filtering): See the VERTICAL STRIATIONS to be filtered')
plt.show()
grad_yplane = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
maxVal = np.amax(grad_yplane)
minVal = np.amin(grad_yplane)
draw = cv2.convertScaleAbs(grad_yplane, alpha=255.0 / (maxVal - minVal), beta=-minVal * 255.0 / (maxVal - minVal))
cv2.namedWindow('Gradient of Image (Before Filtering)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gradient of Image (Before Filtering)', Nx * sf, Ny * sf)
cv2.imshow('Gradient of Image (Before Filtering)', cv2.applyColorMap(draw, 2))
cv2.waitKey(0)

# The filtering algorithm starts here
corrected_2 = 255 * np.ones([Nx, Nz])
upto = 99
im1 = np.zeros([Nx, 2])
m1 = np.zeros([Nx, 1])
m2 = np.zeros([Nx, 1])
diff = np.zeros([Nx, 1])
corrected_r = np.zeros([Nx, Nz, 8])

for num in range(1, 10):
    r = num * 2 + 1
    for column in range(upto):
        if column == 0:
            im1[:, 0] = yplane[:, column]
            im1[:, 1] = yplane[:, column + 1]
        else:
            im1[:, 0] = corrected_2[:, column]
            im1[:, 1] = yplane[:, column + 1]
        m1 = st.movmean(im1[:, 0], windowsize=r, lag='centered')
        m2 = st.movmean(im1[:, 1], windowsize=r, lag='centered')
        diff = m1 - m2
        corrected_2[:, column] = (im1[:, 1] + diff).T
    # corrected_r[:, :, num]=corrected_2
    # Find x and y gradients
    sobelx = cv2.Sobel(corrected_2[:, 1:], cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(corrected_2[:, 1:], cv2.CV_64F, 0, 1)
    # Display the Filtered images in steps
    # (At each step, display the original image first followed by its gradient image)
    plt.imshow(corrected_2[:, 1:], 'summer', aspect=100/130)
    plt.title('Original image (getting filtered in steps)')
    plt.show()
    grad_yplane = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    maxVal = np.amax(grad_yplane)
    minVal = np.amin(grad_yplane)
    draw = cv2.convertScaleAbs(grad_yplane, alpha=255.0 / (maxVal - minVal), beta=-minVal * 255.0 / (maxVal - minVal))
    cv2.namedWindow('Gradient of Image(getting filtered in steps)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gradient of Image(getting filtered in steps)', Nx * sf, Ny * sf)
    cv2.imshow('Gradient of Image(getting filtered in steps)', cv2.applyColorMap(draw, 2))
    cv2.waitKey(0)

