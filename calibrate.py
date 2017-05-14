'''Exact fit lines:
pts = np.array([[593, 450],
                [687, 450],
                [1125, img.shape[0]],
                [190, img.shape[0]]],
               np.int32)'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_images(file_path="camera_cal"):
    images = []
    for filename in os.listdir(file_path):
        image = cv2.imread(file_path + "/" + filename)
        images.append(image)
    return images


def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def display_comparison(img, undistorted_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=50)
    ax2.imshow(undistorted_img)
    ax2.set_title("Altered Image", fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def warp(img):
    offset = 200
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[190, 720],
                      [1125, 720],
                      [593, 450],
                      [687, 450]])
    dst = np.float32([[300, img_size[1]],
                      [980, img_size[1]],
                      [300, 0],
                      [980, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


# Load images for calibration
images = load_images()
# Declare lists for the object and image points
objpoints = []
imgpoints = []
# Define number of x and y points to detect on chessboard
nx = 9
ny = 6

# Run calibration against test images
# Should be method
for image in images:
    temp_objpoints = np.zeros((ny*nx, 3), np.float32)
    temp_objpoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret:
        imgpoints.append(corners)
        objpoints.append(temp_objpoints)
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)

    # undistorted = cal_undistort(image, objpoints, imgpoints)
    # display_comparison(image, undistorted)


img = mpimg.imread("test_images/test6.jpg")

undistorted = cal_undistort(img, objpoints, imgpoints)
# display_comparison(img, undistorted)

# (190, 720), (593, 450)
# Slope -0.66997519
# Intercept 847.295
# (1125, 720), (687, 450)
# Slope 0.616438
# Intercept 26.5
pts = np.array([[593, 450],
                [687, 450],
                [1125, img.shape[0]],
                [190, img.shape[0]]],
               np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255))

plt.imshow(img)
plt.show()

warped = warp(undistorted)
display_comparison(undistorted, warped)
# Perform Sobel


# Distort to birds eye view

# Find lane line start points

# Sliding image search

# Reapply lines to original image
