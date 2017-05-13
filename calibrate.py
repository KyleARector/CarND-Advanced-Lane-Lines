# Save accumulated data in json or something
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
img = cv2.imread("camera_cal/calibration20.jpg")


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
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


objpoints = []
imgpoints = []
nx = 9
ny = 6

temp_objpoints = np.zeros((ny*nx, 3), np.float32)
temp_objpoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret:
    imgpoints.append(corners)
    objpoints.append(temp_objpoints)
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

undistorted = cal_undistort(img, objpoints, imgpoints)
display_comparison(img, undistorted)
