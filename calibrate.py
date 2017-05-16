'''Exact fit lines:
pts = np.array([[593, 450],
                [687, 450],
                [1125, img.shape[0]],
                [190, img.shape[0]]],
               np.int32)'''

# (190, 720), (593, 450)
# Slope -0.66997519
# Intercept 847.295
# (1125, 720), (687, 450)
# Slope 0.616438
# Intercept 26.5

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


def undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# Displays a comparison of an image and another
# Parameter for saving the file?
def display_comparison(img, undistorted_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=50)
    ax2.imshow(undistorted_img)
    ax2.set_title("Altered Image", fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Warps an area of an image to a new perspective
# Set up for birds eye view of road in front of car
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
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


# Uses Soble and HSL color channels to find all steep gradients in
# image
# Could use other channels or the y direction
def gradient_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    combined = np.zeros_like(s_channel)
    combined[((s_channel >= s_thresh[0]) &
             (s_channel <= s_thresh[1])) |
             (sxbinary == 1)] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
    # It might be beneficial to replace this channel with something else.
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return combined


# Run calibration against test images
def calibrate_camera():
    # Load images for calibration
    images = load_images()
    # Process each image by auto-detecting the chessboard
    for image in images:
        temp_objpoints = np.zeros((ny*nx, 3), np.float32)
        temp_objpoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, append object and image points to calibration list
        if ret:
            imgpoints.append(corners)
            objpoints.append(temp_objpoints)
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)


def sliding_window_search(img):
    # Generate histogram of the bottom of the image

    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0),
                      2)
        cv2.rectangle(out_img,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 0),
                      2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their
        # mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return ploty, left_fitx, right_fitx


def draw_lines(undist, warped, ploty, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                            ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using
    # inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1],
                                                     undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()


def pipeline():
    img = cv2.imread("test_images/test6.jpg")
    # img = mpimg.imread("test_images/straight_lines1.jpg")

    # Unidstort using camera calibration data
    undistorted = undistort(img, objpoints, imgpoints)
    # display_comparison(img, undistorted)

    # Distort to birds eye view
    warped, M, Minv = warp(undistorted)

    # Grayscale and mask
    masked = gradient_thresh(warped)
    # display_comparison(warped, masked)

    # Find lane line start points and fit polynomial
    ploty, left_fitx, right_fitx = sliding_window_search(masked)

    # Reapply lines to original image
    draw_lines(undistorted, warped, ploty, left_fitx, right_fitx, Minv)


# Declare lists for the object and image points
objpoints = []
imgpoints = []
# Define number of x and y points to detect on chessboard
nx = 9
ny = 6

calibrate_camera()
pipeline()
