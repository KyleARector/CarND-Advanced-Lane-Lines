from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os

imgpoints, objpoints = [], []
left_history, right_history = [], []
prev_fit = ()


def grayscale(img):
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def undistort(img):
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
    ax2.imshow(undistorted_img, cmap="gray")
    ax2.set_title("Altered Image", fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Warps an area of an image to a new perspective
# Set up for birds eye view of road in front of car
def warp(img):
    offset = 200
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[190, img_size[1]],
                      [1125, img_size[1]],
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


def abs_sobel_thresh(img, orient="x", thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == "y":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag_sobel = ((sobelx**2) + (sobely**2))**(0.5)
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) &
                  (absgraddir <= thresh[1])] = 1
    return binary_output


def sat_threshold(img, thresh=(0, 255)):
    sat_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    # S Channel
    S = sat_img[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


def calc_center_offset(img, left_fit, right_fit):
    # Definined by DOT typical lane width
    ft_per_pixel = 12/700
    car_center = img.shape[1]/2
    y_val = img.shape[0]
    rightx = left_fit[0] * (y_val**2) + left_fit[1] * y_val + left_fit[2]
    leftx = right_fit[0] * (y_val**2) + right_fit[1] * y_val + right_fit[2]
    lane_center = ((rightx - leftx)/2) + leftx
    offset = (car_center - lane_center) * ft_per_pixel
    return offset


def calc_curvature_radius(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    xft_per_pixel = 12/700
    yft_per_pixel = 98/720

    # Generate x and y values
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * yft_per_pixel, left_fitx * xft_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty * yft_per_pixel, right_fitx * xft_per_pixel, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * yft_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * yft_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Average of two curves for lane curvature
    avg_curverad = np.mean([left_curverad, right_curverad])

    return avg_curverad


def sliding_window_search(img):
    global prev_fit
    # Generate histogram of the bottom of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    # Parameterize?
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

    prev_fit = (left_fit, right_fit)

    return left_fit, right_fit


def margin_search(img):
    global prev_fit

    left_fit = prev_fit[0]
    right_fit = prev_fit[1]
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    prev_fit = (left_fit, right_fit)

    return left_fit, right_fit


def check_fit_history(left_fit, right_fit, frame_count=10):
    global prev_fit

    left_history.append(left_fit)
    right_history.append(right_fit)

    if len(left_history) > 1:
        if len(left_history) > frame_count:
            left_history.pop(0)
        if len(right_history) > frame_count:
            right_history.pop(0)

        # Average polynomials from last X frames
        left_fit_avg = np.average(left_history, axis=0)
        right_fit_avg = np.average(right_history, axis=0)

        # Check the deviation of the average against the new fit
        if np.absolute(left_fit_avg[0] - prev_fit[0][0]) <= .0002:
            left_prev = left_fit_avg
        else:
            left_prev = left_fit
        if np.absolute(right_fit_avg[0] - prev_fit[0][1]) <= .0002:
            right_prev = right_fit_avg
        else:
            right_prev = right_fit
    else:
        left_prev = left_fit
        right_prev = right_fit

    prev_fit = (left_prev, right_prev)

    return left_prev, right_prev


def draw_lines(undist, warped, left_fit, right_fit, Minv, video=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,
                                                 ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                            ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    color = (59, 114, 255) if video else (255, 114, 59)
    cv2.fillPoly(color_warp, np.int_([pts]), color)

    # Warp the blank back to original image space using
    # inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1],
                                                     undist.shape[0]))
    # Combine the result with the original image
    out_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return out_img


def draw_labels(img, lane_curvature, offset):
    side = "Left" if offset < 0 else "Right"
    lane_curvature = np.round(lane_curvature, 2)
    offset = np.absolute(np.round(offset, 4))
    cv2.putText(img,
                "Radius of Curvature: " + str(lane_curvature) + " feet",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.putText(img,
                "Car is " + str(offset) + " Feet " + side + " of Center",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    return img


# Process all static images in given directory
def process_images(in_directory, out_directory):
    for image in os.listdir(in_directory):
        out_img = pipeline(cv2.imread(in_directory + image), video=False)
        cv2.imwrite(out_directory + "lane_added_" + image, out_img)


def pipeline(img, video=True):
    global prev_fit
    # # # BEGIN PIPELINE # # #
    # Undistort using camera calibration data
    undistorted = undistort(img)

    gradx = abs_sobel_thresh(undistorted,
                             orient="x",
                             thresh=(120, 255))
    grady = abs_sobel_thresh(undistorted,
                             orient="y",
                             thresh=(20, 100))
    mag_binary = mag_thresh(undistorted,
                            sobel_kernel=3,
                            thresh=(30, 100))
    dir_binary = dir_threshold(undistorted,
                               sobel_kernel=3,
                               thresh=(0.7, 1.3))
    sat_binary = sat_threshold(undistorted,
                               thresh=(200, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) &
             (dir_binary == 1)) | (sat_binary == 1)] = 1

    # Distort to birds eye view
    warped, M, Minv = warp(combined)

    # Find lane line start points and fit polynomial
    # Compare fits against average of last few frames and correct
    if len(prev_fit) != 0 and video:
        left_fit, right_fit = margin_search(warped)
    else:
        left_fit, right_fit = sliding_window_search(warped)
    if video:
        left_fit, right_fit = check_fit_history(left_fit, right_fit)

    lane_curvature = calc_curvature_radius(warped, left_fit, right_fit)
    offset = calc_center_offset(warped, left_fit, right_fit)

    # Reapply lines to original image
    lined_img = draw_lines(undistorted,
                           warped,
                           left_fit,
                           right_fit,
                           Minv,
                           video)

    out_img = draw_labels(lined_img, lane_curvature, offset)
    # # # END PIPELINE # # #
    return out_img


def main():
    global imgpoints, objpoints
    in_directory = "test_images/"
    out_directory = "output_images/"
    # Load calibration data
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    imgpoints, objpoints = map(calibration_data.get,
                               ("imgpoints", "objpoints"))
    # process_images(in_directory, out_directory)
    video_output = "video_out.mp4"
    clip1 = VideoFileClip("project_video.mp4")
    video_clip = clip1.fl_image(pipeline)
    video_clip.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    main()
