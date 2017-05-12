from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # Defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with
    # the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Given two points with x and y coordinates,
# return the slope of the line between them
def calc_slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)


# Given the slope of a line, and the x and y coordinates of a
# point on that line, return the y intercept for slope-intercept form
def calc_y_intercept(x, y, m):
    return y - m * x


# Given the slope and y intercept of a line, as well as a y
# value on the line, return the associated x value
def calc_x_from_line(y, m, b):
    return (y - b)/m


# Given a list of values, calculate the mean
def calc_list_avg(list):
    return sum(list)/len(list)


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    # Initialize lists to hold x and y values of the lines
    # Would like to refactor
    right_x1 = []
    right_x2 = []
    right_y1 = []
    right_y2 = []
    left_x1 = []
    left_x2 = []
    left_y1 = []
    left_y2 = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate slope to determine what side the line is on
            # Filter out approximately horizontal lines
            # If negative slope, add points to left side
            # If positive, add to right side
            if calc_slope(x1, y1, x2, y2) < -.55:
                left_x1.append(x1)
                left_x2.append(x2)
                left_y1.append(y1)
                left_y2.append(y2)
            elif calc_slope(x1, y1, x2, y2) > .55:
                right_x1.append(x1)
                right_x2.append(x2)
                right_y1.append(y1)
                right_y2.append(y2)

    # Create best fit lines based on points from either side
    # Would like to refactor
    # Left side
    l_x1 = calc_list_avg(left_x1)
    l_x2 = calc_list_avg(left_x2)
    l_y1 = calc_list_avg(left_y1)
    l_y2 = calc_list_avg(left_y2)
    l_m = calc_slope(l_x1, l_y1, l_x2, l_y2)
    l_b = calc_y_intercept(l_x1, l_y1, l_m)
    l_x_bottom = int(calc_x_from_line(img.shape[0], l_m, l_b))
    l_x_top = int(calc_x_from_line(img.shape[0]*.593, l_m, l_b))
    cv2.line(img, (l_x_bottom, img.shape[0]),
             (l_x_top, int(img.shape[0]*.593)), color, thickness)
    # Right side
    r_x1 = calc_list_avg(right_x1)
    r_x2 = calc_list_avg(right_x2)
    r_y1 = calc_list_avg(right_y1)
    r_y2 = calc_list_avg(right_y2)
    r_m = calc_slope(r_x1, r_y1, r_x2, r_y2)
    r_b = calc_y_intercept(r_x1, r_y1, r_m)
    r_x_bottom = int(calc_x_from_line(img.shape[0], r_m, r_b))
    r_x_top = int(calc_x_from_line(img.shape[0]*.593, r_m, r_b))
    cv2.line(img, (r_x_bottom, img.shape[0]),
             (r_x_top, int(img.shape[0]*.593)), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # Get the image size
    imshape = image.shape

    # Canny parameters
    low_threshold = 50
    high_threshold = 150
    blur_kernel = 5

    # Image mask parameters, as a function of image size
    vertices = np.array([[(imshape[1]*.132, imshape[0]),
                          (imshape[1]*.469, imshape[0]*.593),
                          (imshape[1]*.531, imshape[0]*.593),
                          (imshape[1]*.938, imshape[0])]],
                        dtype=np.int32)

    # Hough transform parameters
    rho = 2
    theta = np.pi/180
    hough_threshold = 15
    min_line_length = 35
    max_line_gap = 20

    # # # BEGIN PIPELINE # # #
    # Convert the image to grayscale, and apply a gaussian blur
    gs_img = grayscale(image)
    blur_img = gaussian_blur(gs_img, blur_kernel)
    # Perform Canny edge detection on the blurred grayscale
    canny_img = canny(gs_img, low_threshold, high_threshold)
    # Mask off polygonal area
    # Determine dynamically in the future?
    masked_img = region_of_interest(canny_img, vertices)
    # Retrieve lines from Hough transform
    img_lines = hough_lines(masked_img, rho, theta, hough_threshold,
                            min_line_length, max_line_gap)
    result = weighted_img(img_lines, np.copy(image))
    # # # END PIPELINE # # #
    return result


# Cycle through images, process them, and save in same directory
# Directory could be defined from cmdln or config file
in_directory = "test_images"
out_directory = in_directory + "_output/"
in_directory += "/"
for image in os.listdir(in_directory):
    output_img = process_image(mpimg.imread(in_directory + image))
    # Reorder color channels before saving
    cv2.imwrite(out_directory + image,
                cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

# Generate white line video
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
# Generate yellow line video
white_output = 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
# Generate challenge video
white_output = 'test_videos_output/challenge.mp4'
clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
