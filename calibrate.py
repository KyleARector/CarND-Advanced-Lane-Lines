import os
import sys
import pickle
import cv2
import numpy as np


# Load images from supplied path
# Need to check if type is valid
def load_images(file_path="camera_cal"):
    images = []
    for filename in os.listdir(file_path):
        image = cv2.imread(file_path + "/" + filename)
        images.append(image)
    return images


# Run calibration against test images
def calibrate_camera(img_folder, nx=9, ny=6):
    objpoints = []
    imgpoints = []
    # Load images for calibration
    images = load_images(img_folder)
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
    return objpoints, imgpoints


def main():
    # Get the command line arguments, if existant
    img_folder = str(sys.argv[1]) if len(sys.argv) > 1 else "camera_cal"
    out_file = str(sys.argv[2]) if len(sys.argv) > 2 else "calibration_data.p"
    # Append appropriate file type, if missing
    if out_file[-2:] != ".p":
        out_file += ".p"
    # Calibrate the camera based on supplied image folder
    objpoints, imgpoints = calibrate_camera(img_folder)
    # Package points and save for distortion correction in pipeline
    pickle.dump({"objpoints": objpoints, "imgpoints": imgpoints},
                open(out_file, "wb"))


if __name__ == "__main__":
    main()
