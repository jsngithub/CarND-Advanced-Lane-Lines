import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# get a list of files in the calibration folder
filenames = os.listdir('camera_cal/')

# define the number of corners in the chessboard
nx = 9
ny = 6

# create object points
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

objpoints = []
imgpoints = []

for idx, filename in enumerate(filenames):
    # read image
    img = cv2.imread('camera_cal/' + filenames[idx])

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # if found, draw corners
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
# get image size        
img = cv2.imread('camera_cal/' + filenames[11])
img_size = (img.shape[1], img.shape[0])

# gat mtx, dist values
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# save example image for use in the writeup
udst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('examples/0_original_image.jpg', img)
cv2.imwrite('examples/0_undistorted_example.jpg', udst)

# save for future use
import pickle
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('dist_pickle.p', 'wb'))

# display original v. undistorted images side by side
for filename in filenames:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    img = cv2.imread('camera_cal/' + filename)
    ax1.imshow(img)
    ax1.set_title('Original Image: ' + filename, fontsize=30)
    ax2.imshow(cv2.undistort(img, mtx, dist, None, mtx))
    ax2.set_title('Undistorted Image', fontsize=30)
    