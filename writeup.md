## **Advanced Lane Finding Project**

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/0_original_image.jpg "Distorted"
[image2]: ./output_images/0_undistorted_example.jpg "Undistorted"
[image3]: ./output_images/1_original_image.jpg "Distorted"
[image4]: ./output_images/1_undistorted_example.jpg "Undistorted"
[image5]: ./output_images/2_threshold_binary.jpg "Threshold Binary Image"
[image6]: ./output_images/3_threshold_binary_warped.jpg "Perfective Transformed Image"
[image7]: ./output_images/4_line_detect.jpg "Curve Fit Example"
[image8]: ./output_images/5_final.jpg "Lane Line Pixel Detection and Fit"
[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the python code `calibrate_camera.py`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Distorted][image1]
![Undistorted][image2]

The calibration values are pickled for later use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied the distortion correction to one of the test images like this one:
![Distorted][image3]
![Undistorted][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  It uses a yellow color mask to detect yellow lines; white color mask to detect white lines; then it uses a combination of canny and directional masks to further refine the selection.  Lines 40 to 63 and 73 to 117 in `advanced_lanefind.py` show these steps.  Here is an example of my output for this step.  

![Binary Threshold Image][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 65 through 67 in the file `advanced_lanefind.py`  The `warp()` function takes as inputs an image (`img`) and returns a warped image.  I chose the hardcode the source and destination points in lines 27 to 29 of `advanced_lanefind.py`:

`offset = 320`

`img_size = (720, 1280, 3)`

`src = np.float32([[236, img_size[0]], [1187, img_size[0]], [580, 460], [720, 460]])`

`dst = np.float32([[offset, img_size[0]], [img_size[1]-offset, img_size[0]],                  [offset, 0], [img_size[1]-offset, 0]])`

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 236, 720      | 320, 720      | 
| 1187, 720     | 960, 720      |
| 580, 460      | 320, 0        |
| 720, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.  Here is an example of the warped binary thresholded image from the previous step:

![Warped Binary Threshold Image][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane pixels are initially detected by using a windowed search.  The windows start at the bottom of the screen at a hard coded offset and moves up the picture, shifting the search location based on the average pixel location of the previous window.  This is shown in the function `find_pixels_window` in `line.py`.

If lanes were detected in the previous frame, the pixels are detected by looking around the area where the pixels were previously detected.  This is shown in function `find_pixels` in `line.py`.

Once found, lanes are fit by a second order polynomial using `numpy.polyfit`.

Here is an example output at the end of this process:

![Curve fit example][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was calculated usin the concept [here] (https://www.intmath.com/applications-differentiation/8-radius-curvature.php) and the sample code from the lectures.  The `calculate_curvature` function in `line.py` show the implementation.

The position of the vehicle with respect to center is calculated by comparing the center of the image to the center of the two detected lanes with respect to the bottom of the screen.  The code for this is the last part of the `calculate_curvature` function in `line.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the function `overlay_lanes` in `line.py`.

The `detect_lanes` function in `line.py` also contain an algorithm to filter out frames without a good lane detection and to average the last few frames for smoothness.

![Final Output][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The yellow color threshold in HSV color space works very well in picking out the yellow color, so I used it to detect yellow lines.  The white lines were much harder to detect using color segmentation alone, so it had to be combined with edge and directional threshold to augment it.  In cases where the road color changes or the images are really bright, false detect of white occurs a lot.  This is where the pipeline may fail to detect white lines.  This portion of the pipeline may be improved by doing a histogram based calibration of the luminance of the image.

It may also be helpful to improve the window function to use a curved window rather than a boxy rectangle window.  This would help detection in cases where the road is extremely curvy.