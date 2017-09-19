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
[image7]: ./output_images/4_lane_line_detection.jpg "Lane Line Pixel Detection and Fit"
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

I used a combination of color and gradient thresholds to generate a binary image.  It uses a yellow color mask to detect yellow lines; white color mask to detect white lines; then it uses a combination of canny and directional masks to mask out white parts of the image that is not a lane line.  Lines 40 to 63 and 73 to 117 in `advanced_lanefind.py` show these steps.  Here is an example of my output for this step.  

![Binary Threshold Image][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 65 through 67 in the file `advanced_lanefind.py`  The `warp()` function takes as inputs an image (`img`) and returns a warped image.  I chose the hardcode the source and destination points in lines 27 to 29 of `advanced_lanefind.py`:

`offset = 320`

`img_size = (720, 1280, 3)`

`src = np.float32([[203, img_size[0]], [1170, img_size[0]], [565, 460], [723, 460]])`

`dst = np.float32([[offset, img_size[0]], [img_size[1]-offset, img_size[0]],                  [offset, 0], [img_size[1]-offset, 0]])`

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203, 720      | 320, 720      | 
| 1170, 720     | 960, 720      |
| 565, 460      | 320, 0        |
| 723, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.  Here is an example of the warped binary thresholded image from the previous step:

![Warped Binary Threshold Image][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  