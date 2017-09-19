import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.recent_xfitted_left = [] 
        self.recent_xfitted_right = [] 
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        #polynomial coefficients averaged over the last n iterations
        self.best_left_fit = None  
        self.best_right_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_left_fit = [np.array([False])]
        self.current_right_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature_left = None 
        self.radius_of_curvature_right = None
        self.radius_of_curvatore = None
        self.offset = None
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.leftx = None  
        self.rightx = None  
        
        #y values for detected line pixels
        self.lefty = None
        self.righty = None
        
        self.cur_img = None
        self.warped = None
        self.color_mask = None
        self.out_img = None
        self.out_img_warp = None
        
        # used for debug
        self.mean_distance = []
        self.std_distance = []
        self.diff_radius = []
        self.bad_count = 0

    def find_pixels_window(self, binary, nwindows=13, margin=100, minpix=50):
        # Set height of windows
        window_height = np.int(binary.shape[0]/nwindows)
    
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # define starting x coordinates
        offset = 320
        leftx_base = offset        # hard coded offset
        rightx_base = 1280-offset  # hard coded offset
    
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary.shape[0] - (window+1)*window_height
            win_y_high = binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        
        # color the mask
        for i in range(len(self.leftx)):
            self.color_mask[self.lefty[i]][self.leftx[i]] = (255, 0, 0)
            
        for i in range(len(self.rightx)):
            self.color_mask[self.righty[i]][self.rightx[i]] = (255, 0, 0)            

    def find_pixels(self, binary, margin=50):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.best_left_fit[0]*(nonzeroy**2) + self.best_left_fit[1]*nonzeroy + self.best_left_fit[2] - margin)) & (nonzerox < (self.best_left_fit[0]*(nonzeroy**2) + self.best_left_fit[1]*nonzeroy + self.best_left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.best_right_fit[0]*(nonzeroy**2) + self.best_right_fit[1]*nonzeroy + self.best_right_fit[2] - margin)) & (nonzerox < (self.best_right_fit[0]*(nonzeroy**2) + self.best_right_fit[1]*nonzeroy + self.best_right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        
        # color the mask
        for i in range(len(self.leftx)):
            self.color_mask[self.lefty[i]][self.leftx[i]] = (255, 0, 0)
            
        for i in range(len(self.rightx)):
            self.color_mask[self.righty[i]][self.rightx[i]] = (255, 0, 0)  
        
    def calculate_curvature(self, img_size=(720, 1280, 3)):
        ploty = np.linspace(0, img_size[0]-1, img_size[0])

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)
        y_eval = np.max(ploty)

        # Calculate the new radii of curvature
        left = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
                
        # Calculate the offset
        left_lane_position = left_fit_cr[0]*(719*ym_per_pix)**2 + left_fit_cr[1]*(719*ym_per_pix) + left_fit_cr[2]
        right_lane_position = right_fit_cr[0]*(719*ym_per_pix)**2 + right_fit_cr[1]*(719*ym_per_pix) + right_fit_cr[2]
        center_of_lane = right_lane_position - left_lane_position
        center_of_car = (img_size[1] / 2) * xm_per_pix
        offset = center_of_car - center_of_lane
        
        return left, right, offset
        
    def overlay_lanes(self, Minv, img_size=(720, 1280, 3)):
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_size[0]-1, img_size[0])
        left_fitx = self.best_left_fit[0]*ploty**2 + self.best_left_fit[1]*ploty + self.best_left_fit[2]
        right_fitx = self.best_right_fit[0]*ploty**2 + self.best_right_fit[1]*ploty + self.best_right_fit[2]            
            
        # Create an image to draw the lines on
        warp_zero = np.zeros(img_size[:2]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, img_size[:2][::-1])
        #self.newwarp = newwarp
        
        # Combine the result with the original image
        self.out_img = cv2.addWeighted(self.cur_img, 1, newwarp, 0.3, 0)
        self.out_img_warp = cv2.addWeighted(self.color_mask, 1, color_warp, 0.3, 0)
                
    def detect_lanes(self, img, mask, warped, Minv, nwindows=13, margin=100, minpix=50, img_size = (720, 1280, 3), navg=4, debug=False):
        self.cur_img = img
        self.warped = warped
        self.color_mask = np.dstack((mask, mask, mask))
        binary = np.uint8(mask / 255)
        
        # Use the windowed search or look around previous detection
        if (self.detected):
            self.find_pixels(binary, margin=margin/2)
        else:
            self.find_pixels_window(binary, nwindows=nwindows, margin=margin, minpix=minpix)
            
        # Fit a second order polynomial to each
        self.current_left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.current_right_fit = np.polyfit(self.righty, self.rightx, 2)
        
        # Define conversions in x pixels space to meters
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Sanity Check or reset
        left_curve, right_curve, offset = self.calculate_curvature()
        ploty = np.linspace(0, img_size[0]-1, img_size[0])
        left_fitx = self.current_left_fit[0]*ploty**2 + self.current_left_fit[1]*ploty + self.current_left_fit[2]
        right_fitx = self.current_right_fit[0]*ploty**2 + self.current_right_fit[1]*ploty + self.current_right_fit[2]       
        distance = (right_fitx - left_fitx) * xm_per_pix
        
        if (debug):
            print ("Mean Distance: ", np.mean(distance))
            print ("Std Distance: ", np.std(distance))
            
        if (np.mean(distance) > 4.6 or np.mean(distance) < 2.8):
            self.detected = False
        elif (np.std(distance) > 0.3):
            self.detected = False
        elif ((self.current_left_fit[0] * self.current_right_fit[0]) < 0):
            self.detected = False
        else:
            self.detected = True
                          
        if (self.detected):
            self.radius_of_curvature_left = left_curve
            self.radius_of_curvature_right = right_curve
            self.radius_of_curvature = (left_curve + right_curve)/2
            self.offset = offset
            self.bad_count = 0
            
            if (len(self.recent_xfitted_left)>=navg):
                self.recent_xfitted_left[0:-2] = self.recent_xfitted_left[1:-1]
                self.recent_xfitted_left[-1] = left_fitx
                self.recent_xfitted_right[0:-2] = self.recent_xfitted_right[1:-1]
                self.recent_xfitted_right[-1] = right_fitx
            else:
                self.recent_xfitted_left.append(left_fitx)
                self.recent_xfitted_right.append(right_fitx)
            
            if (self.best_left_fit is None):
                self.best_left_fit = self.current_left_fit
                self.best_right_fit = self.current_right_fit
            else:
                self.best_left_fit = (self.current_left_fit + (navg-1)*self.best_left_fit) / navg
                self.best_right_fit = (self.current_right_fit + (navg-1)*self.best_right_fit) / navg
        else:
            self.bad_count += 1
            self.radius_of_curvature_left = -1
            self.radius_of_curvature_right = -1
            
        if (self.best_left_fit is not None):
            self.overlay_lanes(Minv)
        
        if (self.out_img is None):
            self.out_img = self.cur_img
            self.out_img_warp = self.warped
        
        cv2.putText(self.out_img, 'Radius of Curvature {:5.1f}m'.format(self.radius_of_curvature), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0])
        cv2.putText(self.out_img, 'Offset from the Center {:5.1f}m'.format(self.offset), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0])
        #cv2.putText(self.out_img, '{:5.1f}m, {:5.1f}m, {:5.1f}(mean), {:5.1f}(std), bad: {:d}'.format(self.radius_of_curvature_left, self.radius_of_curvature_right, np.mean(distance), np.std(distance), self.bad_count), (250, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0])
        self.mean_distance.append(np.mean(distance))
        self.std_distance.append(np.std(distance))  
        self.diff_radius.append(abs(left_curve-right_curve))
        
        if (debug): cv2.imwrite('output_images/4_line_detect.jpg', self.out_img_warp)
        
        #return np.vstack([self.out_img, self.out_img_warp])
        return self.out_img





   









