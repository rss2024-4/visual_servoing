import cv2
import imutils
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	MIN_MATCH = 10 # Adjust this value as needed
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	kp1, des1 = sift.detectAndCompute(template,None)
	kp2, des2 = sift.detectAndCompute(img,None)

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		h, w = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		########## YOUR CODE STARTS HERE ##########

		# using mask from ransac data to do lmed
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 5.0, mask)
		matchesMask = mask.ravel().tolist()

		dst = cv2.perspectiveTransform(pts, M)

		# prints visual
		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					 	   singlePointColor = None,
						   matchesMask = matchesMask, # draw only inliers
						   flags = 2)
		img3 = cv2.drawMatches(template,kp1,img,kp2,good,None,**draw_params)
		# image_print(img3)

		x_min = y_min = float("inf")
		x_max = y_max = 0

		for [[x,y]] in np.int32(dst):
			if x < x_min: x_min = x
			if x > x_max: x_max = x
			if y < y_min: y_min = y
			if y > y_max: y_max = y

		########### YOUR CODE ENDS HERE ###########

		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print(f"[SIFT] not enough matches; matches: ", len(good))

		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))
		return cd_template_matching(img, template)

def cd_template_matching(img, template):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	template_canny = cv2.Canny(template, 50, 200)

	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)

	# Get dimensions of template
	(img_height, img_width) = img_canny.shape[:2]

	# Keep track of best-fit match
	best_match = 0
	bounding_box = ((),())

	# sharpen_filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
	# img_canny = cv2.filter2D(img_canny, -1, sharpen_filter)
	# img_canny = cv2.GaussianBlur(img_canny, (5,5),0)

	# Loop over different scales of image
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.

		# resized_template = cv2.filter2D(resized_template, -1, sharpen_filter)
		
		# mask = np.zeros((h,w), dtype="uint8")
		resized_template = cv2.GaussianBlur(resized_template, (3,5),0)
		res = cv2.matchTemplate(img_canny, resized_template, cv2.TM_CCORR_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
		# if min_val < best_match:
			# best_match = min_val
			# bounding_box = ((min_loc[0], min_loc[1]), (min_loc[0]+w, min_loc[1]+h))
		if max_val > best_match:
			best_match = max_val

			bounding_box = ((max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h))
			
		########### YOUR CODE ENDS HERE ###########

	# print(bounding_box)
	print(best_match)
	# output = cv2.rectangle(img, bounding_box[0], bounding_box[1], (255,0,0), 3)
	# image_print(img)
	# image_print(test)
	return bounding_box
