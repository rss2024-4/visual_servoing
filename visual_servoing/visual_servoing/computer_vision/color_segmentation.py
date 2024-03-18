import cv2
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
EROSION_FACTOR = 7
DILATION_FACTOR = 5
EROSION_KERNEL = np.ones((EROSION_FACTOR, EROSION_FACTOR), np.uint8)
DILATION_KERNEL = np.ones((DILATION_FACTOR, DILATION_FACTOR), np.uint8)

CONE_HSV = np.array([12, 253, 254])
dHSV = np.array([12, 20, 150])

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template=None):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	pre_processed = cv2.dilate(
						cv2.erode(
							hsv,
							EROSION_KERNEL, iterations=1),
					DILATION_KERNEL, iterations=1)
	mask = cv2.inRange(pre_processed, CONE_HSV-dHSV, CONE_HSV+dHSV)
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print('contours', len(contours))
	x,y,w,h = cv2.boundingRect(contours[-1])
	return (x, y), (x+w, y+h)
	# img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
	# return img

# if __name__ == '__main__':
# 	img = cv2.imread('test_images_cone/test15.jpg')
# 	image_print(cd_color_segmentation(img))
