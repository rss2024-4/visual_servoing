#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocation


# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation

PTS_IMAGE_PLANE = [
	[334., 315.],
	[336., 272.],
	[334., 246.],
	[332., 228.],
	[334., 206.],
	[334., 185.],
	[334., 176.],
	[332., 168.],
	[442., 317.],
	[562., 318.],
	[321., 315.],
	[108., 315.],
	[500., 273.],
	[244., 271.],
	[157., 272.],
	[67., 274.],
	[444., 229.],
	[560., 230.],
	[271., 228.],
	[214., 230.],
	[96., 230.],
	[433., 193.],
	[539., 195.],
	[225., 193.],
	[119., 194.],
	[431., 179.],
	[231., 178.],
	[135., 178.],
]

PTS_GROUND_PLANE = [
	[15, 0],
	[20, 0],
	[25, 0],
	[30, 0],
	[40, 0],
	[60, 0],
	[80, 0],
	[100, 0],
	[15, -5],
	[15, -10],
	[15, 5],
	[15, 10],
	[20, -10],
	[20, 5],
	[20, 10],
	[20, 15],
	[30, -10],
	[30, -20],
	[30, 5],
	[30, 10],
	[30, 20],
	[50, -15],
	[50, -30],
	[50, 15],
	[50, 30],
	[70, -20],
	[70, 20],
	[70, 39],
]

pts_img = np.array(PTS_IMAGE_PLANE, dtype=np.float64)
pts_img = np.flip(pts_img, axis=1)

pts_ground = np.array(PTS_GROUND_PLANE, dtype=np.float64)
pts_ground *= 0.0254

DILATION_FACTOR = 3
DILATION_KERNEL = np.ones((DILATION_FACTOR, DILATION_FACTOR), np.uint8)

class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocation, "/point", 10)
        self.debug_pub = self.create_publisher(Image, "/debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.marker_pub = self.create_publisher(Marker, "/point_marker", 1)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        H, _= cv2.findHomography(pts_img, pts_ground)
        self.H_inv = np.linalg.inv(H)
        self.slope = 4
        self.epsilon = 1e-5
        self.dist_from_line = .25 # Meters
        self.lookahead = 4.0 # Meters
        sensitivity = 45
        self.lower_white = np.array([0, 0, 255-sensitivity])
        self.upper_white = np.array([255, sensitivity, 255])

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        y, _, _ = img.shape
        img[:6*y//10] = 0
        debug_img, m, b = self.get_line(img)
        m_W, b_W = self.transform(m, b) # Transform to world
        m_T, b_T = m_W, self.dist_from_line*np.sqrt(1 + (m_W**2)) + b_W # Transpose
        # point = self.intersection(m_T, b_T, self.lookahead)
        # if point is not None:
        #     x, y = point
        # else:
        #     x, y = 0, 0
        x = self.lookahead
        y = m_T*x + b_T

        # Debug
        self.draw_marker(x, y)
        self.get_logger().info(f"Equation of line in image is: y = {m}*x + {b}")
        self.get_logger().info(f"Equation of line from homography is: y = {m_W}*x + {b_W}")
        self.get_logger().info(f"Line to follow is: y = {m_T}*x + {b_T}")
        self.get_logger().info(f"Point to follow is: ({x}, {y})")
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        p = ConeLocation()
        p.x_pos = x
        p.y_pos = y
        self.cone_pub.publish(p)
    
    def get_line(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        edges = cv2.Canny(mask, 500, 1200)
        edges = cv2.dilate(edges, DILATION_KERNEL, iterations=1)
        debug_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Work in normal polar coordinates one "distance" is 1 and one "angle" is pi/180 radians
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        r, theta = lines[:,0,0], lines[:,0,1]
        c, s = np.cos(theta), np.sin(theta) + self.epsilon # Add to not divide by 0
        m, b = -c/s, r/s
        selection = m > self.slope
        m_selected, b_selected = m[selection], b[selection] # Select for vertical lines on right

        m, b = np.mean(m_selected), np.mean(b_selected)

        # Draw line
        self.draw_lines(debug_rgb, [m], [b])
        return debug_rgb, m, b

    def transform(self, m, b):
        l = np.array([[m, -1, b]])
        l_transformed = l @ self.H_inv
        c_x, c_y, c_1 = l_transformed[0,0], l_transformed[0,1], l_transformed[0,2]
        return -c_x/c_y, -c_1/c_y
    
    @staticmethod
    def draw_lines(img, m, b):
        for m, b in zip(m,b):
            x0, xf = -1000, 1000
            y0, yf = int(m*x0 + b), int(m*xf + b)
            cv2.line(img, (x0,y0), (xf,yf), (0, 0, 255), 2)
    
    @staticmethod
    def intersection(m, b, r):
        x1, y1 = -10, m*-10 + b
        x2, y2 = 10, m*10 + b
        dx = x2-x1
        dy = y2-y1
        dr = np.sqrt(dx**2 + dy**2)
        D = x1*y2 - x2*y1
        Delta = (r**2)*(dr**2) - D**2
        if Delta > 0:
            if (D*dy + np.sign(dy)*dx*np.sqrt(Delta))/dr**2 >= 0:
                return (D*dy + np.sign(dy)*dx*np.sqrt(Delta))/dr**2, (-D*dx + abs(dy)*np.sqrt(Delta))/dr**2
            else:
                return (D*dy - np.sign(dy)*dx*np.sqrt(Delta))/dr**2, (-D*dx - abs(dy)*np.sqrt(Delta))/dr**2

    def draw_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.marker_pub.publish(marker)
            

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
