#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


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
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/point_px", 10)
        self.debug_pub = self.create_publisher(Image, "/debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        # TODO tune lookahead
        self.slope = 1.5
        self.look_ahead = 1000 # Measured in pixels
        self.epsilon = 1e-5
        self.sensitivity = 35
        self.lower_white = np.array([0,0,255-self.sensitivity])
        self.upper_white = np.array([255,self.sensitivity,255])

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        y, _, _ = img.shape
        # TODO find a good crop
        img[0:y//2] = 0
        img[4*y//5:y] = 0
        debug_img, u, v = self.get_point(img, self.lower_white, self.upper_white, self.epsilon, self.slope)
        v = self.look_ahead

        cv2.circle(debug_img, (u,v), 2, (0,0,255), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        px = ConeLocationPixel()
        px.u = u
        px.v = v
        self.cone_pub.publish(px)

    @staticmethod
    def get_point(img, lower_white, upper_white, epsilon, slope):
        # Pre-processing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        edges = cv2.Canny(mask, 500, 1200)
        debug_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Work in normal polar coordinates one "distance" is 1 and one "angle" is pi/180 radians
        lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
        r, theta = lines[:,0,0], lines[:,0,1]
        c, s = np.cos(theta), np.sin(theta) + epsilon # Add to not divide by 0
        m, b = -c/s, r/s
        m_positive, b_positive = m[m > slope], b[m > slope] # Select for vertical lines
        N_positive = m_positive.shape[0]
        m_negative, b_negative = m[m < -slope], b[m < -slope] # Select for vertical lines
        N_negative = m_negative.shape[0]

        # x_intersections = []
        # y_intersections = []
        s_x, s_y, N = 0, 0, 0
        for i in range(N_positive):
            for j in range(N_negative):
                x = (b_positive[i] - b_negative[j]) / (m_negative[j] - m_positive[i])
                s_x += x
                s_y += m_positive[i]*x + b_positive[i]
                N += 1
                # x_intersections.append(x)
                # y_intersections.append(y)

        # Draw positive
        for m_,b_ in zip(m_positive, b_positive):
            x0 = -1000
            y0 = int(m_*x0 + b_)
            xf = 1000
            yf = int(m_*xf + b_)
            cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)
        # Draw negative lines
        for m_,b_ in zip(m_negative, b_negative):
            x0 = -1000
            y0 = int(m_*x0 + b_)
            xf = 1000
            yf = int(m_*xf + b_)
            cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)
        
        return debug_rgb, s_x/N, s_y/N

    @staticmethod
    def get_point_other(img, lower_white, upper_white, epsilon, slope, lookahead):
        # Pre-processing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        edges = cv2.Canny(mask, 500, 1200)
        debug_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Work in normal polar coordinates one "distance" is 1 and one "angle" is pi/180 radians
        lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
        r, theta = lines[:,0,0], lines[:,0,1]
        c, s = np.cos(theta), np.sin(theta) + epsilon # Add to not divide by 0
        m, b = -c/s, r/s
        m_positive, b_positive = m[m > slope], b[m > slope] # Select for vertical lines
        N_positive = m_positive.shape[0]
        m_negative, b_negative = m[m < -slope], b[m < -slope] # Select for vertical lines
        N_negative = m_negative.shape[0]

        x_intersections = []
        y_intersections = []
        lines = []
        for i in range(N_positive):
            for j in range(N_negative):
                x = (b_positive[i] - b_negative[j]) / (m_negative[j] - m_positive[i])
                y = m_positive[i]*x + b_positive[i]
                x_intersections.append(x)
                y_intersections.append(y)
                lines.append((m_positive[i], b_positive[i], m_negative[j], b_negative[j]))
        mp, bp, mn, bn = lines[np.argmin(y_intersections)]
        
        # Draw lines
        x0 = -1000
        y0 = int(mp*x0 + bp)
        xf = 1000
        yf = int(mp*xf + bp)
        cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)
        y0 = int(mn*x0 + bn)
        yf = int(mn*xf + bn)
        cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)

        x = (lookahead - bp) / mp
        offset = 20
        return debug_rgb, x - offset, lookahead

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
