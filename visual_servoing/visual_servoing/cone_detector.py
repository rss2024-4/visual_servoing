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

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):        
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # In case we want only a portion of the image
        # y, _, _ = img.shape
        # img[0:y//2,:,:] = 0
        # img[4*y//5:y,:,:] = 0
        u, v = self.get_point(img)

        cv2.circle(img, (u,v), 2, (0,0,255), 2) # Circle for debug
        debug_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.debug_pub.publish(debug_msg)

        px = ConeLocationPixel()
        px.u = u
        px.v = v
        self.cone_pub.publish(px)

    def get_point(self, img):
        edges = cv2.Canny(img, 50, 500, 3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150, None, 0, 0)
        new = np.zeros(img.shape)
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            cv2.line(new, (int(x0 + 1000*(-b)), int(y0 + 1000*(a))), (int(x0 - 1000*(-b)), int(y0 - 1000*(a))), (0, 0, 255), 2)
        return (0,0)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
