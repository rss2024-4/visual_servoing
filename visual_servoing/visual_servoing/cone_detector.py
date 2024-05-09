#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge#, CvBridgeError
from visualization_msgs.msg import Marker

from sensor_msgs.msg import Image
# from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocation


# import your color segmentation algorithm; call this function in ros_image_callback!
# from computer_vision.color_segmentation import cd_color_segmentation

PTS_IMAGE_PLANE = [
    [118, 263],
    [294, 226],
    [356, 243],
    [548, 249],
]

PTS_GROUND_PLANE = [
    [20.375, 18    ],
    [36.25,  6.56  ],
    [27.25,  0     ],
    [24.25,  -16.75],
]

pts_img = np.array(PTS_IMAGE_PLANE, dtype=np.float64)
pts_img = np.flip(pts_img, axis=1)

pts_ground = np.array(PTS_GROUND_PLANE, dtype=np.float64)
pts_ground *= 0.0254

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

        self.H, _= cv2.findHomography(pts_img, pts_ground)
        self.H_inv = np.linalg.inv(self.H)
        self.slope = 0.2
        self.epsilon = 1e-5
        self.lookahead_px = 30
        self.last_y_buffer = [0, 0, 0]

        dilation_factor = 3
        self.dilation_kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
        sensitivity = 45
        self.lower_white = np.array([0, 0, 255-sensitivity])
        self.upper_white = np.array([255, sensitivity, 255])

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        y, _, _ = img.shape
        img[:13*y//32] = 0
        img[18*y//32:] = 0
        debug_img, x, y = self.get_point_no_transform(img)
        x, y = self.transform_point(x, y)
        y -= 0.14

        # Smoothing y
        if len(self.last_y_buffer) > 3:
            del self.last_y_buffer[0]
        self.last_y_buffer.append(y)
        y = np.mean(self.last_y_buffer)
        if abs(y - self.last_y_buffer[-2]) > 0.15:
            y = self.last_y_buffer[-2]

        # Debug
        self.draw_marker(x, y)
        self.get_logger().info(f"Point to follow is: ({x}, {y})")
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        p = ConeLocation()
        p.x_pos = x
        p.y_pos = y
        self.cone_pub.publish(p)
    
    def transform_point(self, x, y):
        p = np.array([[x],[y],[1]])
        pn = self.H @ p
        return pn[0, 0] / pn[2, 0], pn[1, 0] / pn[2, 0]
    
    def get_point_no_transform(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        edges = cv2.Canny(mask, 500, 1200)
        edges = cv2.dilate(edges, self.dilation_kernel, iterations=1)
        debug_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Work in normal polar coordinates one "distance" is 1 and one "angle" is pi/180 radians
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        r, theta = lines[:,0,0], lines[:,0,1]
        c, s = np.cos(theta), np.sin(theta) + self.epsilon
        m, b = -c/s, r/s
        right, left = m > self.slope, m < -self.slope
        mr, br = m[right], b[right]
        ml, bl = m[left], b[left]

        mr, br = np.mean(mr), np.mean(br)
        ml, bl = np.mean(ml), np.mean(bl)

        k = np.sqrt((mr**2 + 1)/(ml**2 + 1))
        mb, bb = (-k*ml + mr)/(1 - k), (-k*bl + br)/(1 - k)
        self.draw_lines(debug_rgb, [mr, ml, mb], [br, bl, bb])

        intersection_x = (br - bl)/(ml - mr)
        intersection_y = mr*intersection_x + br
        y = intersection_y + self.lookahead_px
        x = (y-bb)/mb

        # cv2.circle(debug_rgb, (int(intersection_x), int(intersection_y)), 5, (255, 0, 0), -1)
        cv2.circle(debug_rgb, (int(x), int(y)), 5, (0, 255, 0), -1)
        return debug_rgb, x, y

    @staticmethod
    def draw_lines(img, m, b):
        for m, b in zip(m,b):
            x0, xf = -1000, 1000
            y0, yf = int(m*x0 + b), int(m*xf + b)
            cv2.line(img, (x0,y0), (xf,yf), (0, 0, 255), 2)

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
