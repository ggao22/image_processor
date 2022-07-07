#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageProcessorNode(Node):

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 10)
        self.subscriber_
        self.bridge = CvBridge()

    def image_callback(self,data):
        try:
           cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8") # TODO: adding image processing
           cv2.imshow("camera", cv_frame)
           cv2.waitKey(1)
        except CvBridgeError as e:
            print(e) # TODO: Error handing

def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()

    rclpy.spin(image_processor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()