#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from .bridge.lanenet_bridge import LaneNetImageProcessor


class ImageProcessorNode(Node):

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 10)
        self.subscriber_
        self.bridge = CvBridge()
        self.weights_path = "$HOME/lanenet-lane-detection/weights/tusimple_lanenet.ckpt"
        self.image_width = 640
        self.image_height = 480 
        self.processor = LaneNetImageProcessor(self.weights_path,self.image_width,self.image_height)
        self.lanenet_status = self.processor.init_lanenet()
        self.centerpts = []

    def image_callback(self,data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8") # TODO: adding image processing
            if self.lanenet_status:
                self.centerpts = self.processor.image_to_trajectory(self, cv_frame)
                
            # debug
            print(self.centerpts)
            print("\n")
            # cv2.imshow("camera", cv_frame)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e) # TODO: Error handing

def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()

    rclpy.spin(image_processor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
