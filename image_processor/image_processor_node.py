#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import sklearn 
import cv2
from cv_bridge import CvBridge, CvBridgeError

from .bridge.lanenet_bridge import LaneNetImageProcessor


class ImageProcessorNode(Node):

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 1)
        self.subscriber_
        self.bridge = CvBridge()
        self.weights_path = "/home/yvxaiver/lanenet-lane-detection/weights/tusimple_lanenet.ckpt"
        self.image_width = 720
        self.image_height = 1280
        self.processor = LaneNetImageProcessor(self.weights_path,self.image_width,self.image_height)
        self.lanenet_status = self.processor.init_lanenet()
        self.centerpts = []
        self.full_lanepts = []

    def image_callback(self,data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.lanenet_status:
                self.full_lanepts, self.centerpts = self.processor.image_to_trajectory(cv_frame)
            print(self.centerpts)
            pass
            if self.full_lanepts:
                for lane in self.full_lanepts:
                    for pt in lane:
                        cv2.circle(cv_frame,tuple(([0,self.image_height] - pt)*[-1,1]), 5, (0, 255, 0), -1)

            if self.centerpts:
                for lane in self.centerpts:
                    for i in range(len(lane[0])):
                        cv2.circle(cv_frame,(int(lane[0][i]),
                                                self.image_height-int(lane[1][i])), 5, (0, 0, 255), -1)
            cv2.imshow("camera", cv_frame)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e) # TODO: Error handing
        except Exception as e:
            print(e)

def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()

    rclpy.spin(image_processor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
