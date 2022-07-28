#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from points_vector.msg import PointsVector

import sklearn 
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

from .bridge.lanenet_bridge import LaneNetImageProcessor


class ImageProcessorNode(Node):

    '''Handles feeding camera frame into lanenet, converting outputs into path to be followed, and publishes that path.'''

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 1)
        self.publisher_ = self.create_publisher(PointsVector, '/lanenet_path', 1)
        self.bridge = CvBridge()
        self.weights_path = "/home/yvxaiver/lanenet-lane-detection/model/tusimple/bisenetv2_lanenet/tusimple_val_miou=0.6789.ckpt-8288"
        self.image_width = 1280
        self.image_height = 720
        self.processor = LaneNetImageProcessor(self.weights_path,self.image_width,self.image_height)
        self.lanenet_status = self.processor.init_lanenet()
        self.centerpts = []
        self.full_lanepts = []
        
        self.image_serial_n = 0
        

    def image_callback(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            if self.lanenet_status:
                self.full_lanepts, self.centerpts = self.processor.image_to_trajectory(cv_frame)
                msg = self.processor.get_point_vector_path()
                if msg: self.publisher_.publish(msg)

            # self.image_save(cv_frame) 
            self.image_display(cv_frame)

        except Exception as e:
            print(e)
    
    def image_display(self, cv_frame):
        if self.full_lanepts:
                for lane in self.full_lanepts:
                    for pt in lane:
                        cv2.circle(cv_frame,tuple(([0,self.image_height] - pt)*[-1,1]), 5, (0, 255, 0), -1)
        if self.centerpts:
            print(self.centerpts)
            for centerlane in self.centerpts:
                for i in range(len(centerlane[0])):
                    cv2.circle(cv_frame,(int(centerlane[0][i]),
                                        self.image_height-int(centerlane[1][i])), 5, (0, 0, 255), -1)
        cv2.imshow("camera", cv_frame)
        cv2.waitKey(1)
    
    def image_save(self, cv_frame):
        status = cv2.imwrite('/home/yvxaiver/output/1/'+str(self.image_serial_n)+".jpg",cv_frame)
        self.image_serial_n += 1
        print(status)


def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()

    rclpy.spin(image_processor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
