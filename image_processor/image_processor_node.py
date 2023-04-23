#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from points_vector.msg import PointsVector
from lanenet_out.msg import OrderedSegmentation

import sklearn 
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

from .bridge.lanenet_bridge import LaneNetImageProcessor


class ImageProcessorNode(Node):

    '''Handles feeding camera frame into lanenet, converting outputs into path to be followed, and publishes that path.'''

    def __init__(self):
        super().__init__('image_processor')

        # Mode should be given upon node run
        self.declare_parameter('mode')

        if str(self.get_parameter('mode').value) == "vanilla":
            cb_group = ReentrantCallbackGroup()
            self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.vanilla_image_callback, 1, callback_group=cb_group)
            self.publisher_ = self.create_publisher(PointsVector, '/lanenet_path', 1)
        elif str(self.get_parameter('mode').value) == "cluster_parall":
            print("Running parallel clustering...")
            self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.cluster_parall_image_callback, 1)
            self.publisher_ = self.create_publisher(OrderedSegmentation, '/lanenet_out', 1)
        else:
            raise ValueError('Mode not provided or does not exist.')

        self.bridge = CvBridge()
        self.weights_path = "/home/yvxaiver/lanenet-lane-detection/modelv3/tusimple/bisenetv2_lanenet/tusimple_val_miou=0.5660.ckpt-312"
        self.image_width = 256
        self.image_height = 128
        self.processor = LaneNetImageProcessor(self.weights_path,self.image_width,self.image_height,520,70.8,[0.005187456983582503, 0.0046280422281588405])
        self.lanenet_status = self.processor.init_lanenet()
        #self.lanenet_status = False
        self.centerpts = []
        self.full_lanepts = []
        self.following_path = []
        self.image_serial_n = 0
        self.current_k = 0

        

    def vanilla_image_callback(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            if self.lanenet_status:
                self.full_lanepts, self.centerpts, self.following_path = self.processor.image_to_trajectory(cv_frame, self.image_serial_n, self.current_k)
                msg = self.processor.get_point_vector_path()
                # print(self.full_lanepts)
                if msg: self.publisher_.publish(msg)
                self.image_serial_n += 1

            #self.image_save(cv_frame) 
            self.image_display(cv_frame)

        except Exception as e:
            print(e)


    def cluster_parall_image_callback(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            if self.lanenet_status:
                binary_seg_image, instance_seg_image, out_index = self.processor.image_to_segmentation(cv_frame)
                print(binary_seg_image.shape)
                print(instance_seg_image.shape)
                print(cv_frame.shape)
                msg = self.processor.get_ordered_segmentation_msg(cv_frame, binary_seg_image, instance_seg_image, out_index)
                self.publisher_.publish(msg)

        except Exception as e:
            print(e)
    

    def image_display(self, cv_frame):
        if self.full_lanepts:
                for lane in self.full_lanepts:
                    for pt in lane:
                        cv2.circle(cv_frame,tuple(([0,self.image_height] - pt)*[-1,1]), 5, (0, 255, 0), -1)
        if self.centerpts:
            for centerlane in self.centerpts:
                for i in range(len(centerlane[0])):
                    cv2.circle(cv_frame,(int(centerlane[0][i]),
                                        self.image_height-int(centerlane[1][i])), 5, (0, 0, 255), -1)
        if self.following_path:
            plt.close()
            plt.plot(self.following_path[0], self.following_path[1], ".r", label="path")
            plt.grid(True)
            #plt.show()
        
        cv2.imshow("camera", cv_frame)
        cv2.waitKey(1)

    
    def image_save(self, cv_frame):
        status = cv2.imwrite('/home/yvxaiver/output/0/'+str(self.image_serial_n)+".png",cv_frame)
        self.image_serial_n += 1
        print(status)


def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()
    rclpy.spin(image_processor)

    image_processor.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
