#!/usr/bin/env python3
"""
ros2 Image to LaneNet Bridge
"""

import os.path as ops
import sys
import time

import sklearn 
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('/home/yvxaiver/lanenet-lane-detection')
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

sys.path.append('/home/yvxaiver/LaneNet_to_Trajectory')
from LaneNetToTrajectory import LaneProcessing, DualLanesToTrajectory


class LaneNetImageProcessor():

    def __init__(self, weights_path, image_width, image_height, max_lane_y=420, WARP_RADIUS=20, WP_TO_M_Coeff=[1,1]):
        self.weights_path = weights_path
        self.image_width = image_width
        self.image_height = image_height
        self.lane_processor = LaneProcessing(self.image_width,self.image_height,max_lane_y,WARP_RADIUS,WP_TO_M_Coeff)
        self.calibration = True
        self.following_path = []


    def init_lanenet(self):

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        self.net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)

        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)

        return True


    def image_to_trajectory(self, cv_image, lane_fit=True):

        lanenet_start = time.time()
        image_vis = cv_image
        image = cv2.resize(cv_image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [image]}
        )

        if not lane_fit:
            out_dict = self.postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                with_lane_fit=lane_fit,
                data_source='tusimple'
            )
            return out_dict

        full_lane_pts = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            data_source='tusimple'
        )
        print('Lanenet cost time: {:.5f}s'.format(time.time() - lanenet_start))

        lanep_start = time.time()
        
        
        self.lane_processor.process_next_lane(full_lane_pts)
        full_lane_pts = self.lane_processor.get_full_lane_pts()
        physical_fullpts = self.lane_processor.get_physical_fullpts()

        if self.calibration:
            self.lane_processor.auto_warp_radius_calibration(FRAME_BOTTOM_PHYSICAL_WIDTH=0.43)
            self.lane_processor.y_dist_calibration_tool(cv_image)
            print(self.lane_processor.WARP_RADIUS)
            print(self.lane_processor.get_wp_to_m_coeff())
            self.calibration = False
            raise SystemExit

        phy_centerpts = []
        phy_splines = []
        closest_lane_dist = float('inf')
        closest_lane_idx = 0
        if physical_fullpts:
            print(physical_fullpts)
            for i in range(len(physical_fullpts)):
                if not i: continue
                traj = DualLanesToTrajectory(physical_fullpts[i-1],physical_fullpts[i],N_centerpts=20)
                phy_centerpts.append(traj.get_centerpoints())
                phy_splines.append(traj.get_spline())
            for i in range(len(phy_splines)):
                new_dist = abs(phy_splines[i](0)-self.image_width/2)
                if new_dist < closest_lane_dist:
                    closest_lane_dist = new_dist
                    closest_lane_idx = i
            if phy_centerpts: self.following_path = phy_centerpts[closest_lane_idx]

        # For display output
        centerpts = []
        if full_lane_pts:
            for i in range(len(full_lane_pts)):
                if not i: continue
                traj = DualLanesToTrajectory(full_lane_pts[i-1],full_lane_pts[i],N_centerpts=20)
                centerpts.append(traj.get_centerpoints())
        
        print('Lane processing cost time: {:.5f}s'.format(time.time() - lanep_start))
        if centerpts: return full_lane_pts, centerpts
        return None, None

