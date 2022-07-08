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

    def __init__(self, weights_path, image_width, image_height):
        self.weights_path = weights_path
        self.image_width = image_width
        self.image_height = image_height


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


    def image_to_trajectory(self, image):

        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [image]}
        )

        full_lane_pts = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            data_source='tusimple'
        )

        full_lane_pts = LaneProcessing(full_lane_pts,image_width=self.image_width,image_height=self.image_height).get_full_lane_pts()

        centerpts = []
        splines = []

        # closest_lane_dist = float('inf')
        # closest_lane_idx = 0

        if full_lane_pts:
            for i in range(len(full_lane_pts)):
                if not i: continue
                traj = DualLanesToTrajectory(full_lane_pts[i-1],full_lane_pts[i],N_centerpts=20)
                centerpts.append(traj.get_centerpoints())
                splines.append(traj.get_spline())
        #     for i in range(len(splines)):
        #         if abs(splines[i](0)-self.image_width/2) < closest_lane_dist:
        #             closest_lane_idx = i
        
        # if centerpts: return full_lane_pts, centerpts[closest_lane_idx]

        if centerpts: return full_lane_pts, centerpts

        return None, None

        
        
