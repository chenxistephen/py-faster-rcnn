#!/usr/bin/env python

# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals, im_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp
import cv2
import h5py
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
                        
    parser.add_argument('--imglist', dest='imglistFile',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
                        
    parser.add_argument('--imgdir', dest='imgPath',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--out_h5file', dest='h5file',
                        help='dataset to test',
                        default='voc_2007_test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    # RPN test settings
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    
    imglist = [l.split('\t')[0] for l in open(args.imglistFile,'r').readlines()]

    fout = h5py.File(args.h5file, 'w')

    im_scales = [] 
    im_to_roi_idx = []
    num_rois = []
    rpn_rois = np.empty((0,4), np.float64)
    rpn_scores = np.empty((0,1), np.float64)
    
    baseIdx = 0 
    for imgName in imglist[:100]:
	imgFileName = osp.join(args.imgPath, imgName)
	print imgFileName
	img = cv2.imread(imgFileName)
	boxes, scores, scale = im_proposals(net, img)
        im_scales.append(scale)
	num_boxes = boxes.shape[0]
	print num_boxes
	im_to_roi_idx.append(baseIdx)
	baseIdx += num_boxes
	num_rois.append(num_boxes)
	rpn_rois = np.vstack((rpn_rois, boxes))
	rpn_scores = np.vstack((rpn_scores, scores))
        #print boxes

    fout.create_dataset('im_scales', data=np.array(im_scales))
    fout.create_dataset('im_to_roi_idx', data=np.array(im_to_roi_idx))
    fout.create_dataset('num_rois', data=np.array(num_rois))
    fout.create_dataset('rpn_rois', data=np.array(rpn_rois))
    fout.create_dataset('rpn_scores', data=np.array(rpn_scores))

    print 'Wrote RPN proposals to {}'.format(rpn_file)
