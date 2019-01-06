#! /usr/bin/env python3
# Copyright (C) 2018  Zhijin Li

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## ---------------------------------------------------------------------------
##
## File: cam-detect.py for Cam-Vision
##
## Created by Zhijin Li
## E-mail:   <jonathan.zj.lee@gmail.com>
##
## Started on  Sat Nov 10 23:52:48 2018 Zhijin Li
## Last update Wed Dec 26 11:38:52 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import cv2
import time
import torch
import argparse
import numpy as np
from lib import yolov3tiny as net
from lib.utils import utils as utils
from lib.utils import capture as cap
import matplotlib.pyplot as plt


FRAME_SIZE = 416
N_CHANNELS = 3

OBJ_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

SKIP_FRAMES      = 10
DARKNET_CFG      = './data/model/yolov3tiny/yolov3-tiny.cfg'
YOLOV3_TINY_W    = './data/model/yolov3tiny/yolov3-tiny.weights'
COCO_CLASSES     = './data/coco/coco.names'
TEST_IMG_DIR     = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 'data/darknet-test-images')

DARKNET_BOX_SIZE = 416


if __name__ == '__main__':

  if not os.path.exists(YOLOV3_TINY_W):
    utils.download_yoolov3tiny_weights(YOLOV3_TINY_W)

  parser = argparse.ArgumentParser()
  parser.add_argument('--run_test', action='store_true')
  parser.add_argument('--use_darknet_boxes', action='store_true')

  args = parser.parse_args()

  yolo, coco_classes = net.init_detector(
    DARKNET_CFG,YOLOV3_TINY_W,COCO_CLASSES, N_CHANNELS)

  if args.run_test:

    if args.use_darknet_boxes:

      imgs, __ = utils.load_img_folder(
        TEST_IMG_DIR,
        ext='ltb.raw',
        permute_br=False,
        normalize=False,
        loader=lambda p: np.transpose(
          np.fromfile(p, dtype=np.float32).reshape(
            3,DARKNET_BOX_SIZE,DARKNET_BOX_SIZE), [1,2,0]))

    else:

      imgs, __ = utils.load_img_folder(
        TEST_IMG_DIR,
        ext='jpg',
        permute_br=True,
        normalize=False)

    for indx, img in enumerate(imgs):

      img = net.detect_color_img(
        model=yolo,
        img=img,
        classes=coco_classes,
        obj_threshold=OBJ_THRESHOLD,
        iou_threshold=NMS_THRESHOLD,
        box_size=FRAME_SIZE if not args.use_darknet_boxes else None,
        permute_br=False,
        normalize=False if args.use_darknet_boxes else True,
        verbose=True)

      plt.figure()
      plt.imshow(img)
    plt.show()

  else:

    fps = 0
    counter = 0
    stream = cap.FastVideoStream(0).read_frames()

    dets = None
    bboxes = None
    while True:
      __start = time.time()

      frame = stream.get_frame()
      frame = cap.trim_resize_frame_square(frame, FRAME_SIZE)

      if (counter % SKIP_FRAMES) == 0:
        img = utils.make_predict_inp(
          frame,
          target_size=None,
          normalize=True,
          permute_br=True,
          to_channel_first=True)
        dets = utils.detect_frame(
          yolo, torch.FloatTensor(img), OBJ_THRESHOLD, NMS_THRESHOLD)

      if dets is not None:
        frame = cap.make_detection_frame(frame, dets, coco_classes)
      cap.print_fps(frame, fps)
      cv2.imshow('Cam Detector', frame)

      if cv2.waitKey(1) == ord('q'):
        stream.stop()
        break

      fps = 1.0/(time.time()-__start)
      counter += 1

    cv2.destroyAllWindows()
