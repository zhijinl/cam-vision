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
## Last update Wed Dec 12 00:17:38 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import cv2
import time
import torch
import numpy as np
from lib import yolov3tiny as net
from lib.utils import utils as utils
from lib.utils import capture as cap
import matplotlib.pyplot as plt


IMG_HEIGHT = 416
IMG_WIDTH  = 416
N_CHANNELS = 3

OBJ_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

SKIP_FRAMES      = 10
DARKNET_CFG      = './data/model/yolov3tiny/yolov3-tiny.cfg'
YOLOV3_TINY_W    = './data/model/yolov3tiny/yolov3-tiny.weights'
COCO_CLASSES     = './data/coco/coco.names'
TEST_IMG_DIR     = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 'data/darknet-test-images')

if __name__ == '__main__':

  weights = utils.load_dkn_weights(YOLOV3_TINY_W, np.float32)
  yolo = net.YOLO(DARKNET_CFG, N_CHANNELS, weights)
  coco_classes = utils.read_txt_as_strs(COCO_CLASSES)
  # print(yolo)

  imgs, paths = utils.load_img_folder(
    TEST_IMG_DIR,
    ext='jpg',
    permute_br=True,
    normalize=False)

  for indx, img in enumerate(imgs):

    __start = time.time()

    img_ltb, shift, ratio = utils.make_predict_inp(
      img,
      target_size=None,
      normalize=True,
      permute_br=False,
      letter_box=(IMG_HEIGHT, 0.5),
      to_channel_first=True)

    dets = utils.detect_frame(
      yolo,
      torch.FloatTensor(img_ltb),
      obj_thresh=OBJ_THRESHOLD,
      nms_thresh=NMS_THRESHOLD,
      box_correction=(shift, ratio))

    if dets is not None:
      img = cap.make_detection_frame(img, dets, coco_classes)

    print('img: {:20s} - prediction time: {:.3f} s'.format(
      os.path.basename(paths[indx]),
      time.time() - __start))

    plt.figure()
    plt.imshow(img)
  plt.show()
