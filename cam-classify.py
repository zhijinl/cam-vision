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
## File: loader.py for Cam-Vision
##
## Created by Zhijin Li
## E-mail:   <jonathan.zj.lee@gmail.com>
##
## Started on  Sun Oct 28 15:09:53 2018 Zhijin Li
## Last update Mon Oct 29 22:26:51 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import cv2
import time
import numpy as np
from utils import utils as utils
from utils import capture as cap
from utils import mobilenet as net


WIDTH_MULTIPLIER = 1
TOP_CLASSES      = 5
TARGET_SIZE      = None
POOLING_TYPE     = 'global_avg'

VERBOSE          = False
SKIP_FRAMES      = 60
IMAGENET_TXT     =  './data/imagenet/imagenet_dict.npy'


if __name__ == '__main__':

  network = net.load_mobilenet_anysize(
    WIDTH_MULTIPLIER, POOLING_TYPE)

  stream = cap.FastVideoStream(0).read_frames()

  fps = 0
  counter = 0
  top_scrs = [0.0]*TOP_CLASSES
  top_labs = ['none']*TOP_CLASSES
  label_dict = np.load(IMAGENET_TXT).item()

  while True:

    __start = time.time()

    frame = stream.get_frame()
    frame = cap.trim_frame_square(frame, .55, 0.5625)

    if (counter % SKIP_FRAMES) == 0:
      top_labs, top_scrs = utils.classify_frame(
        network,
        frame,
        TARGET_SIZE,
        TOP_CLASSES,
        label_dict,
        verbose=VERBOSE)

    cap.print_fps(frame, fps)
    frame = cap.make_pred_frame(frame, top_labs, top_scrs)

    cv2.imshow('Cam Classifier', frame)

    if cv2.waitKey(1) == ord('q'):
      stream.stop()
      break

    fps = 1.0/(time.time()-__start)
    counter += 1

  cv2.destroyAllWindows()
