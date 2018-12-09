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
## File: utils.py for Cam-Vision
##
## Created by Zhijin Li
## E-mail:   <jonathan.zj.lee@gmail.com>
##
## Started on  Sun Oct 28 20:36:56 2018 Zhijin Li
## Last update Sun Dec  9 22:23:17 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import torch
import numpy as np


def print_mat(mat, width=10, prec=4):
  """
  A nice printer for floating point
  matrices.

  Parameters
  ----------
  mat: 2D matrix
  An input 2D matrix to print.

  width: int
  Minimum width for each element to print.

  prec: int
  Floating point precision for each element
  to print.

  """
  for __indx in range(mat.shape[0]):
    __str = '{:{width}.{prec}f} '*mat.shape[1]
    print(__str.format(
      *mat[__indx,:], width=width, prec=prec))


def read_txt_as_strs(txt_path, strip=' ', cmnt=None):
  """

  Read a txt file. Each line will be treated
  as a string.

  Empty lines will be skipped. Spaces will be
  automatically stripped.

  Parameters
  ----------
  txt_path: str
  Path to the txt file.

  strip: bool
  Character(s) stripped from the beginning and
  the end of each line. Defaults to whitespace.
  Use `None` to indicate no-op.

  cmnt: str
  Comment character. If a line (after stripping,
  if strip character(s) is not `None`) starts with
  `cmnt` will not be read.

  Returns
  ----------
  list
  A list of strings, where each string represents
  a line in the txt file.

  """
  __lines = []
  with open(txt_path, 'r') as __f:
    for __l in __f.readlines():
      if strip is not None:
        __l = __l.strip().strip(strip)
      if not __l: continue
      if cmnt is not None:
        if __l.startswith(cmnt): continue
      __lines.append(__l.strip('\n'))
  return __lines


def load_img(img_path, target_size, normalize=True):
  """

  Load image for TF prediction mode.

  Parameters
  ----------
  img_path: str
  Path to image file.

  target_size: int
  Target square size for image resizing.

  normalize: bool
  Whether input image should be divided by 255.

  Returns
  ----------
  np.ndarray
  Tensor with rank 4, to be used for
  TF model prediction.

  """
  img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
  if target_size is not None:
    img = cv2.resize(img, dsize=(target_size, target_size))
  if normalize:
    img = img.astype(np.float32) / 255.0
  return np.expand_dims(img.astype(np.float32), axis=0)


def make_predict_inp(
    img,
    target_size=None,
    normalize=True,
    permute_br=True,
    channel_first=False):
  """

  Transform an image for TF prediction mode.

  Parameters
  ----------
  img: np.ndarray
  An input image array. Assumed to be RGB image.

  target_size: int
  Target square size for image resizing. Defaults
  to None, i.e. no resizing.

  normalize: bool
  Whether input image should be divided by 255.

  permute_br: bool
  Whether permutation of the Blue and Red
  channels should be performed. This is generally
  needed when the input image is read using OpenCV,
  since OpenCV uses BGR ordering, while most neural
  networks assumes input to be RGB ordering. Defaults
  to True.

  channel_first: bool
  When set to true, the input image will be
  converted to `channel_first` ordering. Defaults to
  False.

  Returns
  ----------
  np.ndarray
  Tensor with rank 4, to be used for
  TF model prediction.

  """
  if target_size:
    img = cv2.resize(img, dsize=(target_size, target_size))
  if normalize:
    img = img.astype(np.float32) / 255.0
  if permute_br:
    img[:,:,0], img[:,:,2] = img[:,:,2], img[:,:,0].copy()
  img = np.expand_dims(img.astype(np.float32), axis=0)
  if channel_first:
    return img.transpose(0,3,1,2)
  return img


def predict_top(model, img, top_classes, label_dict):
  """

  Run prediction on input image and get
  prediction scores and class indices for
  `top_classes` classes.

  Parameters
  ----------
  model: tf.keras.models.Model
  A keras model.

  img: np.ndarray
  Input image in form of 4D tensor.

  top_classes: int
  Number of top classes for prediction.

  label_dict: dict
  Dictionary with keys the prediction indices
  (int) and values the corresponding class
  labels (str).

  Returns
  ----------
  tuple
  Tuple with 3 elements:
  - a list of predicted class labels descending
    order,
  - a list with corresponding prediction
    scores.

  """
  scores = np.squeeze(model.predict(img))
  top_indx = np.argsort(scores)[-top_classes:]
  top_scrs = scores[top_indx]
  top_labs = [label_dict[indx] for indx in top_indx]
  return (
    list(reversed(top_labs)),
    list(reversed(top_scrs)))


def get_imagenet_dict(txt_path):
  """

  Make ImageNet ground truth dict.
  The ground truth dictionay maps
  a class index to its label.

  The .txt file can be found at:
  https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

  Parameters
  ----------
  txt_path: str
  Path to the txt file with ImageNet
  class index-to-label mappings.

  Returns
  ----------
  dict
  A dictionay with class indices as
  keys and their corresponding ImageNet
  labels as values.

  """
  imagenet_dict = {}
  with open(txt_path, 'r') as f:
    for __l in f.readlines():
      key, val = __l.split(':')
      __v = val.strip()[:-1].replace("'", '')
      imagenet_dict[int(key)] = __v
  return imagenet_dict


def classify_frame(
    model,
    frame,
    target_size,
    top_classes,
    label_dict,
    normalize=True,
    permute_br=True,
    channel_first=False,
    verbose=True):
  """

  Run classification on input frame.

  Parameters
  ----------
  model: tf.keras.models.Model
  A keras model.

  frame: np.ndarray
  An input image frame.

  target_size: int
  Target square image size for resizing.
  None indicates no resizing.

  top_classes: int
  Number of top classes for prediction.

  label_dict: dict
  Dictionary with keys the prediction indices
  (int) and values the corresponding class
  labels (str).

  normalize: bool
  Whether input image should be divided by 255.

  permute_br: bool
  Whether permutation of the Blue and Red
  channels should be performed. This is generally
  needed when the input image is read using OpenCV,
  since OpenCV uses BGR ordering, while most neural
  networks assumes input to be RGB ordering. Defaults
  to True.

  channel_first: bool
  When set to true, the input image will be
  converted to `channel_first` ordering. Defaults to
  False.

  verbose: bool
  Controls console print verbosity. Defaults
  to True.

  Returns
  ----------
  tuple
  Tuple with 3 elements:
  - a list of predicted class labels with
    prediction scores in descending order,
  - a list with corresponding prediction
    scores.

  """
  inp = make_predict_inp(frame, target_size, normalize)
  top_labs, top_scrs = predict_top(
    model, inp, top_classes, label_dict)
  if verbose:
    print('\nframe size: {} x {}'.format(
      inp.shape[1], inp.shape[2]))
    print('top {} predictions: '.format(top_classes))
    for __c in range(top_classes):
      print('{:25s}: {:.3f}'.format(
        top_labs[__c][:min(len(top_labs[__c]),20)],
        top_scrs[__c]))
  return (top_labs, top_scrs)


def load_dkn_weights(w_path, dtype, skip_bytes=20):
  """

  Load Darknet weight file.

  Parameters
  ----------
  w_path: str
  Path to the weight file.

  dtype: str or datatype
  Data type of stored weights.

  skip_bytes: int
  Number of bytes to skip. Darknet weight
  file starts with 5 x int32 (20 bytes) header
  elements.

  Returns
  ----------
  np.array
  Weight array.

  """
  with open(w_path, 'rb') as __wf:
    __wf.seek(skip_bytes, 0)
    __content = np.fromfile(__wf, dtype)
  return __content


def nms(dets, nms_thresh):
  """

  Do non-maximum suppression.

  """
  return dets


def detect_frame(model, frame, obj_thresh=0.5, nms_thresh=None):
  """

  Detect objects in a frame.

  Parameters
  ----------
  model: YOLO
  The YOLO detector model.

  frame: torch.tensor
  The input frame as a torch rank-4 tensor.

  obj_thresh: float
  Threshold on objectiveness and class
  probabilities.

  nms_thresh: float
  Threshold on IOU used during nms.

  Returns
  ----------
  torch.tensor
  A rank-2 tensor, where each row is a size-7
  vector representing a detection bounding box.
  The meaning of each element in the vector is
  as follows:
  1. bbox begin point x coordinate.
  2. bbox begin point y coordinate.
  3. bbox width.
  4. bbox height.
  5. objectness score.
  6. max class probability.
  7. class index of the corresponding max prob.

  """
  __detections = model(frame)
  __boxes = []
  for __d in __detections:
    __p = __d.permute(0,2,1,3).contiguous().view(__d.shape[2],-1)
    __mprb, __midx = torch.max(__p[5:,:],dim=0)
    __b = torch.cat([
      __p[:5,:], __mprb.unsqueeze(0),
      __midx.type(torch.FloatTensor).unsqueeze(0)],0)
    __b = __b[:, (__b[4,:]*__b[5,:] > obj_thresh)]
    __b[:2,:] -= __b[2:4,:]/2.0
    __boxes.append(__b)
  __dets = torch.cat(__boxes, dim=1)
  if nms_thresh: dets = nms(dets, nms_thresh)
  return __dets
