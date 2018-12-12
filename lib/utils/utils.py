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
## Last update Wed Dec 12 21:12:00 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import cv2
import glob
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
    letter_box=None,
    to_channel_first=False):
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

  letter_box: tuple or None
  Side length and fill value of the square box when
  performing letter box transformation of the image.
  Default to None indicating no letter box transformation.

  to_channel_first: bool
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
  if letter_box:
    img, shift, ratio = letterbox_image(
      img, letter_box[0], fill=letter_box[1], normalize=False)
  img = np.expand_dims(img.astype(np.float32), axis=0)
  if to_channel_first: img = img.transpose(0,3,1,2)
  if letter_box: return (img, shift, ratio)
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
    to_channel_first=False,
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

  to_channel_first: bool
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
  inp = make_predict_inp(
    frame,target_size,normalize,permute_br,to_channel_first)
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


def load_img_folder(
    folder,
    ext,
    permute_br=True,
    normalize=True):
  """

  Load all images inside given folder.

  Parameters
  ----------
  folder: str
  Absolute folder to image folder.

  ext: str
  Image file extension. Must be recognizable by
  OpenCV.

  permute_br: bool
  Whether blue and red channel permutation should
  be performed.

  normalize: bool
  Indicating whether the image pixel value should
  be divided by 255.0.

  Returns
  ----------
  tuple(list)
  The image (np.array) list and path list.

  """
  __imgs = []
  __plst = glob.glob(os.path.join(folder,'*.{}'.format(ext)))
  for __p in __plst:
    __img = cv2.imread(__p, cv2.IMREAD_UNCHANGED)
    if normalize: __img = __img/255.0
    if permute_br:
      __img[:,:,0],__img[:,:,2] = __img[:,:,2],__img[:,:,0].copy()
    __imgs.append(__img)
  return __imgs, __plst


def letterbox_image(img, frame_size, fill=0.5, normalize=True):
  """

  Letter box an input image.

  Image will be centered into a squared frame,
  where the longer side of the image is resized
  to the frame size and the shorter side is resized
  by keepng the same aspect ratio.

  Parameters
  ----------
  img: np.array
  The input image. Assumed to be rank-3, channel-last.

  frame_size: int
  Size of the square frame.

  fill: float
  Value used to fill empty border. Defaults to 0.5.

  normalize: bool
  Whether input image should be divided by 255.

  Returns
  ----------
  tuple
  1. The letter boxed image.
  2. Horizontal and vertical shift in number of pixels
     with respect to square frame size.
  3. The resize ratio: box size/ original longer side.

  """
  if normalize: img = img.astype(np.float32)/255.0
  __lindx = np.argmax(img.shape[:2])
  __ratio = frame_size/img.shape[:2][__lindx]
  __rsize = (np.array(img.shape[:2])*__ratio).astype(np.int32)
  __shift = np.array([frame_size]*2, dtype=np.int32) - __rsize
  __shift = (__shift/2).astype(np.int32)

  __rsimg = torch.nn.functional.interpolate(
    torch.from_numpy(img).permute(2,0,1).unsqueeze(0),
    size=(__rsize[0], __rsize[1]), mode='bilinear', align_corners=True)
  __rsimg = __rsimg.squeeze().permute(1,2,0)

  __ltbox = np.ones((frame_size, frame_size, img.shape[2]))*fill
  __ltbox[__shift[0]:__shift[0]+__rsize[0],
          __shift[1]:__shift[1]+__rsize[1],:] = __rsimg
  return (__ltbox.astype(np.float32),
          np.flip(__shift, axis=0).copy(), __ratio)


def correct_bboxes(dets, shift, ratio):
  """

  Correct bounding box centers and scales
  to match original input image before
  letter boxing.

  Parameters
  ----------
  dets: torch.tensor
  A rank-2 tensor, where each col is a size-6
  vector representing a detection bounding box.
  The meaning of each element in the vector is
  as follows:
  1. bbox begin point x coordinate.
  2. bbox begin point y coordinate.
  3. bbox width.
  4. bbox height.
  5. max proba = max class proba * objectness score.
  6. class index of the corresponding max proba.

  shift: np.array
  Horizontal and vertical shift in number of pixels
  with respect to square frame size.

  ratio
  The resize ratio: box size/ original longer side.

  """
  dets[:2,:] -= torch.from_numpy(shift).float().view(2,1)
  dets[:4,:] /= ratio


def nms(dets, nms_thresh):
  """

  Do non-maximum suppression.

  Parameters
  ----------
  dets: torch.tensor
  A rank-2 tensor, where each col is a size-6
  vector representing a detection bounding box.
  The meaning of each element in the vector is
  as follows:
  1. bbox begin point x coordinate.
  2. bbox begin point y coordinate.
  3. bbox width.
  4. bbox height.
  5. max proba = max class proba * objectness score.
  6. class index of the corresponding max proba.

  nms_thresh: float
  NMS threshold.

  Returns
  ----------
  torch.tensor
  New bounding box attributed with boxes having
  high IOU with the top prediction sppressed.

  """
  __book = {}
  __, __ord = torch.sort(dets[4,:], descending=True)
  __dets, __cls = dets[:,__ord], set(dets[-1,:])
  for __i, __c in enumerate(__dets[-1,:]):
    if int(__c) not in __book: __book[int(__c)] = __i
    else:
      __iou = compute_iou(__dets[:4,__i], __dets[:4,__book[int(__c)]])
      if __iou > nms_thresh: __dets[4,__i] = -1
  return __dets[:, __dets[4,:] >= 0]


def compute_iou(lhs, rhs):
  """

  Compute the intersection over union of two
  bounding boxes.

  Parameters
  ----------
  lhs: torch.tensor
  Bounding box 1.

  rhs: torch.tensor
  Bounding box 2.

  Returns
  ----------
  float
  The intersection over union.

  """
  __beg = np.array([max(lhs[0], rhs[0]), max(lhs[1], rhs[1])])
  __end = np.array([min(lhs[0]+lhs[2], rhs[0]+rhs[2]),
                    min(lhs[1]+lhs[3], rhs[1]+rhs[3])])
  __num = np.prod(__end-__beg)
  if __num <= 0: return 0
  __den = lhs[2]*lhs[3]+rhs[2]*rhs[3] - __num
  return __num/__den


def detect_frame(
    model,
    frame,
    obj_thresh=0.5,
    nms_thresh=None,
    box_correction=None):
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

  box_correction: tuple or None
  A tuple of (shift, ratio) parameters used to
  perform letter box correction to bring the
  centers and scale of the bounding boxes to
  match the original image before letter boxing.
  This need to be set if bounding boxes are plotted
  into the original image before letter boxing.
  Defaults to None, indicating no correction.

  Returns
  ----------
  torch.tensor
  A rank-2 tensor, where each col is a size-6
  vector representing a detection bounding box.
  The meaning of each element in the vector is
  as follows:
  1. bbox begin point x coordinate.
  2. bbox begin point y coordinate.
  3. bbox width.
  4. bbox height.
  5. max proba = max class proba * objectness score.
  6. class index of the corresponding max proba.

  """
  __detections = model(frame)
  __boxes = []
  for __d in __detections:
    __p = __d.permute(0,2,1,3).contiguous().view(__d.shape[2],-1)
    __mprb, __midx = torch.max(__p[5:,:],dim=0)
    __p[4,:] *= __mprb              # obj_score * max class proba
    __b = torch.cat([
      __p[:5,:], __midx.type(torch.FloatTensor).unsqueeze(0)],0)
    __b = __b[:, (__b[4,:] > obj_thresh)]
    if __b.numel():
      __b[:2,:] -= __b[2:4,:]/2.0
      __boxes.append(__b)
  if len(__boxes) == 0: return None
  __dets = torch.cat(__boxes, dim=1)
  if nms_thresh: __dets = nms(__dets, nms_thresh)
  if box_correction: correct_bboxes(__dets, *box_correction)
  return __dets
