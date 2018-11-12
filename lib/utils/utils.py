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
## Last update Mon Nov 12 23:23:24 2018 Zhijin Li
## ---------------------------------------------------------------------------


import os
import numpy as np


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


def make_predict_inp(img, target_size, normalize=True):
  """

  Transform an image for TF prediction mode.

  Parameters
  ----------
  img: np.ndarray
  An input image array.

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
  if target_size is not None:
    img = cv2.resize(img, dsize=(target_size, target_size))
  if normalize:
    img = img.astype(np.float32) / 255.0
  return np.expand_dims(img.astype(np.float32), axis=0)


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


class YOLO():
  """

  Class representing Darknet config file for YOLO
  models.

  """


  def __init__(self, cfg_path):
    """

    Constructor

    Parameters
    ----------
    cfg_path: str
    Path to the Darknet config file.

    """
    self.cfg = self.__parse_darknet_cfg(cfg_path)
    self.yolo_name_dict  = {
      'conv2d': {
        'n_kernels'  : 'filters',
        'kernel_size': 'size',
        'stride'     : 'stride',
        'padding'    : 'pad'
      }
    }


  def __parse_darknet_cfg(self, cfg_path):
    """

    Parse Darknet config file for YOLO models.

    Parameters
    ----------
    cfg_path: str
    Path to the Darknet config file.

    Returns
    ----------
    list
    A list where each entry is a dictionary with key
    equal to the layer name and value equal to a
    dictionary for configurations of layer params.

    For example, an entry representing a convolution
    layer might be:

    ```
    {
      'convolutiona_1':
      {
        'batch_normalize': '1',
        'filters': '16',
        'size': '3',
        ...
      }
    }
    ```

    """
    __lines = read_txt_as_strs(cfg_path, cmnt='#')
    return self.__parse_cfg_sections(__lines)


  def __parse_cfg_sections(self, cfg_lines):
    """

    Parse different config file sections.

    Parameters
    ----------
    cfg_lines: list
    A list of parsed config file lines.

    Returns
    ----------
    dict
    A dictionary where eahc key refers to
    a section of the config file (such as
    'convolution') and the corresponding
    value is a dict representing the section
    configuration.

    """
    __sections = []
    __curr_sect = {}
    __sect_name = None
    for __l in cfg_lines:
      if __l.startswith('['):
        if __curr_sect: __sections.append(__curr_sect)
        __curr_sect = {}
        __sect_name = __l.lstrip('[').rstrip(']')
        __curr_sect[__sect_name] = {}
      else:
        __k, __v = __l.split('=')
        __curr_sect[__sect_name][__k] = __v
    return __sections


  def __get_layer_names(self):
    """

    Get a list of layer names.

    Returns
    ----------
    list
    A list with layer names.

    """
    pass


  def __get_detections(self):
    """

    Get config for detections.

    """
    pass
