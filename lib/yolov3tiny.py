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
## File: yolov3.py for Cam-Vision
##
## Created by Zhijin Li
## E-mail:   <jonathan.zj.lee@gmail.com>
##
## Started on  Sat Nov 10 23:21:29 2018 Zhijin Li
## Last update Tue Nov 13 22:12:49 2018 Zhijin Li
## ---------------------------------------------------------------------------


import torch
import torchvision
from .utils import utils as utils


class YOLO():
  """

  Class representing YOLO models constructed from
  Darknet config file.

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
    print(self.cfg)


  @property
  def dkn_conv(self):
    """

    Name dictionary for parameters of Darknet
    convolutional layers.

    """
    return {
      'filters': 'filters',
      'ksize'  : 'size',
      'stride' : 'stride',
      'padding': 'pad'
    }


  @property
  def dkn_pool(self):
    """

    Name dictionary for parameters of Darknet
    max pooling layers.

    """
    return {
      'ksize' : 'kernel_size',
      'stride': 'stride'
    }


  @property
  def dkn_up(self):
    """

    Name dictionary for parameters of Darknet
    upsample layer.

    """
    return {
      'factor' : 'stride'
    }


  @property
  def dkn_layers(self):
    """

    Name dictionary for Darknet layers.

    """
    return {
      'conv2d'  : 'convolutional',
      'up'      : 'upsample',
      'skip'    : 'route',
      'maxpool' : 'maxpool',
      'detect'  : 'yolo'
    }


  @property
  def dkn_layer_makers(self):
    """

    Name dictionary for factory functions
    creating pytorch layers from corresponding
    darknet counterparts.

    """
    return {
      self.dnk_layers['conv2d']  : self.__make_convolution_block,
      self.dnk_layers['up']      : self.__make_upsample,
      self.dnk_layers['skip']    : self.__make_route,
      self.dnk_layers['maxpool'] : self.__make_maxpool,
      self.dnk_layers['detect']  : self.__make_detection,
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
    __lines = utils.read_txt_as_strs(
      cfg_path, cmnt='#')
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
    __secs = {}
    __sec_name = None
    __sec_count = {}
    for __l in cfg_lines:
      if __l.startswith('['):
        __sec_name = __l.lstrip('[').rstrip(']')
        if __sec_name in self.dkn_layers.values():
          if __sec_name not in __sec_count:
            __sec_count[__sec_name] = 0
          __sec_count[__sec_name] += 1
          __sec_name = '_'.join(
            (__sec_name, str(__sec_count[__sec_name]-1)))
        __secs[__sec_name] = {}
      else:
        __k, __v = __l.split('=')
        __secs[__sec_name][__k] = __v
    return __secs


  def __make_convolution_block(self, conv_dict, in_ch):
    """

    Create a pytorch 2D convolutional block
    from Darknet config.

    A convolutional block consists of a convolution
    layer optionally followed by batchnorm and
    activation.

    Parameters
    ----------
    conv_dict: dict
    Dictionary representing Convolutional
    block parametrization parsed from Darknet
    config file.

    in_ch: int
    Number of input channels.

    Returns
    ----------
    torch.nn.Sequential
    A pytorch 2D convolution block as a Sequential
    container.

    """
    __l = [torch.nn.Conv2d(
      in_channels=in_ch,
      out_channels=int(conv_dict[self.dkn_conv['filters']]),
      kernel_size=int(conv_dict[self.dkn_conv['ksize']]),
      stride=int(conv_dict[self.dkn_conv['stride']]),
      padding=int(conv_dict[self.dkn_conv['padding']]))]
    if (('batch_normalize' in conv_dict) and
        conv_dict['batch_normalize']):
      __l.append(torch.nn.BatchNorm2d(
        num_features=__l.out_channels))
    if (('activation' in conv_dict) and
        conv_dict['activation'] != 'linear'):
      __l.append(self.__make_activation())
    return torch.nn.Sequential(*__l)


  def __make_activation(self, act_name):
    """

    Create a pytorch 2D convolutional layer
    from Darknet config.

    Parameters
    ----------
    act_name: str
    Name of the activation function parsed
    from Darknet config file.

    Returns
    ----------
    torch.nn
    A pytorch activation layer.

    """
    if act_name == 'leaky':
      return torch.nn.LeakyReLU()
    else:
      raise Exception(
        'unknown activation: {}'.format(act_name))


  def __make_maxpool(self, pool_dict):
    """

    Create a pytorch 2D max pooling layer
    from Darknet config.

    Parameters
    ----------
    pool_dict: dict
    Dictionary representing 2D max pooling
    layer parametrization parsed from Darknet
    config file.

    Returns
    ----------
    torch.nn.MaxPool2d
    A pytorch 2D max pooling layer.

    """
    return torch.nn.MaxPool2d(
      kernel_size=int(self.dkn_pool['ksize']),
      stride=int(self.dkn_pool['stride']))


  def __make_upsample(self, up_dict):
    """

    Create a pytorch 2D upsample layer
    from Darknet config.

    Parameters
    ----------
    up_dict: dict
    Dictionary representing upsample
    layer parametrization parsed from Darknet
    config file.

    Returns
    ----------
    torch.nn.Upsample
    A pytorch upsample layer.

    """
    return torch.nn.Upsample(
      scale_factor=float(up_dict[self.dkn_up['factor']]))


  def __make_detection(self):
    pass


  def __make_yolo(self):
    """

    Create pytorch YOLO model.

    Returns
    ----------
    torch.nn.ModuleList
    YOLO model as a pytorch ModuleList.

    """
    for __sec in self.cfg:
      pass


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
