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
## Last update Sun Nov 18 17:55:15 2018 Zhijin Li
## ---------------------------------------------------------------------------


import torch
import torchvision
import collections
from .utils import utils as utils


class YOLORoute(torch.nn.Module):
  """

  The YOLO route layer.

  A route layer can take either one or
  two layer index parameters. When a single
  layer index is given, it forwards the
  feature map indexed by that layer. When
  two parameters are given, it outputs the
  concatenated feature maps (in depth
  direction)of the two layers.

  """

  def __init__(self, curr_indx, lay_lst, features):
    """

    Constructor.

    Parameters
    ----------
    curr_indx: int
    Current layer index.

    layer_lst: list
    List of layer indices for routing.

    features: dict
    A dictionary where keys are layer indices
    and values are output feature maps
    of the layer.

    """
    self.feats = features
    self.cindx = curr_indx
    self.rindx = self.__get_route_indices(lay_lst)
    self.out_channels = self.__get_out_channels()


  def __get_route_indices(self, lay_lst):
    """

    Get the list of indices for routing.

    Parameters
    ----------
    layer_lst: list
    List of layer indices for routing.

    Returns
    ----------
    list
    List of size 1 or 2, representing the
    indices of the layers for routing.

    """
    if (len(lay_lst) != 1 and len(lay_lst) != 2):
      raise Exception(
        'layer index list must have length 1 or 2.')

    return [elem if elem >= 0 else self.cindx+elem
            for elem in lay_lst]


  def __get_out_channels(self):
    """

    Get the number of output channels.

    Returns
    ----------
    int
    The number of output channels.

    """
    return np.sum(
      [self.feats[indx].shape[1] for indx in self.rindx])


  def forward(self):
    """

    Forard pass of the route layer.

    Returns
    ----------
    torch.tensor
    The output tensor.

    """
    if len(self.rindx) == 1:
      return self.feats[self.rindx[0]]
    elif len(self.rindx) == 2:
      return torch.cat(
        (self.feats[self.rindx[0]],
         self.feats[self.rindx[1]]),
        dim=1)


class YOLODetect(torch.nn.Module):
  """

  The YOLO detection lqyer.

  """
  def __init__(self, anchors):
    super(YOLODetect, self).__init_()
    self.anchors = anchors

  def forward(self, inp):
    pass


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
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(self.cfg)


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
  def dkn_route(self):
    """

    Name dictionary for parameters of Darknet
    route layer.

    """
    return {
      'layers': 'layers'
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
  def dkn_detect(self):
    """

    Name dictionary for parameters of Darknet
    yolo detection layer.

    """
    return {
      'anchors' : 'anchors',
      'n_cls' : 'classes'
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
      self.dnk_layers['conv2d']  : self.__make_conv_block,
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
    list(dict)
    A list of two dictionaries.
    - The first dict contains the network's
      configuration for training and inference.
    - The second dict is ordered. Each key refers to
      a section of the config file (such as
      'convolution') and the corresponding
      value is a dict representing the section
      configuration.

    """
    __secs = [{}, collections.OrderedDict()]
    __sec_name = None
    __sec_count = {}
    __switch = 0
    for __l in cfg_lines:
      if __l.startswith('['):
        __sec_name = __l.lstrip('[').rstrip(']')
        if __sec_name in self.dkn_layers.values():
          __switch = 1
          if __sec_name not in __sec_count:
            __sec_count[__sec_name] = 0
          __sec_count[__sec_name] += 1
          __sec_name = '_'.join(
            (__sec_name, str(__sec_count[__sec_name]-1)))
        else:
          __switch = 0
        __secs[__switch][__sec_name] = {}
      else:
        __k, __v = __l.split('=')
        __secs[__switch][__sec_name][
          __k.strip()] = __v.strip()
    return __secs


  def __make_conv_block(self, conv_dict, in_ch):
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
      return torch.nn.LeakyReLU(
        negative_slope=0.1, inplace=False)
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
      mode='bilinear',
      scale_factor=float(up_dict[self.dkn_up['factor']]))


  def __make_route(self, route_dict):
    """

    Create route layer from Darknet config
    dict.

    Parameters
    ----------
    route_dict: dict
    Dictionary with route layer parametrization.

    Returns
    ----------
    YOLORoute
    A route layer object.

    """



  def __make_detection(self, detect_dict):
    """

    Create a pytorch YOLO detection layer
    from Darknet config.

    Parameters
    ----------
    detect_dict: dict
    Dictionary representing Detection
    layer parametrization parsed from Darknet
    config file.

    Returns
    ----------
    YOLODetect
    A pytorch YOLODetect layer.

    """
    __all = [int(elem) for elem in detect_dict[
      self.dkn_detect['anchors']].split(',')]
    __msk = [int(elem) for elem in detect_dict[
      self.dkn_detect['mask']].split(',')]
    __ach = [(__all[2*elem], __all[2*elem+1]) for elem in __msk]
    return YOLODetect(__ach)


  def __make_yolo(self):
    """

    Create pytorch YOLO model.

    Returns
    ----------
    torch.nn.ModuleList
    YOLO model as a pytorch ModuleList.

    """
    __yolo = torch.nn.ModuleList()
    for __lay, __par in self.cfg[1].items():
      __name = __lay[:str.find(__lay, '_')]
      __maker = self.dkn_layer_makers[__name]
      __yolo.append(__maker(__par))
    return __yolo


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
