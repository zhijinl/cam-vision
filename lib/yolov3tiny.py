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
## Last update Mon Dec 10 21:49:42 2018 Zhijin Li
## ---------------------------------------------------------------------------


import torch
import torchvision
import collections
import numpy as np
from .utils import utils as utils


class YOLOConvBlock(torch.nn.Module):
  """

  YOLO convolutional block

  A convolutional block consists of a convolution
  layer optionally followed by batchnorm and
  activation.

  In Darknet, convolution layers do not have bias
  if batch norma is present in the conv block.
  (darknet master-30-oct-2018)

  note:
  When padding is involved, PyTorch and Darknet
  seem to compute the output size in the same manner
  (torch 0.4.1, darknet master-30 oct 2018):
  out_size = (inp_size+2*padding -ksize)/stride+1
  Fpr both frameworks, the same amount of padding
  will be added on both side of each dimension of
  the tensor. Darknet only supports zero-padding.

  There can be two parameters in darknet cfg file
  representing padding:
  1. The `padding` parameter indicates the amount
     of padding to be added to each side of each
     dimension of the tensor.
  2. The `pad` parameter does not indicate the
     amount of padding, but indicates whether
     padding will be performed. When set to a non
     zero value, it invalidates the padding set
     by`padding` parameter. In this case, the amount
     of padding is equal to int(ksize/2).

  In Darknet the epsilon parameter used to avoid
  zero-division during batch norm is set to 1e-6,
  different than 1e-5, the default value used in
  pytorch.

  """

  def __init__(
      self,
      in_channels,
      out_channels,
      ksize,
      stride,
      activation,
      pad,
      batchnorm):
    """

    Constructor

    Parameters
    ----------
    in_channels: int
    Number of input channels.

    out_channels: int
    Number of output channels, i.e. number of
    kernels.

    ksize: int
    Convolution kernel size.

    stride: int
    Convolution stride.

    activation: str
    Name of the activation function, e.g.
    `leaky`, `linear`.

    pad: int
    Number of pixels to pad to each side of
    each dimension of the input tensor.

    batchnorm: bool
    Whether batchnorm will be performed
    between convolution and activation.

    """
    super(YOLOConvBlock, self).__init__()
    self.in_channels  = in_channels
    self.out_channels = out_channels
    self.ksize        = ksize
    self.stride       = stride
    self.pad          = pad
    self.activation   = activation
    self.batchnorm    = batchnorm
    self.conv_block   = self.__make_block()


  def __make_activation(self):
    """

    Create an pytorch activation layer.

    Returns
    ----------
    torch.nn
    A pytorch activation layer.

    """
    if self.activation == 'leaky':
      return torch.nn.LeakyReLU(
        negative_slope=0.1, inplace=False)
    else:
      raise Exception(
        'unknown activation: {}'.format(act_name))


  def __make_block(self):
    """

    Create torch Sequential block with
    convolutional and possibly batchnorm
    & activation.

    Returns
    ----------
    A torch.nn.Sequential representing
    a Darknet convolution block.

    """
    __l = [
      torch.nn.Conv2d(
        in_channels=self.in_channels,
        out_channels=self.out_channels,
        kernel_size=self.ksize,
        stride=self.stride,
        padding=self.pad,
        bias=(not self.batchnorm))]
    if self.batchnorm:
      __l.append(torch.nn.BatchNorm2d(__l[0].out_channels, eps=1e-6))
    if (self.activation and self.activation != 'linear'):
      __l.append(self.__make_activation())
    return torch.nn.Sequential(*__l)


  def forward(self, inp):
    """

    Forward pass for the convolutional block..

    Parameters
    ----------
    inp: torch.tensor
    The input rank-4 torch tensor.

    Returns
    ----------
    torch.tensor
    The output tensor.

    """
    return self.conv_block(inp)


class YOLORoute(torch.nn.Module):
  """

  The YOLO route layer.

  A route layer can take either one or
  two layer index parameters. When a single
  layer index is given, it forwards the
  feature map indexed by that layer. When
  two parameters are given, it outputs the
  concatenated feature maps (in depth
  direction) of the two layers.

  """

  def __init__(self, curr_indx, lay_lst):
    """

    Constructor.

    Parameters
    ----------
    curr_indx: int
    Current layer index.

    lay_lst: list
    List of layer indices for routing.

    """
    super(YOLORoute, self).__init__()
    self.cindx = curr_indx
    self.rindx = self.__get_route_indices(lay_lst)


  def __get_route_indices(self, lay_lst):
    """

    Get the list of indices for routing.

    Parameters
    ----------
    lay_lst: list
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


  def out_channels(self, out_channels_lst):
    """

    Get the number of output channels.

    Parameters
    ----------
    out_channels_lst: list
    A list keeping track of the number of output
    features from each YOLO layer.

    Returns
    ----------
    int
    The number of output channels.

    """
    return np.sum(
      [out_channels_lst[indx] for indx in self.rindx])


  def forward(self, feature_dict):
    """

    Forard pass of the route layer.

    Parameters
    ----------
    feature_dict: dict
    Dictionary where keys are routed layer indices
    and values are corresponding YOLO feature maps.

    Returns
    ----------
    torch.tensor
    The output tensor.

    """
    if len(self.rindx) == 1:
      return feature_dict[self.rindx[0]]
    elif len(self.rindx) == 2:
      return torch.cat(
        (feature_dict[self.rindx[0]],
         feature_dict[self.rindx[1]]),
        dim=1)


class PaddingSame(torch.nn.Module):
  """

  Perform same padding for input tensor.

  This implements the same padding scheme as
  in max pooling layer in Darknet. The amount
  of padding depends on the inp_size, the ksize
  and the pooling stride.

  Darknet always performs the `same` padding
  for max pooling, meaning that when stride = 1,
  the input and output tensor will have the same
  dimension (darknet master-30 oct 2018).

  In Darknet, padding is added to each side
  of each dimension of the input tensor. When
  the amount of padding is not specified in cfg
  file, it is computed as
      padding = ksize-1
  The meaning of the above `padding` is different
  in Darknet than in PyTorch. In Darknet, the
  computed `padding` amount represents the total
  amount of padding to be added to each dimension
  of the input tensor. The amount of padding to be
  added to the beginning of a dimension is
      padding_beg = int(padding/2)
  and the amount of padding to be added to the
  end of a dimension is
      padding_end = padding - padding_beg
  In PyTorch, the `padding` parameter passed to
  max pooling layer indicates the amount of
  padding added to the beginning and the end of
  each dimension of the input tensor.

  The output size of max pooling in Darknet is
  computed as
      (inp_size + padding - size)/stride + 1
  which seems to be different than PyTorch
  (pytorch 0.4.1). Notice that, when stride = 1,
  the output size is strictly equal to the input
  size, since padding = size - 1.


  """
  def __init__(
      self,
      ksize,
      stride,
      dkn_padding=None,
      padding_val=-1e10):
    """

    Constructor.

    Parameters
    ----------
    ksize: int
    Pooling kernel size.

    stride: int
    Pooling stride.

    dkn_padding: int
    Value parsed from the `padding` parameter
    in Darknet max pooling layer.

    padding_val: float
    Value used for padding. Note: for max pooling,
    this should be a really small negative number
    and not 0. Defaults to -1e10.

    """
    super(PaddingSame, self).__init__()
    self.ksize = ksize
    self.stride = stride
    self.dkn_padding = dkn_padding if dkn_padding else ksize-1
    self.padding_val = padding_val


  def forward(self, inp):
    """

    Forward pass of the same padding layer.

    Parameters
    ----------
    inp: torch.tensor
    Input image as torch rank-4 tensor: batch x
    channels x height x width.

    Returns
    ----------
    torch.tensor
    The zero-padded output tensor.

    """
    __inp_size = np.array([inp.shape[2],inp.shape[3]],dtype=int)
    __nops = np.ceil(__inp_size/self.stride)
    __total = self.ksize + (__nops-1)*self.stride - __inp_size
    __beg = int(self.dkn_padding/2)
    __end = (__total - __beg)
    return torch.nn.functional.pad(
      inp, (__beg,int(__end[1]),__beg,int(__end[0])),
      'constant', self.padding_val)


class NearestInterp(torch.nn.Module):
  """

  Nearest neighbor interpolation layer.

  note:
  From the source code, it appears that Darknet uses
  nearest neighbor method for its upsampling layer
  (darknet master-30 oct 2018).

  Internally calls torch.nn.functional.interpolate
  to suppress the warning on calling
  torch.nn.Upsample.

  """
  def __init__(self, factor):
    """
    Constructor.

    Parameters
    ----------
    factor: float
    The interpolation factor.

    """
    super(NearestInterp, self).__init__()
    self.factor = factor


  def forward(self, inp):
    """

    Parameters
    ----------
    inp: torch.tensor
    Input image as torch rank-4 tensor: batch x
    channels x height x width.

    """
    return torch.nn.functional.interpolate(
      inp, scale_factor=self.factor, mode='nearest')


class YOLODetect(torch.nn.Module):
  """

  The YOLO detection lqyer.

  """
  def __init__(self, anchors, classes):
    """

    Constructor.

    Parameters
    ----------
    anchors: list(tuple)
    List of anchor boxes to be used for
    YOLO detection. Each anchor box is
    a size-2 tuple, representing height
    and width in number of pixels with
    respect to the original input image.

    classes: int
    Number of classes.

    """
    super(YOLODetect, self).__init__()
    self.anchors = anchors
    self.classes = classes


  def forward(self, out, img_size):
    """

    Forward pass of the detection layer.

    Parameters
    ----------
    out: torch.tensor
    The output feature map from the previous
    layer.

    img_size: torch.tensor
    The (height x width) of the original input
    image.

    """
    __ratio =torch.FloatTensor(
      [img_size[__i]/out.shape[-2:][__i] for __i in (0, 1)])
    __pred = out.view(
      out.shape[0],len(self.anchors),self.classes+5,-1)
    self.__transform_probabilities(__pred)
    self.__transform_bbox_centers(__pred,out.shape[-2:],__ratio)
    self.__transform_bbox_sizes(__pred)
    return __pred


  def __transform_probabilities(self, pred):
    """

    Transform prediction feature maps to make
    object socres and class probabilities between
    0 and 1.

    Note:
    The final probability of class i used to be
    compared to the threshold is equal to:
    object socre x class probability.

    Parameters
    ----------
    pred: torch.tensor
    The reshaped prediction feature maps with rank 4:
    batch x n_anchors x n_pred x (height*width),
    where n_pred is length of the prediction vector
    for one bounding box, i.e. n_classes+5.

    """
    pred[:,:,4:,:].sigmoid_()


  def __transform_bbox_centers(self, pred, out_size, ratio):
    """

    Transform prediction feature maps to make
    bounding box centers the same scale as the
    input image.

    Parameters
    ----------
    pred: torch.tensor
    The reshaped prediction feature maps with rank 4:
    batch x n_anchors x n_pred x (height*width),
    where n_pred is length of the prediction vector
    for one bounding box, i.e. n_classes+5.

    out_size: torch.tensor
    (height, width) of the prediction feature map.

    ratio: torch.tensor
    The ratio between the original image size and
    the prediction feature map.

    """
    pred[:,:,:2,:].sigmoid_()
    __x, __y = torch.meshgrid(
      [torch.arange(out_size[0]),torch.arange(out_size[1])])
    pred[:,:,0,:] += __y.contiguous().view(1,-1).type(pred.dtype)
    pred[:,:,1,:] += __x.contiguous().view(1,-1).type(pred.dtype)
    pred[:,:,0,:] *= ratio[0]
    pred[:,:,1,:] *= ratio[1]


  def __transform_bbox_sizes(self, pred):
    """

    Transform prediction feature maps to make
    bounding box sizes the same scale as the
    input image.

    Parameters
    ----------
    pred: torch.tensor
    The reshaped prediction feature maps with rank 4:
    batch x n_anchors x n_pred x (height*width),
    where n_pred is length of the prediction vector
    for one bounding box, i.e. n_classes+5.

    """
    pred[:,:,2:4,:].exp_()
    pred[:,:,2:4,:] *= torch.FloatTensor(
      self.anchors).unsqueeze(-1).expand(-1, -1, pred.shape[-1])


class YOLO(torch.nn.Module):
  """

  Class representing YOLO models constructed from
  Darknet config file.

  """

  def __init__(self, cfg_path, nch, weights, set_to_eval=True):
    """

    Constructor

    Parameters
    ----------
    cfg_path: str
    Path to the Darknet config file.

    nch: int
    Number of channels in input image.

    weights: numpy.array
    A 1d numpy array with model weights.

    set_to_eval: bool
    Tag indicating whether the model will be
    set to inference mode systematically.
    Default to True.

    """
    super(YOLO, self).__init__()
    self.cfg = self.__parse_darknet_cfg(cfg_path)
    self.inp_channels = nch
    self.out_chns_lst = []
    self.feature_dict = {}
    self.model = self.__make_yolo()
    self.set_weights(weights)
    if set_to_eval: self.model.eval()


  @property
  def dkn_conv(self):
    """

    Name dictionary for parameters of Darknet
    convolutional layers.

    note:
    There can two parameters in darknet cfg file
    representing padding:
    1. The `padding` parameter indicates the amount
       of padding to be added to each side of the
       tensor.
    2. The `pad` parameter does not indicate the
       amount of padding, but indicates whether
       padding will be performed. When set to a non
       zero value, it invalidates the padding set
       by`padding` parameter. In this case, the amount
       of padding is equal to int(ksize/2).

    """
    return {
      'filters': 'filters',
      'ksize'  : 'size',
      'stride' : 'stride',
      'pad_flag': 'pad',
      'pad_size': 'padding'
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
      'ksize' : 'size',
      'stride': 'stride',
      'padding': 'padding'
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
      'n_cls'   : 'classes',
      'mask'    : 'mask'
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


  def forward(self, inp):
    """

    Forward pass of YOLO detector.

    Parameters
    ----------
    inp: torch.tensor
    Input image as torch rank-4 tensor: batch x
    channels x height x width.

    """
    __out = inp
    detections = []
    for __indx, __lay in enumerate(self.model):
      if (isinstance(__lay, YOLOConvBlock) or
          isinstance(__lay, NearestInterp) or
          isinstance(__lay, torch.nn.Sequential)):
        __out = __lay(__out)
      if isinstance(__lay, YOLORoute):
        __out = __lay(self.feature_dict)
      if isinstance(__lay, YOLODetect):
        __out = __lay(__out, inp.shape[-2:])
        detections.append(__out)
      if __indx in self.feature_dict.keys():
        self.feature_dict[__indx] = __out
    return detections


  def set_weights(self, weights):
    """

    Set YOLO model weights.

    Parameters
    ----------
    weights: np.array
    Array with model weights.

    """
    __c = 0
    __convs = [l for l in self.model if isinstance(l,YOLOConvBlock)]
    for __i, __l in enumerate(__convs):
      if __l.batchnorm:
        __c = self.__set_conv_bn_weights(__l, weights, __c)
      else:
        __c = self.__set_conv_nobn_weights(__l, weights, __c)


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

    Notice that the value type is string.

    """
    __lines = utils.read_txt_as_strs(cfg_path, cmnt='#')
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


  def __set_conv_bn_weights(self, conv, warr, indx):
    """

    Set for conv block with batch norm.

    Parameters
    ----------
    conv: YOLOConvBlock
    The input conv block.

    warr: np.array
    The weight array.

    indx: int
    The index indicating the beginning in
    the weight array to be considered.

    Returns
    ----------
    The incremented new index to be used
    for next call to set weight.

    """
    __b = conv.conv_block
    indx = self.__set_tensor(__b[1].bias        , warr, indx)
    indx = self.__set_tensor(__b[1].weight, warr, indx)
    indx = self.__set_tensor(__b[1].running_mean, warr, indx)
    indx = self.__set_tensor(__b[1].running_var , warr, indx)
    indx = self.__set_tensor(__b[0].weight, warr, indx)
    return indx


  def __set_conv_nobn_weights(self, conv, warr, indx):
    """

    Set for conv block without batch norm.

    Parameters
    ----------
    conv: YOLOConvBlock
    The input conv block.

    warr: np.array
    The weight array.

    indx: int
    The index indicating the beginning in
    the weight array to be considered.

    Returns
    ----------
    The incremented new index to be used
    for next call to set weight.

    """
    __b = conv.conv_block
    indx = self.__set_tensor(__b[0].bias  , warr, indx)
    indx = self.__set_tensor(__b[0].weight, warr, indx)
    return indx


  def __set_tensor(self, ten, warr, indx):
    """

    Set values for a tensor.

    Parameters
    ----------
    ten: torch.tensor
    The input torch tensor.

    warr: np.array
    The weight array.

    indx: int
    The index indicating the beginning in
    the weight array to be considered.

    Returns
    ----------
    The incremented new index to be used
    for next call `to __set_tensor`.

    """
    ten.data.copy_(
      torch.from_numpy(warr[indx:indx+ten.numel()]).view_as(ten))
    return indx + ten.numel()


  def __make_conv_block(self, conv_dict, in_ch):
    """

    Create a pytorch 2D convolutional block
    from Darknet config.

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
    YOLOConvBlock
    A YOLO convolutional block.

    """
    if self.dkn_conv['pad_size'] in conv_dict.keys():
      __pad = int(conv_dict[self.dkn_conv['pad_size']])
    if self.dkn_conv['pad_flag'] in conv_dict.keys():
      __pad = int(int(conv_dict[self.dkn_conv['ksize']])/2)
    __has_bn = (('batch_normalize' in conv_dict) and
                int(conv_dict['batch_normalize']))
    __act = conv_dict['activation']
    return YOLOConvBlock(
      in_ch,
      int(conv_dict[self.dkn_conv['filters']]),
      int(conv_dict[self.dkn_conv['ksize']]),
      int(conv_dict[self.dkn_conv['stride']]),
      conv_dict['activation'] if 'activation' in conv_dict else None,
      __pad,
      __has_bn)


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
    __k = int(pool_dict[self.dkn_pool['ksize']])
    __s = int(pool_dict[self.dkn_pool['stride']])
    return torch.nn.Sequential(
      PaddingSame(__k, __s),
      torch.nn.MaxPool2d(kernel_size=__k, stride=__s, padding=0))


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
    return NearestInterp(
      factor=float(up_dict[self.dkn_up['factor']]))


  def __make_route(self, curr_indx, route_dict):
    """

    Create route layer from Darknet config
    dict.

    Parameters
    ----------
    curr_indx: int
    Current layer index.

    route_dict: dict
    Dictionary with route layer parametrization.

    Returns
    ----------
    YOLORoute
    A route layer object.

    """
    __lay_lst = [
      int(elem) for elem in
      route_dict[self.dkn_route['layers']].split(',')]
    return YOLORoute(curr_indx, __lay_lst)


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
    __cls = int(detect_dict[self.dkn_detect['n_cls']])
    return YOLODetect(__ach, __cls)


  def __make_yolo(self):
    """

    Create pytorch YOLO model.

    Returns
    ----------
    torch.nn.ModuleList
    YOLO model as a pytorch ModuleList.

    """
    __block = None
    __nchs = self.inp_channels
    __yolo = torch.nn.ModuleList()
    for __indx, (__tag, __par) in enumerate(self.cfg[1].items()):
      __name = __tag[:str.find(__tag, '_')]
      if __name.startswith(self.dkn_layers['conv2d']):
        __lay = self.__make_conv_block(__par, __nchs)
        __nchs = __lay.out_channels
      elif __name.startswith(self.dkn_layers['up']):
        __lay = self.__make_upsample(__par)
      elif __name.startswith(self.dkn_layers['skip']):
        __lay = self.__make_route(__indx, __par)
        __nchs = __lay.out_channels(self.out_chns_lst)
        for __rind in __lay.rindx: self.feature_dict[__rind] = None
      elif __name.startswith(self.dkn_layers['maxpool']):
        __lay = self.__make_maxpool(__par)
      elif __name.startswith(self.dkn_layers['detect']):
        __lay = self.__make_detection(__par)
      else:
        raise Exception('unrecognized layer {}.'.format(__tag))
      self.out_chns_lst.append(__nchs)
      __yolo.append(__lay)
    return __yolo
