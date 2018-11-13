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
## File: mobilenet.py for Cam-Vision
##
## Created by Zhijin Li
## E-mail:   <jonathan.zj.lee@gmail.com>
##
## Started on  Sun Oct 28 21:06:23 2018 Zhijin Li
## Last update Mon Oct 29 22:34:54 2018 Zhijin Li
## ---------------------------------------------------------------------------


import numpy as np
import tensorflow as tf


def __load_mobilenet_base(width_mult):
  """

  Load MobileNet with weights pretrained
  ImageNet.

  width_mult: float
  Width multiplier used for decreasing the
  number of filters in conv layers.

  Returns
  ----------
  tf.keras.models.Model
  Loaded MobileNet model.

  """
  res = tf.keras.applications.mobilenet.MobileNet(
    include_top=True,
    weights='imagenet',
    alpha=width_mult,
    pooling='avg')
  return res


def __load_mobilenet_empty(width_mult, pooling):
  """

  Load MobileNet without weights.

  Parameters
  ----------
  width_mult: float
  Width multiplier used for decreasing the
  number of filters in conv layers.

  pooling: str
  The final global pooling type. Either
  `global_avg` or `global_max`.

  Returns
  ----------
  tf.keras.models.Model
  Loaded MobileNet model.

  """
  if pooling not in ['global_avg', 'global_max']:
    raise AttributeError(
      'pooling must be global_avg or global_max.')

  res = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(None,None,3),
    include_top=False,
    weights=None,
    alpha=width_mult,
    depth_multiplier=1, # Current keras impl seems to only works for 1.
    pooling=pooling.split('_')[-1])
  return res


def __get_global2dpooling(model):
  """

  Get the first global 2D pooling layer.

  Parameters
  ----------
  model: tf.keras.models.Model
  The input keras model.

  Returns
  ----------
  The first global pooling layer.

  """
  __lay = None
  for __l in model.layers:
    if (isinstance(__l,tf.keras.layers.GlobalMaxPooling2D) or
        isinstance(__l,tf.keras.layers.GlobalAveragePooling2D)):
      __lay = __l
  return __lay


def __customize_mobilenet(base_model, empty_model):
  """

  Customize MobileNet for inference mode with
  arbitrary input image size.

  Parameters
  ----------
  base_model: tf.keras.models.Model
  The input base model with weights pretrained
  on ImageNet.

  empty_model: tf.keras.models.Model
  The input empty model without pretrained weights,
  configured for arbitrary input image size.

  Returns
  ----------
  tf.keras.models.Model
  The customized model, with weights copied
  from the base model.

  """
  intercept = empty_model.output
  __s = empty_model.layers[-1].get_output_shape_at(0)[-1]
  intercept = tf.keras.layers.Reshape((1,1,__s))(intercept)
  intercept = tf.keras.layers.Conv2D(
    name='prediction', filters=1000, kernel_size=(1,1),
    strides=(1,1), padding='same',activation='softmax')(
      intercept)
  intercept = tf.keras.layers.Flatten()(intercept)
  model = tf.keras.models.Model(
    inputs=empty_model.input, outputs=intercept)

  for idx in range(len(empty_model.layers)):
    model.layers[idx].set_weights(
      base_model.layers[idx].get_weights())
  __w = base_model.get_layer('conv_preds').get_weights()
  model.get_layer('prediction').set_weights(__w)
  return model


def load_mobilenet_anysize(width_mult, pooling):
  """

  Load a MobileNet for images of any size.

  Copy weights of a loaded base MobileNet
  to an anysize empty MobileNet and return it.

  Parameters
  ----------
  width_mult: float
  Width multiplier used for decreasing the
  number of filters in conv layers.

  pooling: str
  The final global pooling type. Either
  `global_avg` or `global_max`.

  Returns
  ----------
  tf.keras.models.Model
  Loaded MobileNet model.

  """
  print(
    ('\nloading MobileNet with\n'
     '- {:25s}: {}\n- {:25s}: {}\n').format(
       'width multiplier', width_mult,
       'pooling type', pooling))
  __base = __load_mobilenet_base(width_mult)
  __res = __load_mobilenet_empty(width_mult,pooling)
  __res = __customize_mobilenet(__base,__res)
  print('model loaded.')
  return __res
