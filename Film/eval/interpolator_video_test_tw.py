# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --video_path <filepath of the input video> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input video>/output_video.mp4. If
`--output_video` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
import numpy as np
import cv2
import mediapy as media
from absl import logging

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_VIDEO = flags.DEFINE_string(
    name='video_path',
    default=None,
    help='The filepath of the input video.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_OUTPUT_VIDEO = flags.DEFINE_string(
    name='output_video',
    default=None,
    help='The output filepath of the interpolated video.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')


def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

  vidcap = cv2.VideoCapture(_VIDEO.value)
  success, image = vidcap.read()
  fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
  
  output_video_frames_list = [image]
  count = 0
  while success:
    # First batched image.
    img_rgb_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    image_1 = np.float32(img_rgb_1 / 255.0)
    image_batch_1 = np.expand_dims(image_1, axis=0)
    
    success, image = vidcap.read()

    if success:
      # Second batched image.
      img_rgb_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
      image_2 = np.float32(img_rgb_2 / 255.0)
      image_batch_2 = np.expand_dims(image_2, axis=0)
      
      # Batched time.
      batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
      
      # Invoke the model for one mid-frame interpolation.
      mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]
      
      output_video_frames_list.append(mid_frame)
      output_video_frames_list.append(image)
      
      logging.info('interpolator frame number %s.', str(count))
      count += 1

  media.write_video(f'{_OUTPUT_VIDEO.value}', output_video_frames_list, fps= fps * 2)
  logging.info('Output video saved at %s.', _OUTPUT_VIDEO.value)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ffmpeg_path = util.get_ffmpeg_path()
  media.set_ffmpeg(ffmpeg_path)

  _run_interpolator()


if __name__ == '__main__':
  app.run(main)

