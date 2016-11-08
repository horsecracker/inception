# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/scratch1/liliyu/cancer/')

import tensorflow as tf

from inception import inception_eval
from inception.density_data import DensityData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch1/liliyu/cancer/inception/train_output/distortion/',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 1015,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")


def main(unused_argv=None):
  dataset = DensityData(subset=FLAGS.subset)
  assert dataset.data_files()
  '''
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  '''
  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
