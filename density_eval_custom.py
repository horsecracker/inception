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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import confusion_matrix
from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

import sys
sys.path.append('/scratch1/liliyu/cancer/')

from inception import image_processing
from inception import inception_model as inception
from inception.density_data import DensityData

FLAGS = tf.app.flags.FLAGS


# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")



##########
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

tf.app.flags.DEFINE_integer('batch_size', 48,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

tf.app.flags.DEFINE_boolean('distort', False, """weather distort the image or not""")


# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 1,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)
    pred = tf.nn.softmax(logits)

    top_1_op = tf.nn.in_top_k(logits, labels, 1)

    
    # Calculate predictions.
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    with tf.Session() as sess:
	    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	    if ckpt and ckpt.model_checkpoint_path:
	      if os.path.isabs(ckpt.model_checkpoint_path):
	        # Restores from checkpoint with absolute path.
	        saver.restore(sess, ckpt.model_checkpoint_path)
	      else:
	        # Restores from checkpoint with relative path.
	        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
	                                         ckpt.model_checkpoint_path))

	      # Assuming model_checkpoint_path looks something like:
	      #   /my-favorite-path/imagenet_train/model.ckpt-0,
	      # extract global_step from it.
	      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
	      print('Succesfully loaded model from %s at step=%s.' %
	            (ckpt.model_checkpoint_path, global_step))
	    else:
	      print('No checkpoint file found')
	      return

	    # Start the queue runners.
	    coord = tf.train.Coordinator()
	    try:
	      threads = []
	      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
	        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
	                                         start=True))
	      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
	      # Counts the number of correct predictions.
	      test_acc = 0.0
	      count_top_1=0
	      confusion_m_all=[]
	      total_sample_count = num_iter * FLAGS.batch_size
	      step = 0

	      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
	      start_time = time.time()
	      while step < num_iter and not coord.should_stop():
	        pred, labels, top_1 = sess.run([pred, labels, top_1_op])
                print(pred.shape)
		print(labels.shape)
	        #correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
	        correct_pred=np.equal(np.argmax(pred,1),labels)
		#print (correct_pred)
		test_acc += np.sum(correct_pred.astype(float))

	        confu_m=confusion_matrix(labels,np.argmax(pred,1))#(np.argmax(labels,1), np.argmax(pred,1))
	        confusion_m_all.append(confu_m)
	        #top_1, top_5 = sess.run([top_1_op, top_5_op])
	        count_top_1 += np.sum(top_1)
	        #count_top_5 += np.sum(top_5)
	        step += 1
	        '''
	        if step % 20 == 0:
	          duration = time.time() - start_time
	          sec_per_batch = duration / 20.0
	          examples_per_sec = FLAGS.batch_size / sec_per_batch
	          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
	                'sec/batch)' % (datetime.now(), step, num_iter,
	                                examples_per_sec, sec_per_batch))
	          start_time = time.time()
	        '''

	      # Compute precision @ 1

              '''
	      precision_at_1 = count_top_1 / total_sample_count
	      #recall_at_5 = count_top_5 / total_sample_count
	      print('%s: precision @ 1 = %.4f  [%d examples]' %
	            (datetime.now(), precision_at_1, total_sample_count))
	      '''
	      print(confusion_m_all.shape)
	      exit()
	      confusion_m_average=np.sum(confusion_m_all, axis=0)
	      print(confusion_m_average)

	      test_acc = float (test_acc) / float(total_sample_count)
	      print ("Test Accuracy: {} \n".format(test_acc))


	      summary = tf.Summary()
	      summary.ParseFromString(sess.run(summary_op))
	      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
	      #summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
	      summary_writer.add_summary(summary, global_step)

	    except Exception as e:  # pylint: disable=broad-except
	      coord.request_stop(e)

	    coord.request_stop()
	    coord.join(threads, stop_grace_period_secs=10)



def main(unused_argv=None):
  dataset = DensityData(subset=FLAGS.subset)
  assert dataset.data_files()
  #if tf.gfile.Exists(FLAGS.eval_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  #tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
