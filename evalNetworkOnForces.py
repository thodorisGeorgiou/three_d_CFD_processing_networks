import numpy
import tensorflow as tf
import math
import sys
import os
import pickle
#import threading

import fiveLayerNetwork1_2WeightInit as network


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/prometheus/thodorisGeorgiou/threeDsteady/netEval', \
        """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', \
        """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch/georgioutk/threeDsteady/trainFiveLayerBNormInitSTD', \
        """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, \
        """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True, \
        """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('allow_growth', True, \
        """Whether to limit memory usage.""")


def loadObj(path):
        return pickle.load(open(path, 'rb'))

def saveObj(path, obj):
        pickle.dump(obj, open(path, 'wb'))


def enqueue_func(sess):
	while True:
		enqueue_op = tf.get_collection("enqueue")
		sess.run(enqueue_op)

def eval_once(differences, predictions, gt, saver, num_examples):
        """Run Eval once.

        Args:
                saver: Saver.
                summary_writer: Summary writer.
                top_k_op: Top K op.
                summary_op: Summary op.
        """
        myconfig = tf.ConfigProto()
        myconfig.gpu_options.allow_growth = FLAGS.allow_growth
        # saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                        # Restores from checkpoint
                        saver.restore(sess, ckpt.model_checkpoint_path)
			#saver.restore(sess, "/home/prometheus/thodorisGeorgiou/threeDsteady/fiveLayer/trainFiveLayerBNormForceInitSTD/model.ckpt-500000")
                        print "Variables should be restored from "+ckpt.model_checkpoint_path
                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
            		threads=[]
                        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                        num_iter = int(math.ceil(num_examples / FLAGS.testBatch_size))
                        mean = numpy.zeros(shape=(1,12), dtype=numpy.float32)
                        std = numpy.zeros(shape=(1,12), dtype=numpy.float32)
                        step = 0
                        examples = []
                        while step < num_iter and not coord.should_stop():
                                results = sess.run([differences, predictions, gt])
                               	mean += numpy.abs(results[0])
                               	std += results[0]*results[0]
                                examples.append([results[1], results[2]])
                                if step % 100 == 0:
                                        print str(step)+" / "+str(num_iter)
                                step += 1
                        mean = mean / float(num_iter)
                        std = numpy.sqrt(std / float(num_iter) - mean*mean)
                        examples.append([mean, std])
                        saveObj("testNetworkWeights500k.pkl", examples)
                        # print counts
                        for val in xrange(12):
                                print "Index#"+str(val)+":\t"+str(mean[0,val])+" - "+str(std[0,val])
                except Exception as e:
                        coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)


def evaluate():
        log = open("test.log", "w", 1)
        cases = os.listdir(FLAGS.testData_dir)
        for c in xrange(len(cases)):
                cases[c] = FLAGS.testData_dir + "/" + cases[c]

        """Eval CIFAR-10 for a number of steps."""
        with tf.Graph().as_default() as g:
                # Get images and labels for CIFAR-10.
                eval_data = FLAGS.eval_data == 'test'
                p, k, nut, u, ground_truth = network.inputForces(cases, log)
                # Build a Graph that computes the logits predictions from the
                # inference model.
                with tf.device('/gpu:0'):
                        code = network.inference([p, k, nut, u], True, log)
                        preds = network.predictForces(code, log, useType="test")
	                # Calculate predictions.
       		        diffs = tf.subtract(preds, ground_truth)
                	mean_diff = tf.reduce_mean(tf.abs(diffs))
                # Restore the moving average version of the learned variables for eval.
                variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                # batchVars = tf.train.ExponentialMovingAverage(fourLayerNetwork.MOVING_AVERAGE_DECAY, name='avg')
                # batch_to_restore = batchVars.variables_to_restore()
                for k in variables_to_restore.keys():
                        if "popMean" in k or "popVar" in k:
                                variables_to_restore[k+"/avg"] = variables_to_restore[k]
                                del variables_to_restore[k]
                # for k in batch_to_restore.keys():
                #         if "popMean" in k or "popVar" in k:
                #                 variables_to_restore[k] = batch_to_restore[k]
                #                 print >> log, k
                saver = tf.train.Saver(variables_to_restore)
                print >> log, variables_to_restore
                #sys.exit(0)
                while True:
                        eval_once(diffs, preds, ground_truth, saver, len(cases))
                        if FLAGS.run_once:
                                break
                        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
        evaluate()

if __name__ == '__main__':
        tf.app.run()


# CUDA_VISIBLE_DEVICES=2
