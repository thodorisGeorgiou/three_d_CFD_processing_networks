import numpy
import tensorflow as tf
import math
import sys
import os
import pickle
#import threading

import fiveLayerNetwork1_2WeightInit as network
import flowInputs as inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/local/georgioutk/threeDsteady/netEval', \
	"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', \
	"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch/georgioutk/threeDsteady/trainFiveLayerBNormInitSTD', \
	"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('architecture', 'FlR', \
	"""Name of saved files""")
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

def eval_once(mean_diff, index, saver, num_examples):
	"""Run Eval once.

	Args:
		saver: Saver.
		summary_writer: Summary writer.
		top_k_op: Top K op.
		summary_op: Summary op.
	"""
	num_examples = num_examples*480
	myconfig = tf.ConfigProto()
	myconfig.gpu_options.allow_growth = FLAGS.allow_growth
	# saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			#saver.restore(sess, "/local/georgioutk/threeDsteady/fiveLayer/trainFiveLayerHBNormForce/model.ckpt-170000")
			print "Variables should be restored from "+ckpt.model_checkpoint_path
		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads=[]
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
			num_iter = int(math.ceil(num_examples / FLAGS.testBatch_size))
			mean = numpy.zeros(shape=(60), dtype=numpy.float32)
			counts = numpy.zeros(shape=(60), dtype=numpy.float32)
			# std = numpy.zeros(shape=(1,12), dtype=numpy.float32)
			step = 0
			global_mean = 0
			examples = []
			while step < num_iter and not coord.should_stop():
				result, indexes = sess.run([mean_diff, index])
				indexes = numpy.reshape(indexes, (FLAGS.testBatch_size))
				numpy.add.at(mean, indexes, result)
				numpy.add.at(counts, indexes, 1)
				global_mean += numpy.mean(result)
				if step % 1000 == 0:
					print str(step)+" / "+str(num_iter)
				step += 1
			mean = mean / counts
			global_mean = global_mean/float(num_iter)
			print "Global average = "+str(global_mean)
			# saveObj("recon_mean_"+FLAGS.architecture+".pkl", mean)
			# saveObj("recon_global_mean_"+FLAGS.architecture+".pkl", global_mean)
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
		# p, k, nut, u = network.inputForcesAllCrops(cases, log)
		p, k, nut, u, index = inputs.inputReconstructionCrops(cases, log, FLAGS.testBatch_size, [36,56,32,6])
		_unProcessed = tf.concat([p, k, nut, u], 4)
		# Build a Graph that computes the logits predictions from the
		# inference model.
		with tf.device('/gpu:0'):
			unProcessed, gt = tf.split(_unProcessed, [24, 12], 1)
			p, k, nut, u = tf.split(unProcessed, [1,1,1,3], 4)
			code = network.inference([p, k, nut, u], True, log)
			# recon = network.reconstruct(code, True, log)
			recon = network.predictFlow(code, True, log)
			# Calculate predictions.
			diffs = tf.subtract(recon, gt)
			mean_diff = tf.reduce_mean(tf.abs(diffs), axis=(1,2,3,4))
		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		for k in variables_to_restore.keys():
			if "popMean" in k or "popVar" in k:
				variables_to_restore[k+"/avg"] = variables_to_restore[k]
				del variables_to_restore[k]
		saver = tf.train.Saver(variables_to_restore)
		print >> log, variables_to_restore
		while True:
			eval_once(mean_diff, index, saver, len(cases))
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
	evaluate()

if __name__ == '__main__':
	tf.app.run()

