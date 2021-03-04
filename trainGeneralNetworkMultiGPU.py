import os
import sys
import numpy
import random
import time
from datetime import datetime
import tensorflow as tf

import flowInputs as inputs
import fiveLayerNetwork1_2WeightInit as network


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', os.getcwd()+'/trainFiveLayerBNormInitSTD', \
	"""Directory where to write event logs """ \
	"""and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 500001	, \
	"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, \
	"""Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('allow_growth', True, \
	"""Whether to limit memory usage.""")


# cases =["/local/georgioutk/formatChange/eval_2", "/local/georgioutk/formatChange/eval_6"]

def train():
	"""Train for a number of steps."""
	log = open(FLAGS.train_dir+".log", "w", 1)
	cases = os.listdir(FLAGS.trainData_dir)
	for c in range(len(cases)):
		cases[c] = FLAGS.trainData_dir + "/" + cases[c]
	random.shuffle(cases)
	print("Num train examples:\t"+str(len(cases)), file=log)
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)
		# Get examples and ground truth.
		print("Creating Graph..", file=log)
		p, k, nut, u, ground_truth = inputs.inputForces(cases, log, FLAGS.trainBatch_size, useType="train")
		pSmall, kSmall, nutSmall, uSmall, ground_truthSmall = inputs.inputFlow(cases, log, 4*FLAGS.trainBatch_size, useType="train")
		with tf.device('/gpu:0'):
			# Build a Graph that computes the logits predictions from the inference model.			
			code = network.inference([p, k, nut, u], True, log, useType="train")
			forces = network.predictForces(code, log)
			forcesLoss = network.loss(forces, ground_truth, True, "ForceRegressionLoss")
			reconstruction = network.reconstruct(code, True, log, useType="train")
			unProcessed = tf.concat([p, k, nut, u], 4)
			reconLoss = network.loss(reconstruction, unProcessed, True, "ReconstructionLoss")

		with tf.device('/gpu:1'):
			codeSmall = network.inference([pSmall, kSmall, nutSmall, uSmall], False, log, useType="train")
			flow = network.predictFlow(codeSmall, log, useType="train")
			flowLoss = network.loss(flow, ground_truthSmall, False, "FlowLoss")
			reconstructionSmall = network.reconstruct(codeSmall, False, log, useType="train")
			unProcessedSmall = tf.concat([pSmall, kSmall, nutSmall, uSmall], 4)
			reconSmallLoss = network.loss(reconstructionSmall, unProcessedSmall, False, "Reconstruction2Loss")
			#unProcessed = tf.concat([p, k, nut, u], 4)
			#reconLoss = network.loss(reconstruction, unProcessed, True, "ReconstructionLoss")
			# Calculate loss.
			# Build a Graph that trains the model with one batch of examples and
			# updates the model parameters.

		totalLoss = network.totalLoss()

		with tf.device('/gpu:1'):
			train_op = network.train(totalLoss, global_step)

		losses = tf.get_collection("losses")

		#Keep summaries

		tf.summary.scalar(totalLoss.op.name +' (raw)', totalLoss)
		for l in losses:
			# Name each loss as '(raw)' and name the moving average version of the loss
			# as the original loss name.
			tf.summary.scalar(l.op.name +' (raw)', l)
			# tf.summary.scalar(l.op.name, loss_averages.average(l))
		# Create a saver.
		saver = tf.train.Saver(tf.global_variables())

		# Build an initialization operation to run below.
		init = tf.global_variables_initializer()
		#saver2 = tf.train.Saver(tf.all_variables())

		# Start running operations on the Graph.
		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
		myconfig = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
		myconfig.gpu_options.allow_growth = FLAGS.allow_growth
		sess = tf.Session(config=myconfig)
		writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
		sess.run(init)
		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)
		print( "Starting train procedure..", file=log)
		# Build the summary operation based on the TF collection of Summaries.
		_summ = tf.summary.merge_all()
		start_time = time.time()
		for step in range(FLAGS.max_steps):
			fLoss, rLoss, _flowLosss, rSLoss, _losses, loss_value, summ, _ = \
			sess.run([forcesLoss, reconLoss, flowLoss, reconSmallLoss, losses, totalLoss, _summ, train_op])
			assert not numpy.isnan(loss_value), 'Model diverged with loss = NaN'
			writer.add_summary(summ, step)
			if step % 200 == 0:
				duration = float(time.time() - start_time)/200
				examples_per_sec = FLAGS.trainBatch_size / duration
				small_examples_per_sec = 4*FLAGS.trainBatch_size / duration
				format_str = ('%s: progress %2.2f, loss = %.10f (%.1f examples/sec; %.1f small_examples/sec; %.3f sec/batch)')
				print(format_str % (datetime.now(), step/FLAGS.max_steps, loss_value, examples_per_sec, small_examples_per_sec, duration), file=log)
				print(step/FLAGS.max_steps, end="\r", flush=True)
				start_time = time.time()
			# Save the model checkpoint periodically.
			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()


if __name__ == '__main__':
	tf.app.run()
