import os
import sys
import numpy
import random
import time
from datetime import datetime
import tensorflow as tf

import flowInputs as inputs
import fiveLayerNetwork1_2WeightInit as network

# orStdOut = sys.stdout
# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")
# sys.stdwar = open(os.devnull, "w")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

release_dir = os.getcwd()+'/trainFiveLayerBNormInitSTD'
MOVING_AVERAGE_DECAY = 0.9999
FLAGS = tf.app.flags.FLAGS
cases = os.listdir(FLAGS.trainData_dir)
for c in range(len(cases)):
	cases[c] = FLAGS.trainData_dir + "/" + cases[c]

testCases = os.listdir(FLAGS.testData_dir)
for c in range(len(testCases)):
	testCases[c] = FLAGS.testData_dir + "/" + testCases[c]

cases += testCases
log = open("test.log", "w", 1)
# Get examples and ground truth.
print("Creating Graph..", file=log)
p, k, nut, u, ground_truth = inputs.inputForces(cases, log, FLAGS.testBatch_size, useType="test")
# pSmall, kSmall, nutSmall, uSmall, ground_truthSmall = inputs.inputFlow(cases, log, 4*FLAGS.trainBatch_size, useType="train")
with tf.device('/gpu:0'):
	# Build a Graph that computes the logits predictions from the inference model.			
	code = network.inference([p, k, nut, u], True, log, useType="test")

# with tf.device('/gpu:1'):
# 	codeSmall = network.inference([pSmall, kSmall, nutSmall, uSmall], False, log, useType="train")

variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()

saver = tf.train.Saver(variables_to_restore)

init = tf.global_variables_initializer()
myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
sess.run(init)
tf.train.start_queue_runners(sess=sess)
writer = tf.summary.FileWriter(release_dir, sess.graph)
ckpt = tf.train.get_checkpoint_state(release_dir)
#if ckpt and ckpt.model_checkpoint_path:
	# Restores from checkpoint
#	print("Model path:\n{}".format(ckpt.model_checkpoint_path))
#	saver.restore(sess, ckpt.model_checkpoint_path)


#To change the layer from which we extract features change the index bellow.
#Index 0  - 3D shape: 40, 28, 16. Num features: 224
#Index 1  - 3D shape: 40, 28, 16. Num features: 128
#Index 2  - 3D shape: 40, 28, 16. Num features: 128
#Index 3  - 3D shape: 20, 14, 8.  Num features: 128
activations = tf.get_collection("activations")[3]

# sys.stdout = orStdOut
features = []
count = 0
while True:
	if count % 100 == 0: print("%.2f" % (count/14238), end="\r", flush=True)
	if count == 14238: break
	count += 1
	feats = sess.run(activations)
	features.append(feats)

#Features has shape (N, X, Y, Z, M), where:
#N: Number of simulations
#X, Y, Z: 3D space resolution
#M: Number of features per point in space
features = numpy.concatenate(features, axis=0)
print(features.shape)
exit(0)
numpy.save("networkActivations.npy", features)
