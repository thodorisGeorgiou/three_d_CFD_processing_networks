import numpy
import random
import os
import time
from datetime import datetime
import tensorflow as tf

import forceRegressionMLP1_3 as network

max_steps = 500000
batch_size = 128
trainData_dir = "/scratch/georgioutk/threeDsteady/trainSet"
testData_dir = "/scratch/georgioutk/threeDsteady/testSet"
train_dir = "/scratch/georgioutk/threeDsteady/MLP1_3_128"

def evaluate():
	log = open("evalFC.log", "w", 1)

	# features, labels = network.inputFeatures(cases, batch_size, False)
	featuresPlaceholder = tf.placeholder(tf.float32, shape=(1,64))
	labelsPlaceholder = tf.placeholder(tf.float32, shape=(6))

	predictions = network.predictForces(featuresPlaceholder, log, batch_size)
	diffs = tf.subtract(tf.reshape(predictions, (6,)), labelsPlaceholder)

	variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
	variables_to_restore = variable_averages.variables_to_restore()
	saver = tf.train.Saver(variables_to_restore)

	myconfig = tf.ConfigProto()
	sess = tf.Session(config=myconfig)
	ckpt = tf.train.get_checkpoint_state(train_dir)
	if ckpt and ckpt.model_checkpoint_path:
		# Restores from checkpoint
		saver.restore(sess, ckpt.model_checkpoint_path)

	means = numpy.zeros(shape=(6), dtype=numpy.float32)
	minn = numpy.zeros(shape=(6), dtype=numpy.float32)
	minn[:] = numpy.finfo(numpy.float32).max
	maxx = numpy.zeros(shape=(6), dtype=numpy.float32)
	maxx[:] = numpy.finfo(numpy.float32).min
	cases = os.listdir(testData_dir)
	print "about to get error"
	for c in cases:
		path = testData_dir + "/" + c
		features = numpy.fromfile(path+"/aggregatedVC_ForceTrainedActivations.raw", dtype=numpy.float32)
		features = numpy.reshape(features, (1,64))
		labels = numpy.take(numpy.loadtxt(path+"/forcesLast.dat", dtype=numpy.float32), [0,1,2,6,7,8])
		differences = sess.run(diffs, feed_dict={featuresPlaceholder:features, labelsPlaceholder:labels})
		means += numpy.abs(differences)
		minn[differences<minn] = differences[differences<minn]
		maxx[differences>maxx] = differences[differences>maxx]


	print maxx
	print minn
	means /= len(cases)
	# print means
	rangee = maxx-minn
	# print rangee
	means /= rangee
	print "Means: "+str(means)
	print numpy.mean(means)



def main(argv=None):  # pylint: disable=unused-argument
	evaluate()

if __name__ == '__main__':
	tf.app.run()
