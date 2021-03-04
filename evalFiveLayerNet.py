import numpy
import random
import os
import time
from datetime import datetime
import tensorflow as tf

import fiveLayerNetwork1_2WeightInit as network

trainData_dir = "/scratch/georgioutk/threeDsteady/trainSet"
testData_dir = "/scratch/georgioutk/threeDsteady/testSet"
train_dir = "/tank/carFlowFields/steady/sets/networks/trainFiveLayerBNormForceInitSTD2ndRun"
batch_size = 1


def evaluate():
	log = open("evalFC.log", "w", 1)

	# features, labels = network.inputFeatures(cases, batch_size, False)
	flowPlaceholder = tf.placeholder(tf.float32, shape=(80,56,32,6))
	labelsPlaceholder = tf.placeholder(tf.float32, shape=(12))

	p, k, nut, u = tf.split(flowPlaceholder, [1,1,1,3], 3)
	u = network.normalizeTensor(u)
	p = network.normalizeTensor(p)
	k = network.normalizeTensor(k)
	nut = network.normalizeTensor(nut)
	u = tf.reshape(u, (1,80,56,32,3))
	p = tf.reshape(p, (1,80,56,32,1))
	k = tf.reshape(k, (1,80,56,32,1))
	nut = tf.reshape(nut, (1,80,56,32,1))
	code = network.inference([p,k,nut,u], True, log)
	predictions = network.predictForces(code, log, "test")
	diffs = tf.subtract(tf.reshape(predictions, (12,)), labelsPlaceholder)

	variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
	variables_to_restore = variable_averages.variables_to_restore()
	for k in variables_to_restore.keys():
		if "popMean" in k or "popVar" in k:
			variables_to_restore[k+"/avg"] = variables_to_restore[k]
			del variables_to_restore[k]

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
		flow = numpy.fromfile(path+"/allInOne.raw", dtype=numpy.float32)
		flow = numpy.reshape(flow, (96,64,32,6))
		flow = flow[10:90, 3:59, :, :]
		labels = numpy.loadtxt(path+"/forcesLast.dat", dtype=numpy.float32)
		differences = sess.run(diffs, feed_dict={flowPlaceholder:flow, labelsPlaceholder:labels})
		differences = numpy.take(differences, [0,1,2,6,7,8])
		means += numpy.abs(differences)
		minn[differences<minn] = differences[differences<minn]
		maxx[differences>maxx] = differences[differences>maxx]

	means /= len(cases)
	rangee = maxx-minn
	means /= rangee
	print "Means: "+str(means)
	print numpy.mean(means)



def main(argv=None):  # pylint: disable=unused-argument
	evaluate()

if __name__ == '__main__':
	tf.app.run()
