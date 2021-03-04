import os
import numpy
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('trainBatch_size', 4, \
	"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('testBatch_size', 1, \
	"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('trainData_dir', "/tank/carFlowFields/steady/sets/trainSet", \
	"""Path to the train data directory.""")
tf.app.flags.DEFINE_string('testData_dir', "/tank/carFlowFields/steady/sets/testSet", \
	"""Path to the test data directory.""")

# RESOLUTIONS = [32, 64, 128]
bn_decay = 0.999
bn_epsilon = 1e-9
RESOLUTIONS = [32]
NUM_OUTPUTS = 12
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001     # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
  	Args:
  		name: name of the variable
  		shape: list of ints
  		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return var


def _variable(name, shape, initializer, trainable=True):
	"""Helper to create a Variable stored on CPU memory.
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	if trainable:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	else:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd, cpu=False):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	if cpu:
		var = _variable_on_cpu(name, shape, \
			tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	else:
		var = tf.get_variable(name, shape, \
			initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv(inpt, kernel, biases, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="VALID", group=1):
	'''From https://github.com/ethereon/caffe-tensorflow
	'''
	c_i = inpt.get_shape()[-1]
	assert c_i%group==0
	assert c_o%group==0
	convolve = lambda i, k: tf.nn.conv3d(i, k, [1, s_h, s_w, s_d, 1], padding=padding)
	if group==1:
		conv = convolve(inpt, kernel)
	else:
		input_groups = tf.split(inpt, num_or_size_splits=group, axis=4)
		kernel_groups = tf.split(kernel, num_or_size_splits=group, axis=4)
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 4)
	return  tf.nn.bias_add(conv, biases)

def unPool3D(x, depth_factor, height_factor, width_factor):
	output = repeat_elements(x, depth_factor, axis=1)
	output = repeat_elements(output, height_factor, axis=2)
	output = repeat_elements(output, width_factor, axis=3)
	return output

def repeat_elements(x, rep, axis):
	x_shape = x.get_shape().as_list()
	if x_shape[axis] is None:
		raise ValueError('Axis ' + str(axis) + ' of input tensor '
										 'should have a defined dimension, but is None. '
										 'Full tensor shape: ' + str(tuple(x_shape)) + '. '
										 'Typically you need to pass a fully-defined '
										 '`input_shape` argument to your first layer.')
	# slices along the repeat axis
	splits = tf.split(x, num_or_size_splits=x_shape[axis], axis=axis)
	# repeat each slice the given number of reps
	x_rep = [s for s in splits for _ in range(rep)]
	return tf.concat(x_rep, axis)

def getWeightsNBiases(first, inptShape, k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope):
	if first:
		stddev=numpy.sqrt(2.0 / (c_i*k_h*k_w*k_d))
		convW = _variable_with_weight_decay('weights', shape=[k_h, k_w, k_d, c_i, c_o], stddev=stddev, wd=0.0001)
		convb = _variable('biases', [c_o], tf.constant_initializer(0.0))
	else:
		convW = tf.get_variable("weights")
		convb = tf.get_variable("biases")
	return convW, convb

def batch_norm(inpt, useType='test', reuse=False):
	dims = inpt.get_shape().as_list()[-1]
	if reuse:
		scale = tf.get_variable("scale")
		beta = tf.get_variable("beta")
		pop_mean = tf.get_variable("popMean")
		pop_var = tf.get_variable("popVariance")
	else:
		scale = _variable('scale', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		beta = _variable('beta', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		pop_mean = _variable('popMean', dims, tf.constant_initializer(0.0), trainable=False)
		pop_var = _variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(inpt, axes=[i for i in range(len(inpt.get_shape().as_list())-1)])
		if not reuse:
			train_mean = tf.assign(pop_mean, pop_mean * bn_decay + batch_mean * (1 - bn_decay))
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inpt, batch_mean, batch_var, beta, scale, bn_epsilon)
		else:
			return tf.nn.batch_normalization(inpt, batch_mean, batch_var, beta, scale, bn_epsilon)
	else:
		return tf.nn.batch_normalization(inpt, pop_mean, pop_var, beta, scale, bn_epsilon)

def scalarModSpecificConv(inpt, modality, first=True, useType="test", varName="PKNut", reuseBatchNorm=False):
	#conv1
	with tf.variable_scope(varName+"Conv1", reuse=(not first)) as scope:
		k_h = 3; k_w = 3; k_d = 3; c_o = 16; c_i = 1; s_h = 1; s_w = 1; s_d = 1
		conv1W, conv1b = getWeightsNBiases(first, inpt.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
	with tf.variable_scope(modality+"Conv1", reuse=reuseBatchNorm) as scope:	
		conv1_in = conv(inpt, conv1W, conv1b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
		conv1_norm = batch_norm(conv1_in, useType, reuseBatchNorm)
		conv1 = tf.nn.leaky_relu(conv1_norm, alpha=0.1)

	#conv2
	with tf.variable_scope(varName+"Conv2", reuse=(not first)) as scope:
		k_h = 3; k_w = 3; k_d = 3; c_o = 32; c_i = 16; s_h = 1; s_w = 1; s_d = 1
		conv2W, conv2b = getWeightsNBiases(first, conv1.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
	with tf.variable_scope(modality+"Conv2", reuse=reuseBatchNorm) as scope:	
		conv2_in = conv(conv1, conv2W, conv2b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
		conv2_norm = batch_norm(conv2_in, useType, reuseBatchNorm)
		conv2 = tf.nn.leaky_relu(conv2_norm, alpha=0.1)

	#maxpool
	with tf.variable_scope(modality+"maxPool") as scope:
		k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
		maxpool = tf.nn.max_pool3d(conv2, ksize=[1, k_h, k_w, k_d, 1], strides=[1, s_h, s_w, s_d, 1], padding=padding, name="maxPool1")
	return maxpool

def vectorModSpecificConv(inpt, modality, first=True, useType="test", reuseBatchNorm=False):
	#conv1
	with tf.variable_scope("uConv1", reuse=(not first)) as scope:
		k_h = 3; k_w = 3; k_d = 3; c_o = 64; c_i = 3; s_h = 1; s_w = 1; s_d = 1
		conv1W, conv1b = getWeightsNBiases(first, inpt.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
	with tf.variable_scope(modality+"Conv1", reuse=reuseBatchNorm) as scope:	
		conv1_in = conv(inpt, conv1W, conv1b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
		conv1_norm = batch_norm(conv1_in, useType, reuseBatchNorm)
		conv1 = tf.nn.leaky_relu(conv1_norm, alpha=0.1)

	#conv2
	with tf.variable_scope("uConv2", reuse=(not first)) as scope:
		k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 64; s_h = 1; s_w = 1; s_d = 1
		conv2W, conv2b = getWeightsNBiases(first, conv1.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
	with tf.variable_scope(modality+"Conv2", reuse=reuseBatchNorm) as scope:	
		conv2_in = conv(conv1, conv2W, conv2b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
		conv2_norm = batch_norm(conv2_in, useType, reuseBatchNorm)
		conv2 = tf.nn.leaky_relu(conv2_norm, alpha=0.1)

	#maxpool
	with tf.variable_scope(modality+"maxPool") as scope:
		k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
		maxpool = tf.nn.max_pool3d(conv2, ksize=[1, k_h, k_w, k_d, 1], strides=[1, s_h, s_w, s_d, 1], padding=padding, name="maxPool1")
	
	return maxpool

def inference(examples, first, log, useType="test"):
	with tf.variable_scope("Encoder") as scope:
		with tf.variable_scope("ScalarConvolutions") as scope:	
			#P specific layers
			convP = scalarModSpecificConv(examples[0], "p", first, useType, reuseBatchNorm=not first)
			#k specific layers
			convK = scalarModSpecificConv(examples[1], "k", False, useType, reuseBatchNorm=not first)
			#nut specific layers
			convNut = scalarModSpecificConv(examples[2], "nut", False, useType, reuseBatchNorm=not first)
		with tf.variable_scope("VectorConvolutions") as scope:
			#U specific layers
			convU = vectorModSpecificConv(examples[3], "u", first, useType, reuseBatchNorm=not first)
	
		fusionIn = tf.concat([convP, convK, convNut, convU], 4)
		tf.add_to_collection("activations", fusionIn)
		#conv3
		with tf.variable_scope("Conv3", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 128+3*32; s_h = 1; s_w = 1; s_d = 1
			conv3W, conv3b = getWeightsNBiases(first, fusionIn.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv3_in = conv(fusionIn, conv3W, conv3b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv3_norm = batch_norm(conv3_in, useType, not first)
			conv3 = tf.nn.leaky_relu(conv3_norm, alpha=0.1)
			tf.add_to_collection("activations", conv3)

		#conv4
		with tf.variable_scope("Conv4", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			conv4W, conv4b = getWeightsNBiases(first, conv3.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv4_norm = batch_norm(conv4_in, useType, not first)
			conv4 = tf.nn.leaky_relu(conv4_norm, alpha=0.1)
			tf.add_to_collection("activations", conv4)

		#maxpool1
		with tf.variable_scope("MaxPool1") as scope:
			k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
			maxpool1 = tf.nn.max_pool3d(conv4, ksize=[1, k_h, k_w, k_d, 1], strides=[1, s_h, s_w, s_d, 1], padding=padding, name="maxPool1")

		#conv5
		with tf.variable_scope("Conv5", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			conv5W, conv5b = getWeightsNBiases(first, maxpool1.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv5_in = conv(maxpool1, conv5W, conv5b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv5_norm = batch_norm(conv5_in, useType, not first)
			conv5 = tf.nn.leaky_relu(conv5_norm, alpha=0.1)
			tf.add_to_collection("activations", conv5)
	return conv5

def predictForces(code, log, useType="train", first=True):
	if useType == "train":
		batch_size = FLAGS.trainBatch_size
	else:
		batch_size = FLAGS.testBatch_size	
	with tf.variable_scope("MLPRegressor") as scope:
		#maxpool2
		with tf.variable_scope("MaxPool2") as scope:
			k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
			maxpool2 = tf.nn.max_pool3d(code, ksize=[1, k_h, k_w, k_d, 1], strides=[1, s_h, s_w, s_d, 1], padding=padding, name="maxPool1")

		#fc4
		with tf.variable_scope("fc4") as scope:
			reshape = tf.reshape(maxpool2, [batch_size, -1])
			dim = reshape.get_shape()[1].value
			print("Before fully conected, shape = "+str(maxpool2.get_shape().as_list()[1:]), file=log)
			print("Before fully conected, reshape = "+str(dim), file=log)
			stddev=numpy.sqrt(2 / numpy.prod(maxpool2.get_shape().as_list()[1:]))
			fc4W = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.01, wd=0.0001)
			fc4b = _variable('biases', [1024], tf.constant_initializer(1.0))
			fc4 = tf.nn.leaky_relu(tf.matmul(reshape, fc4W) + fc4b, name=scope.name)

		#fc5
		with tf.variable_scope("fc5") as scope:
			stddev=numpy.sqrt(2 / numpy.prod(fc4.get_shape().as_list()[1:]))
			fc5W = _variable_with_weight_decay('weights', shape=[1024, 1024], stddev=0.01, wd=0.0001)
			fc5b = _variable('biases', [1024], tf.constant_initializer(1.0))
			fc5 = tf.nn.leaky_relu(tf.matmul(fc4, fc5W) + fc5b, name=scope.name)

		#fc6
		with tf.variable_scope("fc6") as scope:
			stddev=numpy.sqrt(2 / numpy.prod(fc5.get_shape().as_list()[1:]))
			fc6W = _variable_with_weight_decay('weights', shape=[1024, NUM_OUTPUTS], stddev=0.01, wd=0.0)
			fc6b = _variable('biases', [NUM_OUTPUTS], tf.constant_initializer(0.0))
			output = tf.add(tf.matmul(fc5, fc6W), fc6b, name=scope.name)

	return output

def predictFlow(code, log, useType="test"):
	with tf.variable_scope("flowPrediction"):	
		#up1
		with tf.variable_scope("UpSample1") as scope:
			up1 = unPool3D(code, 2, 2, 2)
			
		#conv6
		with tf.variable_scope("Conv6") as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 64; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			conv6W, conv6b = getWeightsNBiases(True, up1.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv6_in = conv(up1, conv6W, conv6b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv6_norm = batch_norm(conv6_in, useType)
			conv6 = tf.nn.relu(conv6_norm)

		#conv7
		with tf.variable_scope("Conv7") as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 32; c_i = 64; s_h = 1; s_w = 1; s_d = 1
			conv7W, conv7b = getWeightsNBiases(True, conv6.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv7_in = conv(conv6, conv7W, conv7b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv7_norm = batch_norm(conv7_in, useType)
			conv7 = tf.nn.relu(conv7_norm)

		#up2
		with tf.variable_scope("UpSample2") as scope:
			up2 = unPool3D(conv7, 1, 2, 2)

		#conv8
		with tf.variable_scope("Conv8") as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 16; c_i = 32; s_h = 1; s_w = 1; s_d = 1
			conv8W, conv8b = getWeightsNBiases(True, up2.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv8_in = conv(up2, conv8W, conv8b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv8_norm = batch_norm(conv8_in, useType)
			conv8 = tf.nn.relu(conv8_norm)

		#conv9
		with tf.variable_scope("Conv9") as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 8; c_i = 16; s_h = 1; s_w = 1; s_d = 1
			conv9W, conv9b = getWeightsNBiases(True, conv8.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv9_in = conv(conv8, conv9W, conv9b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			conv9_norm = batch_norm(conv9_in, useType)
			conv9 = tf.nn.relu(conv9_norm)

		#conv10
		with tf.variable_scope("Conv10") as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 6; c_i = 8; s_h = 1; s_w = 1; s_d = 1
			conv10W, conv10b = getWeightsNBiases(True, conv9.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			conv10_in = conv(conv9, conv10W, conv10b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			# conv10_norm = batch_norm(conv10_in, useType)
			conv10 = tf.tanh(conv10_in)

	return conv10
		

def reconstruct(code, first, log, useType="test"):
	with tf.variable_scope("Decoder"):
		#uconv5
		with tf.variable_scope("uConv5", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			uconv5W, uconv5b = getWeightsNBiases(first, code.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			uconv5_in = conv(code, uconv5W, uconv5b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			uconv5_norm = batch_norm(uconv5_in, useType, not first)
			uconv5 = tf.nn.relu(uconv5_norm)

		#up1
		with tf.variable_scope("UpSample2") as scope:
			up1 = unPool3D(uconv5, 2, 2, 2)

		#uconv4
		with tf.variable_scope("uConv4", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			uconv4W, uconv4b = getWeightsNBiases(first, up1.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			uconv4_in = conv(up1, uconv4W, uconv4b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			uconv4_norm = batch_norm(uconv4_in, useType, not first)
			uconv4 = tf.nn.relu(uconv4_norm)

		#uconv3
		with tf.variable_scope("uConv3", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 128+3*32; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			uconv3W, uconv3b = getWeightsNBiases(first, uconv4.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			uconv3_in = conv(uconv4, uconv3W, uconv3b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			uconv3_norm = batch_norm(uconv3_in, useType, not first)
			uconv3 = tf.nn.relu(uconv3_norm)

		#up2
		with tf.variable_scope("UpSample2") as scope:
			up2 = unPool3D(uconv3, 2, 2, 2)

		#uconv2
		with tf.variable_scope("uConv2", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 64+16; c_i = 128+3*32; s_h = 1; s_w = 1; s_d = 1
			uconv2W, uconv2b = getWeightsNBiases(first, up2.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			uconv2_in = conv(up2, uconv2W, uconv2b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			uconv2_norm = batch_norm(uconv2_in, useType, not first)
			uconv2 = tf.nn.relu(uconv2_norm)

		#uconv1
		with tf.variable_scope("uConv1", reuse=(not first)) as scope:
			k_h = 3; k_w = 3; k_d = 3; c_o = 6; c_i = 64+16; s_h = 1; s_w = 1; s_d = 1
			uconv1W, uconv1b = getWeightsNBiases(first, uconv2.get_shape().as_list()[1:], k_h, k_w, k_d, c_o, c_i, s_h, s_w, s_d, scope)
			uconv1_in = conv(uconv2, uconv1W, uconv1b, k_h, k_w, k_d, c_o, s_h, s_w, s_d, padding="SAME", group=1)
			# uconv1_norm = batch_norm(uconv1_in, useType)
			uconv1 = tf.tanh(uconv1_in)

	return uconv1

def loss(outputs, labels, cond, name):
	"""Add L2Loss to all the trainable variables.

	Add summary for "Loss" and "Loss/avg".
	Args:
		outputs: Outputs from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
			of shape [batch_size]

	Returns:
		Loss tensor of type float.
	"""
	if cond:
		batch_size = FLAGS.trainBatch_size
	else:
		batch_size = 4*FLAGS.trainBatch_size	
	l2loss = tf.divide(tf.nn.l2_loss(outputs - labels), float(batch_size), name=name)
	# cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', l2loss)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return l2loss

def totalLoss():
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	"""Add summaries for losses in CIFAR-10 model.

	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
		total_loss: Total loss from loss().
	Returns:
		loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	# tf.add_to_collection("loss_averages", loss_averages)
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	# for l in losses + [total_loss]:
	# 	# Name each loss as '(raw)' and name the moving average version of the loss
	# 	# as the original loss name.
	# 	tf.summary.scalar(l.op.name +' (raw)', l)
	# 	tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):
	"""Train the model.

	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.

	Args:
		total_loss: Total loss from loss().
		global_step: Integer Variable counting the number of training steps
			processed.
	Returns:
		train_op: op for training.
	"""
	# Generate moving averages of all losses and associated summaries.
	with tf.device('/cpu:0'):
		loss_averages_op = _add_loss_summaries(total_loss)

	# lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, 1e+4, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer(1e-3)
		# opt = tf.train.MomentumOptimizer(lr, 0.9)
		grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables(), colocate_gradients_with_ops=True)
	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
