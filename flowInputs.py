import numpy
import tensorflow as tf

RESOLUTIONS = [32]

def createPath(cases, relevant):
	res = []
	for i in cases:
		res.append(i+relevant)
	return res

def inputCrops(data, resolution, startVoxels):
	p, k, nut, u = tf.split(tf.slice(data, startVoxels[0], resolution), [1,1,1,3], 3)
	u = normalizeTensor(u)
	p = normalizeTensor(p)
	k = normalizeTensor(k)
	nut = normalizeTensor(nut)
	if len(startVoxels) > 1:
		for position in startVoxels[1:]:
			tp, tk, tnut, tu = tf.split(tf.slice(data, position, resolution), [1,1,1,3], 3)
			u = tf.concat([u, normalizeTensor(tu)], 0)
			p = tf.concat([p, normalizeTensor(tp)], 0)
			k = tf.concat([k, normalizeTensor(tk)], 0)
			nut = tf.concat([nut, normalizeTensor(tnut)], 0)
	return p, k, nut, u

def normalizeTensor(tensor):
	maxT = tf.abs(tf.reduce_max(tensor)); minT = tf.abs(tf.reduce_min(tensor))
	tensor = tf.cond(maxT > minT, lambda: tf.divide(tensor, maxT), lambda: tf.divide(tensor,minT))
	return tensor

def inputForces(cases, log, batch_size, useType="test"):
	#if useType == "test":
	#        batch_size = FLAGS.testBatch_size
	#else:
	#        batch_size = FLAGS.trainBatch_size
	with tf.variable_scope("InputForces") as scope:
		path = createPath(cases, "/allInOne.raw")

		pathLab = createPath(cases, "/forcesLast.dat")
		print("Loading labels..", file=log)
		_labels = []
		for c in pathLab:
			lbl = numpy.loadtxt(c, dtype=numpy.float32)
			_labels.append(lbl)
		_labels = numpy.array(_labels, dtype=numpy.float32)
		labels = tf.convert_to_tensor(_labels)

		print("Defining Input queue..", file=log)
		nm = tf.convert_to_tensor(path)

		if useType == "train":
			q, label = tf.train.slice_input_producer([nm, labels], shuffle=True, capacity=len(cases))
		else:
			q, label = tf.train.slice_input_producer([nm, labels], shuffle=False, capacity=len(cases))

		print("Defining preprocessing operations..", file=log)
		reader = tf.read_file(q)
		data = tf.decode_raw(reader, tf.float32)
		res = RESOLUTIONS[0]
		data = tf.reshape(data, [3*res,2*res,res, 6])

		if useType == "train":
			allInOne = tf.random_crop(data, [80, 56, 32, 6])
			p, k, nut, u = tf.split(allInOne, [1,1,1,3], 3)
			u = normalizeTensor(u)
			p = normalizeTensor(p)
			k = normalizeTensor(k)
			nut = normalizeTensor(nut)
		else:
			allInOne = tf.slice(data, [7, 3, 0, 0], [80, 56, 32, 6])
			p, k, nut, u = tf.split(allInOne, [1,1,1,3], 3)
			u = normalizeTensor(u)
			p = normalizeTensor(p)
			k = normalizeTensor(k)
			nut = normalizeTensor(nut)
			# startPoints = [[0, 0, 0, 0], [15, 0, 0, 0], [0, 7, 0, 0], [15, 7, 0, 0], [7, 3, 0, 0]]
			# p, k, nut, u = inputCrops(data, [80, 56, 32, 6], startPoints)

		num_threads = 8
		return tf.train.batch([p, k, nut, u, label], batch_size=batch_size, \
		num_threads=num_threads, capacity=10*batch_size)

def inputForcesAllCrops(cases, log, batch_size, useType="test"):
	#if useType == "test":
	#        batch_size = FLAGS.testBatch_size
	#else:
	#        batch_size = FLAGS.trainBatch_size
	with tf.variable_scope("InputForces") as scope:
		path = createPath(cases, "/allInOne.raw")

		# pathLab = createPath(cases, "/forcesLast.dat")
		# print("Loading labels..", file=log)
		# _labels = []
		# for c in pathLab:
		#       lbl = numpy.loadtxt(c, dtype=numpy.float32)
		#       _labels.append(lbl)
		# _labels = numpy.array(_labels, dtype=numpy.float32)
		# labels = tf.convert_to_tensor(_labels)

		print("Defining Input queue..", file=log)
		nm = tf.convert_to_tensor(path)

		q, = tf.train.slice_input_producer([nm], shuffle=False, capacity=len(cases))

		print("Defining preprocessing operations..", file=log)
		reader = tf.read_file(q)
		data = tf.decode_raw(reader, tf.float32)
		res = RESOLUTIONS[0]
		data = tf.reshape(data, [3*res,2*res,res, 6])

		startPoints = [[i,j,0,0] for i in range(96-80) for j in range(64-56)]
		# startX = numpy.array([i for i in range(96-36) for j in range(64-56)], dtype=numpy.int)
		p, k, nut, u = inputCrops(data, [80, 56, 32, 6], startPoints)
		allInOne = tf.concat([p,k,nut,u], 3)
		allInOne = tf.train.batch([tf.reshape(allInOne, shape=((96-80)*(64-56), 80, 56, 32, 6))], 1, 1, capacity=4*(96-80)*(64-56), enqueue_many=True)
		# allInOne, index = tf.train.batch([tf.reshape(allInOne, shape=((96-80)*(64-56), 80, 56, 32, 6)), tf.constant(startX)], 1, 1, capacity=4*(96-36)*(64-56), enqueue_many=True)

		num_threads = 8
		# return tf.train.batch([p, k, nut, u, label], batch_size=batch_size, \
		# num_threads=num_threads, capacity=10*batch_size)
		return tf.train.batch([p, k, nut, u], batch_size=batch_size, \
		num_threads=num_threads, capacity=10*batch_size)

def inputFlow(cases, log, batch_size, useType="test"):
	with tf.variable_scope("InputFlow") as scope:
		path = createPath(cases, "/allInOne.raw")

		print("Defining Input queue..")
		nm = tf.convert_to_tensor(path)

		if useType == "train":
			q, = tf.train.slice_input_producer([nm], shuffle=True, capacity=len(cases))
		else:
			q, = tf.train.slice_input_producer([nm], shuffle=False, capacity=len(cases))

		print("Defining preprocessing operations..")
		reader = tf.read_file(q)
		data = tf.decode_raw(reader, tf.float32)
		res = RESOLUTIONS[0]
		data = tf.reshape(data, [3*res,2*res,res, 6])

		num_threads = 8
		if useType == "train":
			allInOne = tf.random_crop(data, [36, 56, 32, 6])
			p, k, nut, u = tf.split(allInOne, [1,1,1,3], 3)
			u = normalizeTensor(u)
			p = normalizeTensor(p)
			k = normalizeTensor(k)
			nut = normalizeTensor(nut)
			allInOne = tf.concat([p,k,nut,u], 3)
		else:
			startPoints = [[i,j,0,0] for i in range(96-36) for j in range(64-56)]
			startX = numpy.array([i for i in range(96-36) for j in range(64-56)], dtype=numpy.int)
			p, k, nut, u = inputCrops(data, [36, 56, 32, 6], startPoints)
			allInOne = tf.concat([p,k,nut,u], 3)
			allInOne, index = tf.train.batch([tf.reshape(allInOne, shape=((96-36)*(64-56), 36, 56, 32, 6)), tf.constant(startX)], 1, 1, capacity=4*(96-36)*(64-56), enqueue_many=True)

		allInOne, label = tf.split(tf.reshape(allInOne, shape=(36,56,32,6)), [24, 12], 0)
		p, k, nut, u = tf.split(allInOne, [1,1,1,3], 3)
		return tf.train.batch([p, k, nut, u, label], batch_size=batch_size, \
		num_threads=num_threads, capacity=10*batch_size)


def inputReconstructionCrops(cases, log, batch_size, resolution, isFlow=False):
	with tf.variable_scope("ReconstructionInput") as scope:
		path = createPath(cases, "/allInOne.raw")
		print("Defining Input queue..", file=log)
		nm = tf.convert_to_tensor(path)
		q, = tf.train.slice_input_producer([nm], shuffle=False, capacity=len(cases))

		print("Defining preprocessing operations..", file=log)
		reader = tf.read_file(q)
		data = tf.decode_raw(reader, tf.float32)
		res = RESOLUTIONS[0]
		data = tf.reshape(data, [3*res,2*res,res, 6])

		num_threads = 8
		startPoints = [[i,j,0,0] for i in range(96-resolution[0]) for j in range(64-resolution[1])]
		startX = numpy.array([i for i in range(96-resolution[0]) for j in range(64-resolution[1])], dtype=numpy.int)
		p, k, nut, u = inputCrops(data, resolution, startPoints)
		allInOne = tf.concat([p,k,nut,u], 3)
		allInOne, index = tf.train.batch([tf.reshape(allInOne, shape=[(96-resolution[0])*(64-resolution[1]), \
			resolution[0], resolution[1], resolution[2], resolution[3]]), tf.constant(startX)], 1, 1, \
			capacity=4*(96-resolution[0])*(64-resolution[1]), enqueue_many=True)

		allInOne = tf.reshape(allInOne, shape=resolution)
		if isFlow:
			allInOne, label = tf.split(allInOne, [24, 12], 0)

		p, k, nut, u = tf.split(allInOne, [1,1,1,3], 3)
		return tf.train.batch([p, k, nut, u, index], batch_size=batch_size, \
		num_threads=num_threads, capacity=10*batch_size)
