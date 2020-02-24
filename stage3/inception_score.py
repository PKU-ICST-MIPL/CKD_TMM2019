#! /usr/bin/python
# -*- coding: utf8 -*-

import os, sys, math, pdb, pickle
import numpy as np
import scipy.misc
import tensorflow as tf

MODEL_DIR = '../models/inception_score/'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
	assert(type(images) == list)
	assert(type(images[0]) == np.ndarray)
	assert(len(images[0].shape) == 3)
	assert(np.max(images[0]) > 10)
	assert(np.min(images[0]) >= 0.0)
	
	
	global softmax
	with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')
	# Works with an arbitrary minibatch size.
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True
	with tf.Session(config=config) as sess:
		pool3 = sess.graph.get_tensor_by_name('pool_3:0')
		ops = pool3.graph.get_operations()
		for op_idx, op in enumerate(ops):
			for o in op.outputs:
				shape = o.get_shape()

				try:
					shape = [s.value for s in shape]
				except:
					continue
				
				new_shape = []
				for j, s in enumerate(shape):
					if s == 1 and j == 0:
						new_shape.append(None)
					else:
						new_shape.append(s)
				o._shape = tf.TensorShape(new_shape)
		w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
		logits = tf.matmul(tf.squeeze(pool3), w)
		softmax = tf.nn.softmax(logits)
	
	
	inps = []
	for img in images:
		img = img.astype(np.float32)
		inps.append(np.expand_dims(img, 0))
	bs = 100
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True
	with tf.Session(config=config) as sess:
		preds = []
		n_batches = int(math.ceil(float(len(inps)) / float(bs)))
		for i in range(n_batches):
			sys.stdout.write(".")
			sys.stdout.flush()
			inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
			inp = np.concatenate(inp, 0)
			pred = sess.run(softmax, {'ExpandDims:0': inp})
			preds.append(pred)
		preds = np.concatenate(preds, 0)
		scores = []
		for i in range(splits):
			part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
			kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
			kl = np.mean(np.sum(kl, 1))
			scores.append(np.exp(kl))
		return np.mean(scores), np.std(scores)