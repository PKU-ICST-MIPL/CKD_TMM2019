#! /usr/bin/python
# -*- coding: utf8 -*-

import os, pdb, pickle, time, random, sys
import scipy.io
import tensorflow as tf
import numpy as np

from inception_score import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----设置参数-----
BATCH_SIZE = 16
TRAINSET_SIZE = 8855
TESTSET_SIZE = 2933
GENERATE_RATIO = 10
TEXT_NUM = 10
load_epoch = 1

# -----设置路径-----
DATASET = 'cub'
DATA_DIR = os.path.join('../data/', DATASET)
RESULT_DIR = os.path.join('result', DATASET)

def build_net(ntype, nin, nwb=None, name=None, shape=None, out_shape=None, need_relu=True):
	W_init_args = {}
	b_init_args = {}
	if ntype=='conv':
		if nwb == None:
			nwb = []
			W_init = tf.random_normal_initializer(stddev=0.02)
			b_init = tf.constant_initializer(value=0.0)
			nwb.append(tf.get_variable(name=name+'_W', shape=shape, initializer=W_init, **W_init_args))
			nwb.append(tf.get_variable(name=name+'_b', shape=(shape[-1]), initializer=b_init, **b_init_args))
		if need_relu:
			return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1,1,1,1], padding='SAME', name=name) + nwb[1])
		else:
			return tf.nn.conv2d(nin, nwb[0], strides=[1,1,1,1], padding='SAME', name=name) + nwb[1]
	elif ntype=='pool':
		return tf.nn.avg_pool(nin, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	elif ntype=='deconv':
		if nwb == None:
			nwb = []
			W_init = tf.random_normal_initializer(stddev=0.02)
			b_init = tf.constant_initializer(value=0.0)
			nwb.append(tf.get_variable(name=name+'_W', shape=shape, initializer=W_init, **W_init_args))
			nwb.append(tf.get_variable(name=name+'_b', shape=(shape[-2]), initializer=b_init, **b_init_args))
		if need_relu:
			return tf.nn.relu(tf.nn.conv2d_transpose(nin, nwb[0], output_shape=out_shape, strides=[1,2,2,1], padding='SAME', name=name) + nwb[1])
		else:
			return tf.nn.conv2d_transpose(nin, nwb[0], output_shape=out_shape, strides=[1,2,2,1], padding='SAME', name=name) + nwb[1]
	elif ntype=='fc':
		if nwb == None:
			nwb = []
			W_init = tf.random_normal_initializer(stddev=0.1)
			b_init = tf.constant_initializer(value=0.0)
			nwb.append(tf.get_variable(name=name+'_W', shape=shape, initializer=W_init, **W_init_args))
			nwb.append(tf.get_variable(name=name+'_b', shape=(shape[-1]), initializer=b_init, **b_init_args))
		if need_relu:
			return tf.nn.relu(tf.matmul(nin, nwb[0]) + nwb[1])
		else:
			return tf.matmul(nin, nwb[0]) + nwb[1]
	elif ntype=='upsample':
		return tf.image.resize_images(nin, size=out_shape, method=1, align_corners=False)

def build_g_net(input, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	net={}
	# net['g_input_reshape'] = tf.reshape(input, shape=(BATCH_SIZE, 10240), name='g_input_reshape')
	# net['g_input_flatten'] = tf.contrib.layers.flatten(input)
	net['g_fc7'] = build_net('fc', input, nwb=None, name='g_fc7', shape=(1024, 4096))
	net['g_fc6'] = build_net('fc', net['g_fc7'], nwb=None, name='g_fc6', shape=(4096, 4096))
	net['g_pool5'] = build_net('fc', net['g_fc6'], nwb=None, name='g_pool5', shape=(4096, 25088))
	net['g_pool5_reshape'] = tf.reshape(net['g_pool5'], shape=(BATCH_SIZE, 7, 7, 512), name='g_pool5_reshape')
	
	net['g_conv5_4'] = build_net('upsample', net['g_pool5_reshape'], name='g_conv5_4', out_shape=(14, 14))
	net['g_conv5_3'] = build_net('conv', net['g_conv5_4'], nwb=None, name='g_conv5_3', shape=(3, 3, 512, 512))
	net['g_conv5_2'] = build_net('conv', net['g_conv5_3'], nwb=None, name='g_conv5_2', shape=(3, 3, 512, 512))
	net['g_conv5_1'] = build_net('conv', net['g_conv5_2'], nwb=None, name='g_conv5_1', shape=(3, 3, 512, 512))
	net['g_pool4'] = build_net('conv', net['g_conv5_1'], nwb=None, name='g_pool4', shape=(3, 3, 512, 512))
	
	net['g_conv4_4'] = build_net('upsample', net['g_pool4'], name='g_conv4_4', out_shape=(28, 28))
	net['g_conv4_3'] = build_net('conv', net['g_conv4_4'], nwb=None, name='g_conv4_3', shape=(3, 3, 512, 512))
	net['g_conv4_2'] = build_net('conv', net['g_conv4_3'], nwb=None, name='g_conv4_2', shape=(3, 3, 512, 512))
	net['g_conv4_1'] = build_net('conv', net['g_conv4_2'], nwb=None, name='g_conv4_1', shape=(3, 3, 512, 512))
	net['g_pool3'] = build_net('conv', net['g_conv4_1'], nwb=None, name='g_pool3', shape=(3, 3, 512, 256))
	
	net['g_conv3_4'] = build_net('upsample', net['g_pool3'], name='g_conv3_4', out_shape=(56, 56))
	net['g_conv3_3'] = build_net('conv', net['g_conv3_4'], nwb=None, name='g_conv3_3', shape=(3, 3, 256, 256))
	net['g_conv3_2'] = build_net('conv', net['g_conv3_3'], nwb=None, name='g_conv3_2', shape=(3, 3, 256, 256))
	net['g_conv3_1'] = build_net('conv', net['g_conv3_2'], nwb=None, name='g_conv3_1', shape=(3, 3, 256, 256))
	net['g_pool2'] = build_net('conv', net['g_conv3_1'], nwb=None, name='g_pool2', shape=(3, 3, 256, 128))
	
	net['g_conv2_2'] = build_net('upsample', net['g_pool2'], name='g_conv2_2', out_shape=(112, 112))
	net['g_conv2_1'] = build_net('conv', net['g_conv2_2'], nwb=None, name='g_conv2_1', shape=(3, 3, 128, 128))
	net['g_pool1'] = build_net('conv', net['g_conv2_1'], nwb=None, name='g_pool1', shape=(3, 3, 128, 64))
	
	net['g_conv1_2'] = build_net('upsample', net['g_pool1'], name='g_conv1_2', out_shape=(224, 224))
	net['g_conv1_1'] = build_net('conv', net['g_conv1_2'], nwb=None, name='g_conv1_1', shape=(3, 3, 64, 64))
	net['g_img'] = build_net('conv', net['g_conv1_1'], nwb=None, name='g_img', shape=(3, 3, 64, 3), need_relu=False)
	
	return net, net['g_img']
	

# -----生成测试集样本-----
def generate_img(sess, g_image, input_text, test_texts):
	text_batch = []
	img_list = []
	for index in range(TESTSET_SIZE*TEXT_NUM):
		if len(text_batch) < BATCH_SIZE - 1:
			text_batch.append(test_texts[index//TEXT_NUM][index%TEXT_NUM])
			continue
			
		text_batch.append(test_texts[index//TEXT_NUM][index%TEXT_NUM])
		text_batch_narray = np.asarray(text_batch)
		text_batch = []
		
		outputs = sess.run(g_image, feed_dict = {input_text: np.asarray(text_batch_narray)})
		outputs = np.minimum(np.maximum(outputs, 0.0), 255.0)
		for i in range(BATCH_SIZE):		
			img_list.append(outputs[i,:,:,:].astype(np.uint8))
	
	random.shuffle(img_list)
	return img_list
			
def main():
	# -----初始化model-----
	with tf.variable_scope(tf.get_variable_scope()):
		input_text = tf.placeholder(tf.float32, (BATCH_SIZE, 1024))
		g_net, g_image = build_g_net(input_text)	
	
	# -----设置session-----
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		
		# -----设置saver-----	
		saver_g = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
		generator_model_path = ('%s/models/model_%04d.ckpt' % (RESULT_DIR, load_epoch))
		saver_g.restore(sess, generator_model_path)
		
		# -----读取数据集-----	
		with open(os.path.join(DATA_DIR, 'test', 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
			test_texts = pickle.load(f)
		
		# -----生成图像-----
		img_list = generate_img(sess, g_image, input_text, test_texts)
	
	# -----计算Inception score-----
	score_mean, score_std = get_inception_score(img_list)
	print('\nInception score: %.2f %.2f' % (score_mean, score_std))

if __name__ == '__main__':
	main()