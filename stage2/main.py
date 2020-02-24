#! /usr/bin/python
# -*- coding: utf8 -*-

import os, pdb, pickle, time, random
import scipy.io
import tensorflow as tf
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 12
LEARNING_RATE = 0.0001
MAX_EPOCH = 100
DISPLAY_STEP = 200
SAVE_STEP = 10
GENERATE_STEP = 10
IS_TRAINING = True
TRAINSET_SIZE = 8855
TESTSET_SIZE = 2933
GENERATE_RATIO = 10
TEXT_NUM = 10

DATASET = 'cub'
DATA_DIR = os.path.join('../data/', DATASET)
RESULT_DIR = os.path.join('result/', DATASET)
PREMODEL_PATH = os.path.join('pretrained_model/model_0100.ckpt')

def get_weight_bias(vgg_layers, i, name=None):
	weights = vgg_layers[i][0][0][2][0][0]
	if name == 'vgg_fc6':
		weights = np.reshape(weights, (7*7*512, 4096))
	if name == 'vgg_fc7':
		weights = np.reshape(weights, (4096, 4096))
	weights = tf.Variable(weights, name=name+'_W')
	bias = vgg_layers[i][0][0][2][0][1]
	bias = tf.Variable(np.reshape(bias, (bias.size)), name=name+'_b')
	return weights, bias

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

def build_vgg_net(input, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	net = {}
	vgg_rawnet = scipy.io.loadmat('../models/imagenet-vgg-verydeep-19.mat')
	vgg_layers = vgg_rawnet['layers'][0]
	# net['input'] = input-np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
	net['input'] = input
	net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0, name='vgg_conv1_1'), name='vgg_conv1_1')	
	net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2, name='vgg_conv1_2'), name='vgg_conv1_2')
	net['pool1'] = build_net('pool', net['conv1_2'])
	net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5, name='vgg_conv2_1'), name='vgg_conv2_1')
	net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7, name='vgg_conv2_2'), name='vgg_conv2_2')
	net['pool2'] = build_net('pool', net['conv2_2'])
	net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10, name='vgg_conv3_1'), name='vgg_conv3_1')
	net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12, name='vgg_conv3_2'), name='vgg_conv3_2')
	net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14, name='vgg_conv3_3'), name='vgg_conv3_3')
	net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16, name='vgg_conv3_4'), name='vgg_conv3_4')
	net['pool3'] = build_net('pool', net['conv3_4'])
	net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19, name='vgg_conv4_1'), name='vgg_conv4_1')
	net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21, name='vgg_conv4_2'), name='vgg_conv4_2')
	net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23, name='vgg_conv4_3'), name='vgg_conv4_3')
	net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25, name='vgg_conv4_4'), name='vgg_conv4_4')
	net['pool4'] = build_net('pool', net['conv4_4'])
	net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28, name='vgg_conv5_1'), name='vgg_conv5_1')
	net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30, name='vgg_conv5_2'), name='vgg_conv5_2')
	net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32, name='vgg_conv5_3'), name='vgg_conv5_3')
	net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34, name='vgg_conv5_4'), name='vgg_conv5_4')
	net['pool5'] = build_net('pool', net['conv5_4'])
	# net['pool5_flatten'] = tf.contrib.layers.flatten(net['pool5'])
	# net['fc6'] = build_net('fc', net['pool5_flatten'], get_weight_bias(vgg_layers, 37, name='vgg_fc6'), name='vgg_fc6')
	# net['fc7'] = build_net('fc', net['fc6'], get_weight_bias(vgg_layers, 39, name='vgg_fc7'), name='vgg_fc7')
	# net['fc8'] = build_net('fc', net['fc7'], nwb=None, shape=(4096, 1), name='vgg_fc8', need_relu=False)
	
	return net

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

def compute_error(real, fake):
	return tf.reduce_mean(tf.abs(fake - real))
	# return tf.reduce_mean(tf.square(fake - real))

def arrange_text(test_texts, index):
	temp = []
	temp_index = random.randint(0, 9)
	for i in range(10):
		temp.append(test_texts[index][temp_index])
	return temp

def generate_img(sess, g_image, input_text, test_texts, test_filenames, epoch, target_set):
	output_image_dir = os.path.join(RESULT_DIR, target_set + '_images/%04d/' % epoch)
	if not os.path.exists(output_image_dir):
		os.makedirs(output_image_dir)
	
	random_index = np.random.permutation(TESTSET_SIZE)
	text_batch = []
	filename_batch = []
	for index in random_index[0:TESTSET_SIZE//GENERATE_RATIO]:
		if len(text_batch) < BATCH_SIZE - 1:
			text_batch.append(test_texts[index][0])
			filename_batch.append(test_filenames[index])
			continue
			
		text_batch.append(test_texts[index][0])
		filename_batch.append(test_filenames[index])
		text_batch_narray = np.asarray(text_batch)
		filename_batch_narray = np.asarray(filename_batch)
		text_batch = []
		filename_batch = []
		
		outputs = sess.run(g_image, feed_dict = {input_text: np.asarray(text_batch_narray)})
		outputs = np.minimum(np.maximum(outputs, 0.0), 255.0)
		for i in range(BATCH_SIZE):
			filename = filename_batch_narray[i]
			output_image_class_dir = os.path.join(output_image_dir, filename.split('/')[0])
			if not os.path.exists(output_image_class_dir):
				os.makedirs(output_image_class_dir)
			output_path = os.path.join(output_image_dir, filename + '.jpg')
			scipy.misc.toimage(outputs[i,:,:,:], cmin=0, cmax=255).save(output_path)

			
def main():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	sess = tf.Session(config=config)
	with tf.variable_scope(tf.get_variable_scope()):
		input_image = tf.placeholder(tf.float32,[BATCH_SIZE, 224, 224, 3])
		input_text = tf.placeholder(tf.float32, (BATCH_SIZE, 1024))
		g_net, g_image = build_g_net(input_text)
		
		vgg_net = build_vgg_net(input_image)
		vgg_net_fake = build_vgg_net(g_image, reuse=True)

		G_loss = 0
		
		G_loss += compute_error(vgg_net['input'], vgg_net_fake['input'])* 1
		G_loss += compute_error(vgg_net['conv1_1'], vgg_net_fake['conv1_1'])* 1
		G_loss += compute_error(vgg_net['conv1_2'], vgg_net_fake['conv1_2'])* 1
		G_loss += compute_error(vgg_net['pool1'], vgg_net_fake['pool1'])* 1
		
		G_loss += compute_error(vgg_net['conv2_1'], vgg_net_fake['conv2_1'])* 1
		G_loss += compute_error(vgg_net['conv2_2'], vgg_net_fake['conv2_2'])* 1
		G_loss += compute_error(vgg_net['pool2'], vgg_net_fake['pool2'])* 1
		
		G_loss += compute_error(vgg_net['conv3_1'], vgg_net_fake['conv3_1'])* 1
		G_loss += compute_error(vgg_net['conv3_2'], vgg_net_fake['conv3_2'])* 1
		G_loss += compute_error(vgg_net['conv3_3'], vgg_net_fake['conv3_3'])* 1
		G_loss += compute_error(vgg_net['conv3_4'], vgg_net_fake['conv3_4'])* 1
		G_loss += compute_error(vgg_net['pool3'], vgg_net_fake['pool3'])* 1
		
		G_loss += compute_error(vgg_net['conv4_1'], vgg_net_fake['conv4_1'])* 1
		G_loss += compute_error(vgg_net['conv4_2'], vgg_net_fake['conv4_2'])* 1
		G_loss += compute_error(vgg_net['conv4_3'], vgg_net_fake['conv4_3'])* 1
		G_loss += compute_error(vgg_net['conv4_4'], vgg_net_fake['conv4_4'])* 1
		G_loss += compute_error(vgg_net['pool4'], vgg_net_fake['pool4'])* 1
		
		G_loss += compute_error(vgg_net['conv5_1'], vgg_net_fake['conv5_1'])* 1
		G_loss += compute_error(vgg_net['conv5_2'], vgg_net_fake['conv5_2'])* 1
		G_loss += compute_error(vgg_net['conv5_3'], vgg_net_fake['conv5_3'])* 1
		G_loss += compute_error(vgg_net['conv5_4'], vgg_net_fake['conv5_4'])* 1
		G_loss += compute_error(vgg_net['pool5'], vgg_net_fake['pool5'])* 1

		
	lr = tf.placeholder(tf.float32)
	G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
	
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
	saver.restore(sess, PREMODEL_PATH)
	
	if IS_TRAINING:
		
		with open(os.path.join(DATA_DIR, 'train/images_224.pickle'), 'rb') as f:
			train_images = pickle.load(f)
		with open(os.path.join(DATA_DIR, 'train/char-CNN-RNN-embeddings.pickle'), 'rb') as f:
			train_texts = pickle.load(f)
		with open(os.path.join(DATA_DIR, 'train/filenames.pickle'), 'rb') as f:
			train_filenames = pickle.load(f)
		with open(os.path.join(DATA_DIR, 'test/char-CNN-RNN-embeddings.pickle'), 'rb') as f:
			test_texts = pickle.load(f)
		with open(os.path.join(DATA_DIR, 'test/filenames.pickle'), 'rb') as f:
			test_filenames = pickle.load(f)

		for epoch in range(1, MAX_EPOCH + 1):
			count = 0
			random_index = np.random.permutation(TRAINSET_SIZE*TEXT_NUM)
			
			image_batch = []
			text_batch = []
			for index in random_index:
				count += 1
				if len(image_batch) < BATCH_SIZE - 1:
					image_batch.append(train_images[index//TEXT_NUM])
					text_batch.append(train_texts[index//TEXT_NUM][index%TEXT_NUM])
					continue
				
				image_batch.append(train_images[index//TEXT_NUM])
				text_batch.append(train_texts[index//TEXT_NUM][index%TEXT_NUM])
				image_batch_narray = np.asarray(image_batch)
				text_batch_narray = np.asarray(text_batch)
				image_batch = []
				text_batch = []
				
				_G, G_current = sess.run([G_opt, G_loss], feed_dict={input_image: image_batch_narray, input_text: text_batch_narray, lr: LEARNING_RATE})
				if count % DISPLAY_STEP == 0:
					print('Epoch: %d Count: %d G_loss: %.2f' % (epoch, count, np.mean(G_current)))
			
			if epoch % SAVE_STEP == 0:
				saver.save(sess, os.path.join(RESULT_DIR, 'models/model_%04d.ckpt' % epoch))
					
			if epoch % GENERATE_STEP == 0:
				generate_img(sess, g_image, input_text, train_texts, train_filenames, epoch, 'train')
				generate_img(sess, g_image, input_text, test_texts, test_filenames, epoch, 'test')

if __name__ == '__main__':
	main()