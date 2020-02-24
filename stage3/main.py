#! /usr/bin/python
# -*- coding: utf8 -*-

import os, pdb, pickle, time, random
import scipy.io
import tensorflow as tf
import numpy as np

from img2txt import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 12
LEARNING_RATE = 0.00001
MAX_EPOCH = 1
DISPLAY_STEP = 200
SAVE_STEP = 1
GENERATE_STEP = 1
IS_TRAINING = True
TRAINSET_SIZE = 8855
TESTSET_SIZE = 2933
GENERATE_RATIO = 10
TEXT_NUM = 10

LR_STEP = 10

DATASET = 'cub'
DATA_DIR = os.path.join('../data/', DATASET)
RESULT_DIR = os.path.join('result/', DATASET)
PRETRAINED_DISCRIMINATOR = os.path.join('../models', 'img2txt_model', 'model.ckpt-2000000')
PRETRAINED_GENERATOR = os.path.join('pretrained_model', 'model_0080.ckpt')

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

def compute_error(real, fake):
	return tf.reduce_mean(tf.abs(fake - real))
	# return tf.reduce_mean(tf.square(fake - real))

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
		
		d_model = show_and_tell_model.ShowAndTellModel(configuration.ModelConfig(), mode="inference")
		d_model.images = input_image
		d_model.build_image_embeddings()
	
		d_embed = d_model.image_embeddings
		d_net = d_model.inception_net
		d_out = d_model.inception_output
		
		d_model_fake = show_and_tell_model.ShowAndTellModel(configuration.ModelConfig(), mode="inference", reuse_img_embed=True)
		d_model_fake.images = g_image
		d_model_fake.build_image_embeddings()
		
		d_embed_fake = d_model_fake.image_embeddings
		d_net_fake = d_model_fake.inception_net
		d_out_fake = d_model_fake.inception_output
		
		G_loss = compute_error(input_image, g_image)
		
		G_loss += compute_error(d_model.inception_net['Conv2d_1a_3x3'], d_model_fake.inception_net['Conv2d_1a_3x3']) * 1
		G_loss += compute_error(d_model.inception_net['Conv2d_2a_3x3'], d_model_fake.inception_net['Conv2d_2a_3x3']) * 1
		G_loss += compute_error(d_model.inception_net['MaxPool_3a_3x3'], d_model_fake.inception_net['MaxPool_3a_3x3']) * 1
		G_loss += compute_error(d_model.inception_net['Conv2d_4a_3x3'], d_model_fake.inception_net['Conv2d_4a_3x3']) * 1
		G_loss += compute_error(d_model.inception_net['MaxPool_5a_3x3'], d_model_fake.inception_net['MaxPool_5a_3x3']) * 1
		G_loss += compute_error(d_model.inception_net['Mixed_6a'], d_model_fake.inception_net['Mixed_6a']) * 1
		G_loss += compute_error(d_model.inception_net['Mixed_7a'], d_model_fake.inception_net['Mixed_7a']) * 1
		
		G_loss += compute_error(d_embed, d_embed_fake)
		G_loss += compute_error(d_out, d_out_fake)

		
	lr = tf.placeholder(tf.float32)
	G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
	
	sess.run(tf.global_variables_initializer())
	
	saver_d = tf.train.Saver(var_list=[var for var in tf.global_variables() if var.name.startswith('InceptionV3') or var.name.startswith('image_embedding')])
	saver_d.restore(sess, PRETRAINED_DISCRIMINATOR)
	
	saver_g = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')], max_to_keep=0)
	saver_g.restore(sess, PRETRAINED_GENERATOR)
	
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
		
		lr_step = LEARNING_RATE
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
				
				_G, G_current = sess.run([G_opt, G_loss], feed_dict={input_image: image_batch_narray, input_text: text_batch_narray, lr: lr_step})
				if count % DISPLAY_STEP == 0:
					print('Epoch: %d Count: %d G_loss: %.2f' % (epoch, count, np.mean(G_current)))
			
			if epoch % SAVE_STEP == 0:
				saver_g.save(sess, os.path.join(RESULT_DIR, 'models/model_%04d.ckpt' % epoch))
					
			if epoch % GENERATE_STEP == 0:
				generate_img(sess, g_image, input_text, train_texts, train_filenames, epoch, 'train')
				generate_img(sess, g_image, input_text, test_texts, test_filenames, epoch, 'test')
				
			if epoch % LR_STEP == 0:
				lr_step = lr_step * 0.5

if __name__ == '__main__':
	main()