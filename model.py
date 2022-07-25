import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
import sys
import os
import estimate as esti

'''
Reproduced from:
  Kolchinsky, Artemy, Brendan D. Tracey, and David H. Wolpert.
  "Nonlinear information bottleneck." Entropy 21.12 (2019): 1181.
'''

class NonLinearIB(layers.Layer):
	def __init__(self, init_noisevar=None, **kwargs):
		super(NonLinearIB, self).__init__(**kwargs)
		if init_noisevar is None:
			init_noisevar = 0.0
		self.init_noisevar = init_noisevar
	def build(self, input_shape):
		assert len(input_shape)>=2
		self.logvar = tf.Variable(self.init_noisevar,trainable=True,dtype=tf.float32)
	def call(self, inputs):
		noisevar = tf.exp(self.logvar)
		self.noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,stddev=1.0, dtype=tf.float32, name="noise")
		return inputs + self.noise * tf.exp(0.5* self.logvar), noisevar

class nIB(keras.Model):
	def __init__(self,gamma,latent_dim,name=None,**kwargs):
		super(nIB,self).__init__(name=name)
		self.gamma = gamma
		self.latent_dim = latent_dim
		self.img_width = 28
		self.img_ch    = 1
		self.nclass = 10
		self.nib_layer = NonLinearIB(name="nib_layer")

		enc_input = keras.Input(shape=(self.img_width,self.img_width,self.img_ch))
		x = layers.Conv2D(16,3,activation="relu",strides=2,padding="same")(enc_input)
		x = layers.Conv2D(16,3,activation="relu",strides=2,padding="same")(x)
		x = layers.Flatten()(x)
		z_mean = layers.Dense(latent_dim,activation="relu")(x)
		enc_out, noise_var = self.nib_layer(z_mean)
		self.encoder = keras.Model(enc_input,[z_mean,noise_var,enc_out],name="encoder")
		#self.encoder.summary()

		dec_input = keras.Input(shape=(latent_dim,))
		# FIXME: limited to linear classifier
		dec_output = layers.Dense(self.nclass,activation="linear")(dec_input) # predicting logits
		self.decoder = keras.Model(dec_input,dec_output,name="decoder")
		#self.decoder.summary()

		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.ce_loss_tracker    = keras.metrics.Mean(name="ce_loss")
		self.mi_loss_tracker    = keras.metrics.Mean(name="mi_loss")
		self.acc_tracker        = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
		self.test_total_loss_tracker= keras.metrics.Mean(name="val_total_loss")
		self.test_ce_loss_tracker   = keras.metrics.Mean(name="val_ce_loss")
		self.test_mi_loss_tracker   = keras.metrics.Mean(name="val_mi_loss")
		self.test_acc_tracker       = keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.ce_loss_tracker,
			self.mi_loss_tracker,
			self.acc_tracker,
			self.test_total_loss_tracker,
			self.test_ce_loss_tracker,
			self.test_mi_loss_tracker,
			self.test_acc_tracker,
		]
	def train_step(self,data):
		x_data, y_label = data
		#batchsize = tf.cast(tf.shape(x_data)[0],tf.float32)
		with tf.GradientTape() as tape:
			z_mean, z_var, z = self.encoder(x_data,training=True)
			logits = self.decoder(z,training=True)
			# calculate the losses
			soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
			ce_loss = tf.reduce_mean(soft_out)
			# mi loss
			pdist = esti.pairwiseDistance(z_mean)
			mi_loss = esti.gmEstimate(pdist,z_var)
			#
			total_loss = ce_loss + self.gamma * mi_loss
		grads = tape.gradient(total_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
		# update losses
		self.total_loss_tracker.update_state(total_loss)
		self.ce_loss_tracker.update_state(ce_loss)
		self.mi_loss_tracker.update_state(mi_loss)
		# update accuracy
		self.acc_tracker.update_state(y_label,logits)
		return {
			"accuracy":self.acc_tracker.result(),
			"total_loss":self.total_loss_tracker.result(),
			"ce_loss":self.ce_loss_tracker.result(),
			"mi_loss":self.mi_loss_tracker.result(),
		}
	def test_step(self,data):
		x_data, y_label  =data
		#batchsize = tf.cast(tf.shape(x_data)[0],tf.float32)
		z_mean, z_var, z = self.encoder(x_data,training=False)
		logits = self.decoder(z,training=False)
		# calculate the losses
		soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
		ce_loss = tf.reduce_mean(soft_out)
		# 
		pdist = esti.pairwiseDistance(z_mean)
		mi_loss = esti.gmEstimate(pdist,z_var)
		#
		total_loss = ce_loss + self.gamma*mi_loss
		self.test_total_loss_tracker.update_state(total_loss)
		self.test_ce_loss_tracker.update_state(ce_loss)
		self.test_mi_loss_tracker.update_state(mi_loss)
		#
		self.test_acc_tracker.update_state(y_label,logits)
		return {
			"accuracy":self.test_acc_tracker.result(),
			"total_loss":self.test_total_loss_tracker.result(),
			"ce_loss":self.test_ce_loss_tracker.result(),
			"mi_loss":self.test_mi_loss_tracker.result(),
		}
	def call(self,data):
		_,_,z = self.encoder(data,training=False)
		return self.decoder(z,training=False)

'''
Reproduced from:

  Alemi, Alexander A., et al.
  "Deep variational information bottleneck."
  arXiv preprint arXiv:1612.00410 (2016).
  
'''

# MC sampling layer for VIB

class Sampling(layers.Layer):
	def call(self,inputs):
		z_mean, z_log_var = inputs
		epsilon = tf.random.normal(shape=(tf.shape(z_mean)))
		return z_mean + epsilon * tf.exp(0.5 * z_log_var)

class vIB(keras.Model):
	def __init__(self,gamma,latent_dim,name=None,**kwargs):
		super(vIB,self).__init__(name=name)
		self.gamma = gamma
		self.latent_dim = latent_dim
		self.img_width = 28
		self.img_ch    = 1
		self.nclass    = 10

		enc_input = keras.Input(shape=(self.img_width,self.img_width,self.img_ch))
		x = layers.Conv2D(16,3,activation="relu",strides=2,padding="same")(enc_input)
		x = layers.Conv2D(16,3,activation="relu",strides=2,padding="same")(x)
		x = layers.Flatten()(x)
		z_mean = layers.Dense(latent_dim,activation="relu")(x)
		z_log_var = layers.Dense(latent_dim,activation="relu")(x)
		z = Sampling()([z_mean,z_log_var])
		self.encoder = keras.Model(enc_input,[z_mean,z_log_var,z],name="encoder")
		#self.encoder.summary()

		dec_input = keras.Input(shape=(latent_dim,))
		# FIXME: limited to linear classifier
		dec_output = layers.Dense(self.nclass,activation="linear")(dec_input)
		self.decoder = keras.Model(dec_input,dec_output,name="decoder")

		# losses
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.ce_loss_tracker    = keras.metrics.Mean(name="ce_loss")
		self.mi_loss_tracker    = keras.metrics.Mean(name="mi_loss")
		self.acc_tracker        = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
		self.test_total_loss_tracker= keras.metrics.Mean(name="val_total_loss")
		self.test_ce_loss_tracker   = keras.metrics.Mean(name="val_ce_loss")
		self.test_mi_loss_tracker   = keras.metrics.Mean(name="val_mi_loss")
		self.test_acc_tracker       = keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.ce_loss_tracker,
			self.mi_loss_tracker,
			self.acc_tracker,
			self.test_total_loss_tracker,
			self.test_ce_loss_tracker,
			self.test_mi_loss_tracker,
			self.test_acc_tracker,
		]

	def train_step(self,data):
		x_data, y_label = data
		with tf.GradientTape() as tape:
			z_mean,z_log_var,z = self.encoder(x_data,training=True)
			logits = self.decoder(z,training=True)
			# calculate losses
			soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
			ce_loss =tf.reduce_mean(soft_out)
			# calculate kl loss
			kl_loss = tf.reduce_mean(
					tf.reduce_sum(
							-0.5 * ( 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) , axis=1
						)
				)
			total_loss = self.gamma*kl_loss + ce_loss
			# update
		grads = tape.gradient(total_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
		self.total_loss_tracker.update_state(total_loss)
		self.ce_loss_tracker.update_state(ce_loss)
		self.mi_loss_tracker.update_state(kl_loss)
		self.acc_tracker.update_state(y_label,logits)
		return {
			"accuracy":self.acc_tracker.result(),
			"total_loss":self.total_loss_tracker.result(),
			"ce_loss":self.ce_loss_tracker.result(),
			"mi_loss":self.mi_loss_tracker.result(),
		}
	def test_step(self,data):
		x_data, y_label = data
		z_mean,z_log_var,z = self.encoder(x_data,training=False)
		logits = self.decoder(z,training=False)
		# losses
		soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
		ce_loss = tf.reduce_mean(soft_out)
		# calculate kl loss
		mi_loss = tf.reduce_mean(
				tf.reduce_sum(
						-0.5 * ( 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
					)
			)
		total_loss = self.gamma * mi_loss + ce_loss
		self.test_total_loss_tracker.update_state(total_loss)
		self.test_ce_loss_tracker.update_state(ce_loss)
		self.test_mi_loss_tracker.update_state(mi_loss)
		self.test_acc_tracker.update_state(y_label,logits)
		return {
			"accuracy": self.test_acc_tracker.result(),
			"total_loss":self.test_total_loss_tracker.result(),
			"ce_loss":self.test_ce_loss_tracker.result(),
			"mi_loss":self.test_mi_loss_tracker.result(),
		}
	def call(self,data):
		_,_,z = self.encoder(data,training=False)
		return self.decoder(z,training=False)
