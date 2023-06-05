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
		return self.decoder(z,training=False), z

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
		return self.decoder(z,training=False), z

# Bernoulli sampling
# Should use a relaxation: the Grumble-Softmax function
# Then the training and testing phase behavior is different
# during testing time, feed the decoder with 0/1 samples
# but it is trained with relaxed 0/1 samples.
class BernoulliSampling(layers.Layer):
	def call(self,inputs,training=False):
		# the input is the logits
		if training:
			# use smooth relaxation of the zero-one function
			z_logits = inputs #the is the log-likelihood ratio log(v/(1-v))
			cdf = tf.random.uniform(shape=(tf.shape(z_logits)))
			log_prob = tf.math.log(cdf+1e-9)-tf.math.log(1-cdf+1e-9)
			return tf.nn.sigmoid(z_logits-log_prob)
		else:
			z_logits = inputs
			z_prob = tf.nn.sigmoid(z_logits)
			cdf = tf.random.uniform(shape=(tf.shape(z_logits)))
			return tf.cast(z_prob>cdf,dtype=tf.float32)

def createVbEnc(latent_dim,img_width,img_chs):
	# model
	enc_input = keras.Input(shape=(img_width,img_width,img_chs))
	x = layers.Conv2D(8,3,activation="relu",strides=2,padding="same")(enc_input)
	x = layers.Conv2D(8,3,activation="relu",strides=2,padding="same")(x)
	x = layers.Flatten()(x)
	# predicting the logits for Bernoulli
	z_logits = layers.Dense(latent_dim,activation="linear")(x)
	z = BernoulliSampling()(z_logits)
	#self.encoder = keras.Model(enc_input,[z_logits,z],name="encoder")
	return keras.Model(enc_input,[z_logits,z],name="encoder")

def createVbDec(latent_dim,img_width,img_chs,ld_chs):
	dec_input = keras.Input(shape=(latent_dim))
	lat_width = int(img_width/4) # two stack
	lat_prod = int(img_width * img_width * ld_chs / (4**2))
	x = layers.Dense(lat_prod,activation="relu")(dec_input)
	x = layers.Reshape(target_shape=(lat_width,lat_width,ld_chs))(x)
	x = layers.Conv2DTranspose(8,3,activation="relu",strides=2,padding="same")(x)
	x = layers.Conv2DTranspose(8,3,activation="relu",strides=2,padding="same")(x)
	dec_logits = layers.Conv2DTranspose(img_chs,3,activation=None,strides=1,padding="same")(x)
	return keras.Model(dec_input,dec_logits,name="decoder")

def createVgEnc(latent_dim,img_width,img_chs):
	# model
	enc_input = keras.Input(shape=(img_width,img_width,img_chs))
	x = layers.Conv2D(8,3,activation="relu",strides=2,padding="same")(enc_input)
	x = layers.Conv2D(8,3,activation="relu",strides=2,padding="same")(x)
	x = layers.Flatten()(x)
	# predicting the logits for Bernoulli
	#z_logits = layers.Dense(latent_dim,activation="linear")(x)
	#z = BernoulliSampling()(z_logits)
	z_mean = layers.Dense(latent_dim,activation="linear")(x)
	z_log_var = layers.Dense(latent_dim,activation="linear")(x)
	z = Sampling()([z_mean,z_log_var])
	#self.encoder = keras.Model(enc_input,[z_logits,z],name="encoder")
	return keras.Model(enc_input,[z_mean,z_log_var,z],name="encoder")

'''
def createVgDec(latent_dim,img_width,img_chs,ld_chs):
	dec_input = keras.Input(shape=(latent_dim))
	lat_width = int(img_width/4) # two stack
	lat_prod = int(img_width * img_width * ld_chs / (4**2))
	x = layers.Dense(lat_prod,activation="relu")(dec_input)
	x = layers.Reshape(target_shape=(lat_width,lat_width,ld_chs))(x)
	x = layers.Conv2DTranspose(8,3,activation="relu",strides=2,padding="same")(x)
	x = layers.Conv2DTranspose(8,3,activation="relu",strides=2,padding="same")(x)
	dec_logits = layers.Conv2DTranspose(img_chs,3,activation=None,strides=1,padding="same")(x)
	return keras.Model(dec_input,dec_logits,name="decoder")
'''

# Variational Bernoulli Information Bottleneck
class vBIB(keras.Model):
	def __init__(self,gamma,latent_dim,name=None,**kwargs):
		super(vBIB,self).__init__(name=name)
		self.gamma = gamma
		self.latent_dim = latent_dim
		self.img_width = 28
		self.img_ch    = 1
		self.nclass    = 10
		# model
		self.encoder = createVbEnc(latent_dim,28,1)
		# decoder
		dec_input = keras.Input(shape=(self.latent_dim))
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
			z_logits, z = self.encoder(x_data,training=True)
			logits = self.decoder(z,training=True)
			# calculate losses
			soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
			ce_loss =tf.reduce_mean(soft_out)
			# calculate kl loss
			kl_loss = tf.reduce_mean(
					tf.reduce_sum(
							tf.math.log(2.0)+z_logits *tf.nn.sigmoid(z_logits)-tf.math.softplus(z_logits), axis=1
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
	def test_step(self,inputs):
		x_data, y_label = inputs
		z_logits, z = self.encoder(x_data,training=False)
		logits = self.decoder(z,training=False)
		# losses
		soft_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,logits=logits)
		ce_loss = tf.reduce_mean(soft_out)
		# calculate kl loss
		mi_loss = tf.reduce_mean(
				tf.reduce_sum(
						tf.math.log(2.0)+z_logits * tf.nn.sigmoid(z_logits)-tf.math.softplus(z_logits), axis=1
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
		_,z = self.encoder(data,training=False)
		return self.decoder(z,training=False), z

class vBAE(keras.Model):
	def __init__(self,latent_dim,name=None,**kwargs):
		super(vBAE,self).__init__(name=name)
		self.latent_dim = latent_dim
		self.img_width = 28
		self.img_chs    = 1
		# model
		self.encoder = createVbEnc(latent_dim,self.img_width,self.img_chs)
		self.encoder.summary()
		self.decoder = createVbDec(latent_dim,self.img_width,self.img_chs,8)
		self.decoder.summary()

		self.total_loss_tracker = keras.metrics.Mean(name="total")
		self.mi_loss_tracker = keras.metrics.Mean(name="mi")
		self.bce_loss_tracker = keras.metrics.Mean(name="bce")
		self.test_total_loss_tracker = keras.metrics.Mean(name="val_total")
		self.test_mi_loss_tracker = keras.metrics.Mean(name="val_mi")
		self.test_bce_loss_tracker = keras.metrics.Mean(name="val_bce")
		self.test_mse_loss_tracker = keras.metrics.Mean(name="val_mse")
	@property
	def metrics(self):
		return [self.total_loss_tracker,
						self.mi_loss_tracker,
						self.bce_loss_tracker,
						self.test_total_loss_tracker,
						self.test_mi_loss_tracker,
						self.test_bce_loss_tracker,
						self.test_mse_loss_tracker,]
	def train_step(self,data):
		# x_in == x_out, can be supervised denoising as well
		x_in, x_out = data
		sum_axis = tf.range(1,tf.size(tf.shape(x_in)))
		with tf.GradientTape() as tape:
			z_logits, z = self.encoder(x_in,training=True)
			x_re = self.decoder(z,training=True)
			kl_loss = tf.reduce_mean(
					tf.reduce_sum(
							tf.math.log(2.0)+z_logits * tf.nn.sigmoid(z_logits) - tf.math.softplus(z_logits), axis=1
						)
				)
			bce_loss = tf.reduce_mean(
					tf.reduce_sum(
							tf.nn.sigmoid_cross_entropy_with_logits(labels=x_out,logits=x_re),axis=sum_axis,
						)
				)
			total_loss = kl_loss + bce_loss
		grads = tape.gradient(total_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
		self.total_loss_tracker.update_state(total_loss)
		self.mi_loss_tracker.update_state(kl_loss)
		self.bce_loss_tracker.update_state(bce_loss)
		return {
			"total":self.total_loss_tracker.result(),
			"bce":self.bce_loss_tracker.result(),
			"kl":self.mi_loss_tracker.result(),
		}

	def test_step(self,data):
		x_in, x_out = data
		batch_size = tf.cast(tf.shape(x_in)[0],tf.float32)
		sum_axis = tf.range(1,tf.size(tf.shape(x_in)))
		z_logits, z = self.encoder(x_in,training=False)
		x_re = self.decoder(z,training=False)
		kl_loss = tf.reduce_mean(
				tf.reduce_sum(
						tf.math.log(2.0) + z_logits * tf.nn.sigmoid(z_logits) - tf.math.softplus(z_logits), axis=1
					)
			)
		bce_loss = tf.reduce_mean(
				tf.reduce_sum(
						tf.nn.sigmoid_cross_entropy_with_logits(labels=x_out,logits=x_re),axis=sum_axis,
					)
			)
		total_loss = kl_loss + bce_loss
		# mse for comparison
		mse_loss = (2.0/batch_size) * tf.nn.l2_loss(tf.nn.sigmoid(x_re)-x_out)
		self.test_total_loss_tracker.update_state(total_loss)
		self.test_mi_loss_tracker.update_state(kl_loss)
		self.test_bce_loss_tracker.update_state(bce_loss)
		self.test_mse_loss_tracker.update_state(mse_loss)
		return {
			"total":self.test_total_loss_tracker.result(),
			"bce":self.test_bce_loss_tracker.result(),
			"kl":self.test_mi_loss_tracker.result(),
			"mse":self.test_mse_loss_tracker.result(),
		}

	def call(self,data):
		x_in = data
		_, z = self.encoder(x_in,training=False)
		return self.decoder(z,training=False), z

class vAE(keras.Model):
	def __init__(self,latent_dim,name=None,**kwargs):
		super(vAE,self).__init__(name=name)
		self.latent_dim = latent_dim
		self.img_width = 28
		self.img_chs    = 1
		# model
		#self.encoder = createVbEnc(latent_dim,self.img_width,self.img_chs)
		self.encoder = createVgEnc(latent_dim,self.img_width,self.img_chs)
		self.encoder.summary()
		# reused the decoder architecture
		self.decoder = createVbDec(latent_dim,self.img_width,self.img_chs,8)
		self.decoder.summary()

		self.total_loss_tracker = keras.metrics.Mean(name="total")
		self.mi_loss_tracker = keras.metrics.Mean(name="mi")
		self.bce_loss_tracker = keras.metrics.Mean(name="bce")
		self.test_total_loss_tracker = keras.metrics.Mean(name="val_total")
		self.test_mi_loss_tracker = keras.metrics.Mean(name="val_mi")
		self.test_bce_loss_tracker = keras.metrics.Mean(name="val_bce")
		self.test_mse_loss_tracker = keras.metrics.Mean(name="val_mse")
	@property
	def metrics(self):
		return [self.total_loss_tracker,
						self.mi_loss_tracker,
						self.bce_loss_tracker,
						self.test_total_loss_tracker,
						self.test_mi_loss_tracker,
						self.test_bce_loss_tracker,
						self.test_mse_loss_tracker,]
	def train_step(self,data):
		# x_in == x_out, can be supervised denoising as well
		x_in, x_out = data
		sum_axis = tf.range(1,tf.size(tf.shape(x_in)))
		with tf.GradientTape() as tape:
			z_mean,z_log_var, z = self.encoder(x_in,training=True)
			x_re = self.decoder(z,training=True)
			kl_loss = tf.reduce_mean(
					tf.reduce_sum(
							-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
						)
				)
			bce_loss = tf.reduce_mean(
					tf.reduce_sum(
							tf.nn.sigmoid_cross_entropy_with_logits(labels=x_out,logits=x_re),axis=sum_axis,
						)
				)
			total_loss = kl_loss + bce_loss
		grads = tape.gradient(total_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
		self.total_loss_tracker.update_state(total_loss)
		self.mi_loss_tracker.update_state(kl_loss)
		self.bce_loss_tracker.update_state(bce_loss)
		return {
			"total":self.total_loss_tracker.result(),
			"bce":self.bce_loss_tracker.result(),
			"kl":self.mi_loss_tracker.result(),
		}

	def test_step(self,data):
		x_in, x_out = data
		batch_size = tf.cast(tf.shape(x_in)[0],tf.float32)
		sum_axis = tf.range(1,tf.size(tf.shape(x_in)))
		z_mean, z_log_var, z = self.encoder(x_in,training=False)
		x_re = self.decoder(z,training=False)
		kl_loss = tf.reduce_mean(
				tf.reduce_sum(
						-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
					)
			)
		bce_loss = tf.reduce_mean(
				tf.reduce_sum(
						tf.nn.sigmoid_cross_entropy_with_logits(labels=x_out,logits=x_re),axis=sum_axis,
					)
			)
		total_loss = kl_loss + bce_loss
		# mse for comparison
		mse_loss = (2.0/batch_size) * tf.nn.l2_loss(tf.nn.sigmoid(x_re)-x_out)
		self.test_total_loss_tracker.update_state(total_loss)
		self.test_mi_loss_tracker.update_state(kl_loss)
		self.test_bce_loss_tracker.update_state(bce_loss)
		self.test_mse_loss_tracker.update_state(mse_loss)
		return {
			"total":self.test_total_loss_tracker.result(),
			"bce":self.test_bce_loss_tracker.result(),
			"kl":self.test_mi_loss_tracker.result(),
			"mse":self.test_mse_loss_tracker.result(),
		}

	def call(self,data):
		x_in = data
		_, _, z = self.encoder(x_in,training=False)
		return self.decoder(z,training=False), z

class vBQ(keras.Model):
	def __init__(self,latent_dim,src_vae,src_latent_dim,gamma,name=None,**kwargs):
		super(vBQ,self).__init__(name=name)
		self.latent_dim = latent_dim
		# fixed modules
		self.src_vae = src_vae
		self.gamma = gamma
		# model
		enc_input = keras.Input(shape=(src_latent_dim,))
		x = layers.Dense(64,activation=None)(enc_input)
		#x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)
		x = layers.Dense(64,activation=None)(x)
		#x = layers.BatchNormalization()(x)
		z_logits = layers.Dense(latent_dim,activation=None)(x)
		z = BernoulliSampling()(z_logits)
		self.encoder = keras.Model(enc_input,[z_logits,z],name="encoder")
		self.encoder.summary()

		dec_input = keras.Input(shape=(latent_dim,))
		x = layers.Dense(64,activation=None)(dec_input)
		#x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)
		x = layers.Dense(64,activation=None)(x)
		#x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)
		dec_out = layers.Dense(src_latent_dim,activation=None)(x)
		self.decoder = keras.Model(dec_input,dec_out,name="decoder")
		self.decoder.summary()

		self.total_loss_tracker = keras.metrics.Mean(name="total")
		self.mi_loss_tracker = keras.metrics.Mean(name="mi")
		#self.bce_loss_tracker = keras.metrics.Mean(name="bce")
		self.mse_loss_tracker = keras.metrics.Mean(name="mse")
		self.test_total_loss_tracker = keras.metrics.Mean(name="val_total")
		self.test_mi_loss_tracker = keras.metrics.Mean(name="val_mi")
		#self.test_bce_loss_tracker = keras.metrics.Mean(name="val_bce")
		self.test_mse_loss_tracker = keras.metrics.Mean(name="val_mse")
	@property
	def metrics(self):
		return [self.total_loss_tracker,
						self.mi_loss_tracker,
						self.mse_loss_tracker,
						self.test_total_loss_tracker,
						self.test_mi_loss_tracker,
						#self.test_bce_loss_tracker,
						self.test_mse_loss_tracker,]
	def train_step(self,data):
		# x_in == x_out, can be supervised denoising as well
		x_in, x_out = data
		# transform the source into embeddings
		_,_,q_in = self.src_vae.encoder(x_in,training=False) # no gradients
		batch_size = tf.cast(tf.shape(q_in)[0],tf.float32)
		sum_axis = tf.range(1,tf.size(tf.shape(q_in)))
		with tf.GradientTape() as tape:
			z_logits, z = self.encoder(q_in,training=True)
			q_re = self.decoder(z,training=True)
			kl_loss = tf.reduce_mean(
					tf.reduce_sum(
							tf.math.log(2.0)+z_logits * tf.nn.sigmoid(z_logits) - tf.math.softplus(z_logits), axis=1
						)
				)
			mse_loss = (2.0/batch_size) * tf.nn.l2_loss(tf.nn.sigmoid(q_re)-q_in)
			total_loss = kl_loss*self.gamma + mse_loss
		grads = tape.gradient(total_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
		self.total_loss_tracker.update_state(total_loss)
		self.mi_loss_tracker.update_state(kl_loss)
		self.mse_loss_tracker.update_state(mse_loss)
		return {
			"total":self.total_loss_tracker.result(),
			"mse":self.mse_loss_tracker.result(),
			"kl":self.mi_loss_tracker.result(),
		}

	def test_step(self,data):
		x_in, x_out = data
		# transform domain
		_,_,q_in = self.src_vae.encoder(x_in,training=False)
		batch_size = tf.cast(tf.shape(q_in)[0],tf.float32)
		sum_axis = tf.range(1,tf.size(tf.shape(q_in)))
		z_logits, z = self.encoder(q_in,training=False)
		q_re = self.decoder(z,training=False)
		kl_loss = tf.reduce_mean(
				tf.reduce_sum(
						tf.math.log(2.0) + z_logits * tf.nn.sigmoid(z_logits) - tf.math.softplus(z_logits), axis=1
					)
			)
		#bce_loss = tf.reduce_mean(
		#		tf.reduce_sum(
		#				tf.nn.sigmoid_cross_entropy_with_logits(labels=q_in,logits=q_re),axis=sum_axis,
		#			)
		#	)
		# mse for comparison
		mse_loss = (2.0/batch_size) * tf.nn.l2_loss(tf.nn.sigmoid(q_re)-q_in)
		total_loss = kl_loss * self.gamma + mse_loss

		self.test_total_loss_tracker.update_state(total_loss)
		self.test_mi_loss_tracker.update_state(kl_loss)
		#self.test_bce_loss_tracker.update_state(bce_loss)
		self.test_mse_loss_tracker.update_state(mse_loss)
		return {
			"total":self.test_total_loss_tracker.result(),
			#"bce":self.test_bce_loss_tracker.result(),
			"kl":self.test_mi_loss_tracker.result(),
			"mse":self.test_mse_loss_tracker.result(),
		}

	def call(self,data):
		x_in = data
		_,_,q_in = self.src_vae.encoder(x_in,training=False)
		_, z = self.encoder(q_in,training=False)
		return self.decoder(z,training=False), z