import tensorflow as tf
import numpy as np

def pairwiseDistance(xin):
	m64 = tf.cast(xin,tf.float64)
	xsq = tf.reduce_sum(tf.square(m64),1,keepdims=True)
	dist = xsq - m64 @ tf.transpose(m64) + tf.transpose(xsq)
	dist = tf.cast(dist,tf.float32)
	dist = tf.nn.relu(dist)
	return dist

def gmEstimate(dist,vari):
	ndim = tf.cast(tf.shape(dist)[0],tf.float32)
	norm_dist = -0.5 * dist/vari
	logsumexp = tf.reduce_logsumexp(norm_dist,1)
	return -1* tf.reduce_mean(logsumexp) + tf.math.log(ndim)