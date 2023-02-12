import numpy as np
import tensorflow as tf
from tensorflow import keras
import model as mds
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vib","nib","vbib"])
parser.add_argument("--latent_dim", type=int, default=16, help="Latent Dimension")
parser.add_argument("--batch_size", type=int, default=256, help="mini-batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs (no early stopping)")
parser.add_argument("--debug", action="store_true", default=False, help="loading 10 smaples only")
parser.add_argument("--gamma_min", type=float, default=1e-2, help="minimum Trade-off parameter")
parser.add_argument("--grid", type=int, default=50, help="number of gamma values to run")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for ADAM")
parser.add_argument("--nrun", type=int, default=10, help="number of trials for each setting")


args = parser.parse_args()
argsdict = vars(args)
print(argsdict)

gamma_range =np.geomspace(args.gamma_min,1,num=args.grid)
num_classes = 10
input_shape = (28,28,1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train,-1)
x_test  = np.expand_dims(x_test,-1)

y_train = y_train.astype("int32")
y_test  = y_test.astype("int32")


debug = args.debug
if debug:
	x_train = x_train[:10,...]
	x_test  = x_test[:10,...]
	y_train = y_train[:10,...]
	y_test  = y_test[:10,...]

# keep the best result for each run
class ModelRecorder(keras.callbacks.Callback):
	def __init__(self):
		super(ModelRecorder,self).__init__()
		self.best_weights = None
	def on_train_begin(self,logs=None):
		self.best_loss = np.Inf
		self.best_logs = None
	def on_train_end(self, logs=None):
		if self.best_logs:
			self.model.set_weights(self.best_weights)
	def on_epoch_end(self, epoch, logs=None):
		if not logs.get("val_total_loss",False):
			# TODO: add something you want
			pass
		else:
			if logs['val_total_loss'] < self.best_loss:
				self.best_loss = logs['val_total_loss']
				self.best_weights = self.model.get_weights()
				self.best_logs = {"best_epoch":epoch,**logs}
def valFormatter(item):
	if type(item) == float:
		return "{:.5f}".format(item)
	elif type(item) == np.float64 or type(item) == np.float32:
		return "{:.5f}".format(item)
	else:
		return "{:}".format(item)
hdr_tex = "gamma,nrun,test_acc,train_acc,val_acc,train_ce,train_mi,val_ce,val_mi,test_ce,test_mi,best_epoch"
for gidx, gamma in enumerate(gamma_range):
	for nidx in range(args.nrun):
		if args.model == "vib":
			model = mds.vIB(gamma,args.latent_dim)
		elif args.model == "nib":
			model = mds.nIB(gamma,args.latent_dim)
		elif args.model == "vbib":
			model = mds.vBIB(gamma,args.latent_dim)
		else:
			sys.exit("Error: undefined model {:}, abort".format(args.model))

		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),run_eagerly=debug)
		tmp_recoder = ModelRecorder()
		history = model.fit(x_train,y_train,
			batch_size=args.batch_size,
			epochs=args.epochs,
			#validation_data=(x_test,y_test),
			validation_split=0.1,
			shuffle=True,
			verbose=0,
			callbacks=[tmp_recoder],
			)
		eva_out = model.evaluate(x_test,y_test,
			batch_size=args.batch_size,
			verbose=0,
			return_dict=True)
		if gidx == 0 and nidx ==0:
			print(hdr_tex)
		print(",".join([
				valFormatter(item) for item in [gamma,nidx,
				eva_out['accuracy'],
				tmp_recoder.best_logs['accuracy'],
				tmp_recoder.best_logs['val_accuracy'],
				tmp_recoder.best_logs['ce_loss'],
				tmp_recoder.best_logs['mi_loss'],
				tmp_recoder.best_logs['val_ce_loss'],
				tmp_recoder.best_logs['val_mi_loss'],
				eva_out['ce_loss'],
				eva_out['mi_loss'],
				tmp_recoder.best_logs['best_epoch']
				]
			]))
		del model


