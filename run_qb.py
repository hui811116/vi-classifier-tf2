import numpy as np
import tensorflow as tf
from tensorflow import keras
import model as mds
import argparse
import sys
import copy
import os
import pickle

parser = argparse.ArgumentParser()
#parser.add_argument("model", choices=["vbae","vae"])
parser.add_argument("weights",type=str,default="vae.h5",help="path to the .h5 vae model weights")
parser.add_argument("--gamma",type=float,default=1e-3,help="kl-mse scaling ratio")
parser.add_argument("--latent_dim", type=int, default=64, help="Latent Dimension")
parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs (no early stopping)")
parser.add_argument("--debug", action="store_true", default=False, help="loading 10 smaples only")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for ADAM")
parser.add_argument("--save_dir",type=str,default="saved_ae_models",help="saved directory")


args = parser.parse_args()
argsdict = vars(args)
print(argsdict)

#gamma_range =np.geomspace(args.gamma_min,1,num=args.grid)
#num_classes = 10
input_shape = (28,28,1)

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train,-1)
x_test  = np.expand_dims(x_test,-1)


debug = args.debug
if debug:
	x_train = x_train[:10,...]
	x_test  = x_test[:10,...]

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
		if not logs.get("val_total",False):
			# TODO: add something you want
			pass
		else:
			if logs['val_total'] < self.best_loss:
				self.best_loss = logs['val_total']
				self.best_weights = self.model.get_weights()
				self.best_logs = {"best_epoch":epoch,**logs}
def valFormatter(item):
	if type(item) == float:
		return "{:.5f}".format(item)
	elif type(item) == np.float64 or type(item) == np.float32:
		return "{:.5f}".format(item)
	else:
		return "{:}".format(item)

# reading the saved vae model
parse_fname = ".".join(args.weights.split(".")[:-1])
with open(parse_fname+".pkl",'rb') as fid:
	load_pkl = pickle.load(fid)
cfg = load_pkl['config']


if cfg['model'] == "vbae":
	vae = mds.vBAE(cfg['latent_dim'])
elif cfg['model'] == "vae":
	vae = mds.vAE(cfg['latent_dim'])
else:
	sys.exit("Error: undefined model {:}, abort".format(cfg['model']))

# dummy pass
dummy_data = tf.zeros((1,28,28,1))
vae(dummy_data)
vae.trainable = False
vae.compile(optimizer="adam")
# the quantizer model
model = mds.vBQ(args.latent_dim,vae,cfg['latent_dim'],args.gamma)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),run_eagerly=debug)
tmp_recoder = ModelRecorder()
# no labels in autoencoders
history = model.fit(x_train,x_train,
	batch_size=args.batch_size,
	epochs=args.epochs,
	validation_data=(x_test,x_test),
	validation_split=0.1,
	shuffle=True,
	verbose=0,
	callbacks=[tmp_recoder],
	)
eva_out = model.evaluate(x_test,x_test,
	batch_size=args.batch_size,
	verbose=0,
	return_dict=True)

#print(eva_out.keys())

print("ev_mse,ev_kl,best_mse,best_kl,val_mse,val_kl,best_epoch")
print(",".join([
		valFormatter(item) for item in [
				eva_out['mse'],
				#eva_out['bce'],
				eva_out['kl'],
				#tmp_recoder.best_logs['bce'],
				tmp_recoder.best_logs['mse'],
				tmp_recoder.best_logs['kl'],
				tmp_recoder.best_logs['val_mse'],
				#tmp_recoder.best_logs['val_bce'],
				tmp_recoder.best_logs['val_kl'],
				tmp_recoder.best_logs['best_epoch'],
			]
	]))

# saving the model
'''
os.makedirs(args.save_dir,exist_ok=True)
base_name = "{:}_ld{:}_bs{:}_ep{:}".format(args.model,args.latent_dim,args.batch_size,args.epochs)
repeat_cnt =0
safe_fname = copy.copy(base_name)
while os.path.isfile(os.path.join(args.save_dir,safe_fname+".h5")):
	repeat_cnt+=1
	safe_fname = "{:}_{:}".format(base_name,repeat_cnt)
model.save_weights(os.path.join(args.save_dir,safe_fname+".h5"))
result_pkl = {"config":argsdict}
with open(os.path.join(args.save_dir,safe_fname+".pkl"),'wb') as fid:
	pickle.dump(result_pkl,fid)
print("saving the weights to:{:}".format(os.path.join(args.save_dir,safe_fname+".h5")))
'''