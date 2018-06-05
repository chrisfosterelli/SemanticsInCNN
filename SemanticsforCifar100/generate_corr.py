from keras.models import load_model
import numpy as np
import h5py
import Activations
import os
import glob
import sys
import argparse
import keras
import time
from keras.utils import plot_model
from scipy.stats.stats import pearsonr
from keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint
)
from keras.datasets import cifar100
from sklearn.utils import shuffle
from keras.layers import (
    Activation,
    Input,
    Dense,
    Flatten
)
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
#from keras.utils.visualize_util import plot
from keras.utils import np_utils
from sklearn.externals import joblib
from fractalnet import fractal_net
#

NB_CLASSES = 100
NB_EPOCHS = 400
LEARN_START = 0.02
BATCH_SIZE = 128
MOMENTUM = 0.9

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print X_train[0].shape
sorted_test_images={}
for i in range(len(y_test)):
	if y_test[i][0] not in sorted_test_images:
		sorted_test_images[y_test[i][0]]=[X_test[i]]
	else:
		sorted_test_images[y_test[i][0]].append(X_test[i])

#for item in sorted_test_images:
#	print item, len(sorted_test_images[item])



sorted_train_images={}
for i in range(len(y_train)):
	if y_train[i][0] not in sorted_train_images:
		sorted_train_images[y_train[i][0]]=[X_train[i]]
	else:
		sorted_train_images[y_train[i][0]].append(X_train[i])

#for item in sorted_train_images:
#	print item, len(sorted_train_images[item])

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

train_paths =[[] for i in range(5)]
for i in range(100):
	data=sorted_train_images[i][0:5]
	train_paths[0].append(data[0])
	train_paths[1].append(data[1])
	train_paths[2].append(data[2])
	train_paths[3].append(data[3])
	train_paths[4].append(data[4])
	
test_paths =[[] for i in range(5)]
for i in range(100):
	data=sorted_test_images[i][0:5]
	test_paths[0].append(data[0])
	test_paths[1].append(data[1])
	test_paths[2].append(data[2])
	test_paths[3].append(data[3])
	test_paths[4].append(data[4])

print len(test_paths[0])


print ("data loaded")


def build_network(deepest=False):
    dropout = [0., 0.1, 0.2, 0.3, 0.4]
    conv = [(64, 3, 3), (128, 3, 3), (256, 3, 3), (512, 3, 3), (512, 2, 2)]
    input= Input(shape=(32, 32,3))
    output = fractal_net(
        c=3, b=5, conv=conv,
        drop_path=0.15, dropout=dropout,
        #drop_path=0.15, dropout=None,
        deepest=deepest)(input)
    output = Flatten()(output)
    #output = Dense(NB_CLASSES, init='he_normal')(output)
    output=Dense(NB_CLASSES, kernel_initializer='he_normal')(output)
    output = Activation('softmax')(output)
    #model = Model(input=input, output=output)
    model = Model(inputs=input, outputs=output)
    optimizer = SGD(lr=LEARN_START, momentum=MOMENTUM)
    #optimizer = RMSprop(lr=LEARN_START)
    #optimizer = Adam()
    #optimizer = Nadam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    #plot(model, to_file='model.png')
    return model

def get_act_vector(x,model,layer):
    x = np.expand_dims(x, axis=0)
    act=Activations.get_activations(model, x, print_shape_only=True, layer_name=layer)
    return act

net = build_network(deepest="deepest")
#plot_model(net, to_file='FracNet.png',show_shapes=True)
#print net.summary()
#layers =['join_layer_'+ str(i) for i in range(1,11)]
#layers.append('flatten_1')
#layers.append('dense_1')
#layers.insert(0,'conv2d_1')
layers=['flatten_1']
#epoch ='0297'
#epoch ='0001'
epoch=str(sys.argv[1])
print (epoch)
#epochs=["%04d" % x for x in range(375,401)]
mode='train'
#load the model of a particular epoch

#model_location="/Volumes/HDD/FractNET_Models/weights."+epoch+".h5"
model_location="/home/dhanushd/scratch/snapshots/weights."+epoch+".h5"
weights= h5py.File(model_location, 'r')
net.load_weights(model_location)
for layer in layers:
	a=time.time()
	for i in range(5):
		if mode=='train':
			paths = train_paths[i]
		else:
			paths = test_paths[i]	
		CNN_vector = []
		count=0
		for path in paths:
			vec= get_act_vector(path,net,layer)[0]
			vec=vec.flatten()
			vec = vec.tolist()
			CNN_vector.append(vec)

			#Get the Pearson Correlation
		word_vector = np.array(CNN_vector)
		length = word_vector.shape[0]    
		# get the length of the word vector
		input_mat = np.empty((length, length))      
		input_mat.fill(0)                       # initialize the mattrix made by input word vecto
		# calculating correlation and generate the matrix

		for word1 in range (0,length):
			vector1 = word_vector[word1,:]
			for word2 in range (0,length):
				vector2 = word_vector[word2,:]
				input_mat[word1][word2] = pearsonr(vector1, vector2)[0]
		#print (input_mat.shape)
		#name= '/Volumes/HDD/CNNMAT_FRACNET/FracNet_' +mode+"_"+epoch+"_"+ layer + '_'+ str(i)+'.npy'
		name= '/home/dhanushd/scratch/ResultsTrain/FracNet_' +mode+"_"+epoch+"_"+ layer + '_'+ str(i)+'.npy'
		joblib.dump(input_mat,name)
	print (time.time()-a)

