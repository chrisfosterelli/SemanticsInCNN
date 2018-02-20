#Running Prediction for Various Networks

import numpy as np
from scipy.stats.stats import pearsonr
import sys
#import resnet50
import inception_v3
#import vgg16
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import numpy as np
from sklearn.externals import joblib
from Activations import *
import glob
import os
import random
from scipy.stats.stats import pearsonr
from keras.preprocessing import image
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from Activations import *
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys
import time
from shutil import copyfile


#This program decodes the predict for Vgg16, Inception V3 and resnet50 and does the following

#It stores the top 200 wrong predictions with its original predictions if :
#if the predicted new word is not in the reserved vocab but in 738 remaining words of word2vec.
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

#Load the vocabulary of 838 concepts
vocab=joblib.load('/Users/Dhanush/Desktop/Cnn2Word2Vec/vocabSkipGram.pkl')
vocab.sort()
print ("Total SkipGram Vocabulary is : ", len(vocab))
#Load the 100 reserved concepts

reserved=joblib.load('/Users/Dhanush/Desktop/Cnn2Word2Vec/fixedwordvocab.pkl')
print ("Total reserved vocab is: ",len(reserved))

#Load the models
#model = vgg16.VGG16(include_top=True, weights='imagenet')
#model = resnet50.ResNet50(include_top=True, weights='imagenet')
model = inception_v3.InceptionV3(include_top=True, weights='imagenet')
#Load the image paths, use only first image out of 5 (lpath1)
paths =joblib.load('/Users/Dhanush/Desktop/Cnn2Word2Vec/lpath1.pkl')
print ("Total number of images is ",len(paths))
correct=0
#Lets first find all correct predictions
correct_predictions =[]
for i in range(len(paths)):
	img = image.load_img(paths[i], target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	out= decode_predictions(preds)[0]
	topprediction= out[0][1].lower().strip().replace("-"," ")
	topprediction=topprediction.replace("_"," ")
	print (topprediction)
	raise Exception("stop")
	sys.exit()
	if topprediction==vocab[i]: #It got this correct
		correct_predictions.append(topprediction)
		correct+=1
		continue

predictions ={}
incorrect=0
for i in range(len(paths)):
	print ('============================================')
	print ("The current word is ", vocab[i])
	print ("The current path is ", paths[i])
	if vocab[i] in correct_predictions:
		print ("correct_predictions word... Skipping")
		continue
	img = image.load_img(paths[i], target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	out= decode_predictions(preds)[0]
	topprediction= out[0][1].lower().strip().replace("-"," ")
	topprediction=topprediction.replace("_"," ")
	print ("top prediction is ", topprediction)
	if topprediction==vocab[i]: #It got this correct
		print ("here")
		continue

	#From here we have the incorrect predictions
	if topprediction in correct_predictions:
		print ("correct_predictions word predicted... Skipping")
		continue
	if topprediction not in vocab:
		print ("Predicted not in Vocab... Skipping")
		continue

	predictions[vocab[i]] = [topprediction,out[0][2]]
	#print (out[0][2])

print (len(predictions))


joblib.dump([correct_predictions,predictions],'inceptionv3_predictions.pkl')





