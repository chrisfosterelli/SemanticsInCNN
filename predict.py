#Running Prediction for Various Networks

import numpy as np
from scipy.stats.stats import pearsonr
import sys
#import resnet50
#import inception_v3
import vgg16
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
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from Activations import *

import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys

import time

from shutil import copyfile

def get_act_vector(path,model,layer):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    #print('Predicted:', decode_predictions(preds))
    model_inputs = x
    act=get_activations(model, model_inputs, print_shape_only=True, layer_name=layer)
    #print act
    #print (type(act[0]))
    #print (act)
    return act
#Only for Inception V3, use default keras Preprocessing for other networks.
"""
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
"""
word_vec_in ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/"
vocab = joblib.load("/Users/Dhanush/Desktop/Projects/CNNSTUDYDOCS/Paths/vocab.pkl")
size_words = len(vocab)
paths=joblib.load("/Users/Dhanush/Desktop/Projects/CNNSTUDYDOCS/Paths/path1.pkl")


#model = resnet50.ResNet50(include_top=True, weights='imagenet')
#model = inception_v3.InceptionV3(include_top=True, weights='imagenet')

"""
j=0
top1=0
top5=0
incorrect =[]
correct=[]
correct_vector=[]
incorrect_vector=[]
wrong_word_predicted=[]
layer='fc2'
exempt=['hog','file','rule','sidewinder','pretzel','sunscreen','racket','harmonica','ballplayer','notebook','microwave', 'ant', 'sax', 'zebra', 'impala', 'hippopotamus']
for path in paths:
	if vocab[j] in exempt:
		print (vocab[j])
		j+=1
		continue
	gtruth=vocab[j].decode('utf-8')
	#img = image.load_img(path, target_size=(299, 299))
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	out= decode_predictions(preds)[0]
	vec= get_act_vector(path,model,layer)[0]
	vec=vec.flatten()
	vec = vec.tolist()
	#print (out)
	topprediction= out[0][1]
	if (topprediction==gtruth):
		top1+=1
		correct_vector.append(vec)
		correct.append(gtruth)
	else:
		incorrect_vector.append(vec)
		incorrect.append(gtruth)
		wrong_word_predicted.append(topprediction)

	
	for pred in out:
		if pred[1] == gtruth:
			top5+=1
	
	j+=1
	



print (top1)
print (top5)
print (top1/(float(size_words)))
print (top5/(float(size_words)))
print (len(correct))
print (len(incorrect))
print (len(correct_vector))
print (len(incorrect_vector))


joblib.dump(correct,"/Users/Dhanush/Desktop/CNNStudy/correctwords.pkl")
joblib.dump(incorrect,"/Users/Dhanush/Desktop/CNNStudy/incorrectwords.pkl")
joblib.dump(correct_vector,"/Users/Dhanush/Desktop/CNNStudy/correctwordsvector.pkl")
joblib.dump(incorrect_vector,"/Users/Dhanush/Desktop/CNNStudy/incorrectwordsvector.pkl")
joblib.dump(wrong_word_predicted,"/Users/Dhanush/Desktop/CNNStudy/wrong_word_predicted.pkl")


#Lets get all the word vectors for correct.
input_vec ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/Skip_gram_corrected.txt"
file_in=open(input_vec,'r')
correct_vocab=joblib.load("/Users/Dhanush/Desktop/CNNStudy/correctwords.pkl")
print len(correct_vocab)#114
correct_vocab_word2Vec=GetCorrectWord2VecMatrix(file_in,correct_vocab) #114 * 300
print correct_vocab_word2Vec.shape
length = len(correct_vocab)#114 * 300
#Lets load all the incorrect words and get their vectors and pearson correlation with correct as dictionary and store for faster computation.



incorrect_vocab=joblib.load("/Users/Dhanush/Desktop/CNNStudy/incorrectwords.pkl")
wrong_word_predicted=joblib.load("/Users/Dhanush/Desktop/CNNStudy/wrong_word_predicted.pkl")
incorrect_and_wrong_prediction= list(set(incorrect_vocab+wrong_word_predicted))
incorrect_and_wrong_prediction.sort()
incorrect_and_wrong_prediction_word2Vec_pearson_dict ={}
print len(incorrect_and_wrong_prediction)
file_in=open(input_vec,'r')
incorrect_and_wrong_prediction_word2Vec_dict=GetwrongWord2VecDict(file_in,incorrect_and_wrong_prediction)
#print incorrect_and_wrong_prediction_word2Vec_dict

for word in incorrect_and_wrong_prediction:

	vector2=np.array(incorrect_and_wrong_prediction_word2Vec_dict[word]) #1* 300
	print vector2.shape
	input_mat = np.empty((1,length))
	input_mat.fill(0)	
	print input_mat.shape
	for word1 in range (0,length):
		vector1 = correct_vocab_word2Vec[word1]
		input_mat[0][word1]=pearsonr(vector1, vector2)[0]
	print input_mat
	print input_mat.shape
	incorrect_and_wrong_prediction_word2Vec_pearson_dict[word]=input_mat

print incorrect_and_wrong_prediction_word2Vec_pearson_dict
joblib.dump(incorrect_and_wrong_prediction_word2Vec_pearson_dict,"/Users/Dhanush/Desktop/CNNStudy/word2vecpearson.pkl")
"""

word2Vec_pearson_dict=joblib.load("/Users/Dhanush/Desktop/CNNStudy/word2vecpearson.pkl")
correct_vocab=joblib.load("/Users/Dhanush/Desktop/CNNStudy/correctwords.pkl")
incorrect_vocab=joblib.load("/Users/Dhanush/Desktop/CNNStudy/incorrectwords.pkl")
print len(incorrect_vocab)
j=0
correct_cnn_vector =[]
layer='block5_conv1'
print layer
model = vgg16.VGG16(include_top=True, weights='imagenet')
"""
for path in paths:

	if vocab[j] not in correct_vocab:
		j+=1
		continue
	print vocab[j],j
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vec= get_act_vector(path,model,layer)[0]
	vec=vec.flatten()
	vec = vec.tolist()
	correct_cnn_vector.append(vec)
	j+=1
correct_cnn_vector=np.array(correct_cnn_vector)
print correct_cnn_vector.shape
#joblib.dump(correct_cnn_vector,"/Users/Dhanush/Desktop/CNNStudy/CNNCorrectVector_"+layer+".pkl")
"""
correct_cnn_vector=joblib.load("/Users/Dhanush/Desktop/CNNStudy/CNNCorrectVector_"+layer+".pkl")
length=correct_cnn_vector.shape[0]
#By now we have all the correct classified CNN vector.

j=0
passed=0
total=0
store=[]
top_prediction_store=[]
for path in paths:
	if vocab[j] not in incorrect_vocab:
		j+=1
		continue
	total+=1
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vector2= get_act_vector(path,model,layer)[0]
	vector2=np.array(vector2)
	vector2=vector2.flatten()
	#print vector2.shape
	input_mat = np.empty((1,length))
	input_mat.fill(0)
	#print input_mat.shape
	for word1 in range (0,length):
		vector1 = correct_cnn_vector[word1]
		input_mat[0][word1]=pearsonr(vector1, vector2)[0]
	CNN_Pearson_mat=input_mat
	#print CNN_Pearson_mat.shape
	preds = model.predict(x)
	out= decode_predictions(preds)[0]
	probablity= out[0][2]
	topprediction= out[0][1]
	top_prediction_store.append(topprediction)
	print "=========================================================="
	print ("  The Correct Word ","     Top Predicted     ","    Confidence")
	print (vocab[j],"        ",topprediction,"    ",probablity)
	corr_correct_word2vec=pearsonr(CNN_Pearson_mat[0],word2Vec_pearson_dict[vocab[j]][0])[0]
	corr_incorrect_word2vec=pearsonr(CNN_Pearson_mat[0],word2Vec_pearson_dict[topprediction][0])[0]
	print (corr_correct_word2vec,corr_incorrect_word2vec,corr_correct_word2vec>corr_incorrect_word2vec)
	if corr_correct_word2vec >corr_incorrect_word2vec:
		passed+=1

	j+=1
	store.append([vocab[j],topprediction,probablity,corr_correct_word2vec,corr_incorrect_word2vec])
print "Total tests passed: " +str(passed)+ "out of: "+ str(total)
joblib.dump(top_prediction_store,"/Users/Dhanush/Desktop/CNNStudy/top_prediction_store.pkl")
#joblib.dump(store,"/Users/Dhanush/Desktop/onevstwo_"+layer+'.pkl')
