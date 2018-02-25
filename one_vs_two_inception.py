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

from random import shuffle

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def get_act_vector(path,model,layer):
    img = image.load_img(path, target_size=(299, 299))
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


#This program performs one vs two test for all the three neural networks. Does the one vs two and also 
#permutations test for each layer.


#This program will be designed to do the test layer by layer and would be a parameter passed to it.
#It outputs the one vs two score, number of correct out of 100. Its Permutation scores
#The permutation is done by shuffling the word vectors 1000 times and repeating the observation


#Lets load the correct predicted and wrong predicted vocabulary

[correct,predictions]= joblib.load('./inceptionv3_predictions.pkl')
vocab = joblib.load("./vocabSkipGram.pkl")
paths=joblib.load("./lpath3.pkl")
correct.sort()
print (correct)
#Lets load the word vectors for using in this test.
skipgram=joblib.load('./wordvectors/Skip_gram_corrected_838.pkl')

temp={}
for item in skipgram:
	temp[item]=list(skipgram[item])
skipgram=temp

correct_skipgram = []

for concept in correct:
	correct_skipgram.append(skipgram[concept])
correct_skipgram_vector=np.array(correct_skipgram)
#Now we have the correct skipgram vectors
print ("Done with correct Skipgram vector  and its shape is: ",correct_skipgram_vector.shape)
incorrect =[]
misclassified_word =[]
for item in predictions:
	incorrect.append(item)
	misclassified_word.append(predictions[item][0])
#We need to get the activations for the Correct words as well


#Get the activations for incorrect 
model = inception_v3.InceptionV3(include_top=True, weights='imagenet')
layer=str(sys.argv[1])
print ("The passed layer is: ",layer)
correct_cnn_vector=[]
for i in range(len(paths)):
	if vocab[i] not in correct:
		continue
	vec= get_act_vector(paths[i],model,layer)[0]
	vec=vec.flatten()
	vec = vec.tolist()
	correct_cnn_vector.append(vec)

correct_cnn_vector=np.array(correct_cnn_vector)
print ("Done with correct CNN vector activations and its shape is: ",correct_cnn_vector.shape)

#Now we get activations for incorrect concepts.
incorrect_cnn_vectors =[]

for i in range(len(paths)):
	if vocab[i] not in incorrect:
		continue
	vec= get_act_vector(paths[i],model,layer)[0]
	vec=vec.flatten()
	vec = vec.tolist()
	incorrect_cnn_vectors.append(vec)
#Now we have all the incorrect cnn vectors.
print ("We have all the incorrect cnn vectors and its length is: ", len(incorrect_cnn_vectors))

length=correct_cnn_vector.shape[0]
#Get the pearson correlations of the incorrect cnn vectors with the correct cnn vectors
#resulting in 150 * 100 correlation matrix
cnn_correlation_matrix =[]
for i in range(len(incorrect)):
	input_mat = np.empty((1,length))
	input_mat.fill(0)
	#print input_mat.shape
	vector2=np.array(incorrect_cnn_vectors[i])
	for word1 in range (0,length):
		vector1 = correct_cnn_vector[word1]
		input_mat[0][word1]=pearsonr(vector1, vector2)[0]
	cnn_correlation_matrix.append(input_mat)
print ("We have the cnn correlation matrix and its length is: ", len(cnn_correlation_matrix))

correct_class_correlation_matrix=[]

for i in range(len(incorrect)):
	input_mat = np.empty((1,length))
	input_mat.fill(0)
	vector2=np.array(list(skipgram[incorrect[i]]))
	for word1 in range (0,length):
		vector1 = correct_skipgram_vector[word1]
		input_mat[0][word1]=pearsonr(vector1, vector2)[0]
	correct_class_correlation_matrix.append(input_mat)
print ("We have the correct class skipgram correlation matrix and its length is: ", len(correct_class_correlation_matrix))

wrong_class_correlation_matrix =[]
for i in range(len(incorrect)):
	input_mat = np.empty((1,length))
	input_mat.fill(0)
	vector2=np.array(skipgram[misclassified_word[i]])
	for word1 in range (0,length):
		vector1 = correct_skipgram_vector[word1]
		input_mat[0][word1]=pearsonr(vector1, vector2)[0]
	wrong_class_correlation_matrix.append(input_mat)
print ("We have the incorrect prediction skipgram correlation matrix and its length is: ", len(wrong_class_correlation_matrix))



#Computationally expensive steps are all completed by now
passed=0
total=0
store= []
correct_class_correlation_matrix=np.array(correct_class_correlation_matrix)
wrong_class_correlation_matrix=np.array(wrong_class_correlation_matrix)
cnn_correlation_matrix=np.array(cnn_correlation_matrix)
for i in range(len(incorrect)):
	total+=1
	correct_class_corr_wv = correct_class_correlation_matrix[i]
	incorrect_class_corr_wv=wrong_class_correlation_matrix[i]
	cnn_act_corr= cnn_correlation_matrix[i]
	actual_class_correlation=pearsonr(cnn_act_corr[0],correct_class_corr_wv[0])[0]
	wrong_class_correlation=pearsonr(cnn_act_corr[0],incorrect_class_corr_wv[0])[0]
	if actual_class_correlation > wrong_class_correlation:
		passed+=1
	store.append([actual_class_correlation,wrong_class_correlation])
	#Now we find the pearson correlation

print ("One vs Two results for layer ",layer,"_", " is ",str(passed)," out of ", str(total) )

#Now dump the store which has the actual correlations, might be useful for more results.
lib ="./OneVsTwo/inceptionV3_"+layer+"_stored.pkl"
joblib.dump([passed,total,store],lib)
actual_total = total
actual_passed = passed
actual_score= passed/float(total)
#Now lets do the permutation tests.

#The CNN vector will not be shuffled and this removes the complexity of creating the expensive
#Computations required to calculate correlation matrices.
#cnn_correlation_matrix
#Lets create a copy of wordvectors
permutation_score=[]
for i in range(1000):
	print ('--------------------------------------------------------------')
	print ("This is the permutation test iteration: ",str(i+1))
	passed=0
	total=0
	np.random.shuffle(correct_class_correlation_matrix)
	#np.random.shuffle(correct_class_correlation_matrix.T)
	np.random.shuffle(wrong_class_correlation_matrix)
	#np.random.shuffle(wrong_class_correlation_matrix.T)

	for i in range(len(incorrect)):
		total+=1
		correct_class_corr_wv = correct_class_correlation_matrix[i]
		incorrect_class_corr_wv=wrong_class_correlation_matrix[i]
		cnn_act_corr= cnn_correlation_matrix[i]
		actual_class_correlation=pearsonr(cnn_act_corr[0],correct_class_corr_wv[0])[0]
		wrong_class_correlation=pearsonr(cnn_act_corr[0],incorrect_class_corr_wv[0])[0]			
		if actual_class_correlation > wrong_class_correlation:
			passed+=1
	permutation_score.append(passed/float(total))
	print ("Passed: " + str(passed)+" total: " + str(total) + " Score: " + str(passed/float(total)))

print ("Permutation tests completed")
print ("One vs Two results for layer ",layer,"_", " is ",str(actual_passed)," out of ", str(actual_total))	

lib ="./OneVsTwo/inceptionV3_"+layer+"_permutation.pkl"
joblib.dump(permutation_score,lib)


permutation_score.sort()

print (permutation_score[951:])
print ("Correct program")
