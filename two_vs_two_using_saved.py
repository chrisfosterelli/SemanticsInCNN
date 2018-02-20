#This program focus on study of InceptionV3.
import scipy.io as sio
import numpy as np
from scipy.stats.stats import pearsonr
import sys
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import numpy as np
from sklearn.externals import joblib
import glob
import os
import random
from scipy.stats.stats import pearsonr
from keras.engine.topology import get_source_inputs
import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys

import time

#For VGG 16, its 553 for all 5 word vectors
vocab=joblib.load('./vocab.pkl')

#vocab=joblib.load('./vocabSkipGram.pkl')

size_words = len(vocab)
def get_act_vector(path,model,layer):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #preds = model.predict(x)
    model_inputs = x
    act=Activations.get_activations(model, model_inputs, print_shape_only=True, layer_name=layer)
    return act

def get_matrix_and_mask(vector_file):
	unavailable = []	# list of indexes of word in brain data that did not appear in the input
	word_vector = []	# input word vector

	# dic for dictionary
	dictionary = {}
	for line in vocab:
		dictionary[line] = 0
	# dic for input vectors
	input_words = {}
	# filter out words from the input that is not in the dictionary
	count = 0
	added_word = {}
	for index, line in enumerate(vector_file):
		tokens = line.strip().split()
		word = tokens.pop(0)
		count+=1;
		word = word.lower()										
		if word in dictionary: 
			if word not in added_word:						
				input_words[word] = (list(map(float, tokens)))
				added_word[word] = 0
	#print len(dictionary)
	# find words that is in dictionary but not in the input, record their indexs for making a mask
	for i, line in enumerate(vocab):
		if line not in input_words: 
			unavailable.append(i) 
	keylist = list(input_words.keys())
	keylist.sort()
	for key in keylist:
	    word_vector.append(input_words[key])
	#print len(keylist)
	    # print "%s: %s" % (key, input_words[key])
	# print word_vector

	word_vector = np.array(word_vector)

	#Temp code to random shuffle an array
	#np.random.shuffle(word_vector)
	# cast word vector from a list of list to an array
	length = word_vector.shape[0]		
	# get the length of the word vector
	input_mat = np.empty((length, length))		
	input_mat.fill(0)						# initialize the matrix made by input word vector

	# calculating correlation and generate the mattrix
	for word1 in range (0,length):
		vector1 = word_vector[word1]
		for word2 in range (0,length):
			vector2 = word_vector[word2]
			#print vector1
			#print vector2
			input_mat[word1][word2] = pearsonr(vector1, vector2)[0]
			#print pearsonr(vector1, vector2)[0]
	# print (input_mat)

	#create mask
	mask = np.ones((size_words,size_words), dtype=bool)
	for i in range(0, size_words):
		for j in range(0,size_words):
			if (i in unavailable) or (j in unavailable):
				mask[i,j] = False
	# print mask 

	print ("Unavailable is: " + str(len(unavailable)))
	length = len(word_vector)
	return {
		'input_mat' : input_mat,
		'mask' : mask,
		'length' : length
	}

def two_vs_two (input_mat, brain_mat, length):
	brain_1 = input_mat
	brain_2 = brain_mat
	#brain_2 = np.load(open(brain_2_name, 'r'))
	s = 0
	total = 0
	diff =0
	list1 =[]
	for line_a in range (0,length):
		b_1_a = brain_1[line_a,:]
		b_2_a = brain_2[line_a,:]

		for line_b in range (line_a+1, length):
			b_1_b = brain_1[line_b,:]
			b_2_b = brain_2[line_b,:]
			mask = np.ones(len(b_1_a), dtype=bool)
			mask[[line_a, line_b]] = False
			#b_1_a_masked = ma.masked_array(b_1_a, mask = mask)
			#b_2_a_masked = ma.masked_array(b_2_a, mask = mask)
			#b_1_b_masked = ma.masked_array(b_1_b, mask = mask)
			#b_2_b_masked = ma.masked_array(b_2_b, mask = mask)
			b_1_a_masked = b_1_a[mask]
			b_2_a_masked = b_2_a[mask]
			b_1_b_masked = b_1_b[mask]
			b_2_b_masked = b_2_b[mask]
			#print mask
			#print len(b_2_b_masked)
			#print mask

			part_a = pearsonr(b_1_a_masked, b_2_a_masked)[0] + pearsonr(b_1_b_masked, b_2_b_masked)[0]
			part_b = pearsonr(b_1_a_masked, b_2_b_masked)[0] + pearsonr(b_1_b_masked, b_2_a_masked)[0]
			# part_a = distance.cosine(b_1_a_masked, b_2_a_masked) + distance.cosine(b_1_b_masked, b_2_b_masked)
			# part_b = distance.cosine(b_1_a_masked, b_2_b_masked) + distance.cosine(b_1_b_masked, b_2_a_masked)
			
			#print part_a
			#print part_b

			total += 1
			
			if part_a > part_b:
				s += 1		

	return s/float(total)



def run_test(obj,cnn_mat):
	input_mat = obj['input_mat']
	mask = obj['mask']
	length = obj['length']
	
	cnn_mat = np.reshape(cnn_mat[mask], (length, length))
	score = two_vs_two(input_mat, cnn_mat, length)
	
	return score
word_vec_in='./wordvectors/'
"""
input_file_list =['Global_context_553.txt','Skip_gram_corrected_553.txt','RNN_553.txt','Cross_lingual_553.txt','glove_553.txt','Non-Distributional_553.txt']
word_vector_objects=[]
for j in range(5):
	input_vec = word_vec_in +str(input_file_list[j])
	output=get_matrix_and_mask(open(input_vec,'r'))
	print (output['length'])
	word_vector_objects.append(output)
joblib.dump(word_vector_objects,'word_vector_objects.pkl')
"""
word_vector_objects=joblib.load('./word_vector_objects.pkl')
#The below are for VGG16

network ='Vgg16'
loc = '/home/dhanushd/vgg16dump/'
first='vgg16_'
last='.npy'
layers =['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3','fc1','fc2']

#The below are for ResNet50
"""
network ='ResNet50'
loc = '/home/dhanushd/resdump/'
first='ResNet50_'
last='.npy'
layers=['activation_'+str(i) for i in range(1,50,1)]
layers.append('flatten_1')
"""
"""
#The below are for InceptionV3
network ='InceptionV3'
loc = '/home/dhanushd/V3dump/'
first='inceptionv3_'
last='.npy'
layers=['activation_'+str(i) for i in range(1,95,1)]
#layers=['mixed'+str(i) for i in range(0,11,1)]
#layers.append('avg_pool')
"""
test_scores =[]
for layer in layers:
	print ("Vgg16"," ",layer)
	#print ("ResNet50"," ",layer)
	#print ("InceptionV3"," ",layer)
	for i in range(5):
		#Load the Numpy matrix
		path=loc+first+layer+'_'+str(i)+last
		print (path)
		input_mat= joblib.load(path)
		input_file_list =['Global_context','Skip_gram','RNN','Cross_lingual','glove','Non-Distributional']
		for j in range(5):
			print (input_file_list[j])
			score=run_test(word_vector_objects[j],input_mat)
			print (layer,input_file_list[j],score)
			test_scores.append([network,layer,input_file_list[j],score])


joblib.dump(test_scores,"/home/dhanushd/scores/test_scores_vgg16.pkl")
#joblib.dump(test_scores,"/home/dhanushd/scores/test_scores_resnet50.pkl")
#joblib.dump(test_scores,"/home/dhanushd/scores/test_scores_inceptionv3_activation.pkl")
