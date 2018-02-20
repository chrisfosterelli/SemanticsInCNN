import sys
import inception_v3
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import glob
import os
import random
from scipy.stats.stats import pearsonr
from keras.preprocessing import image
import Activations
from keras.layers import Input
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import time

#layers =['mixed0']
#word_vec_in ="/home/dhanushd//Word_Vectors/"
vocab = joblib.load("./vocabSkipGram.pkl")
size_words = len(vocab)
print (size_words)
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

path_list =[]
path_list.append(joblib.load("./lpath1.pkl"))
path_list.append(joblib.load("./lpath2.pkl"))
path_list.append(joblib.load("./lpath3.pkl"))
path_list.append(joblib.load("./lpath4.pkl"))
path_list.append(joblib.load("./lpath5.pkl"))
#word_vector_objects=joblib.load("/home/dhanushd/Paths/word_vectors.pkl")

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
    act=Activations.get_activations(model, model_inputs, print_shape_only=True, layer_name=layer)
    return act

#Total params: 143,667,240.0
network ='InceptionV3'

#,
#layers =['conv2d_1','conv2d_10','conv2d_21','conv2d_31','conv2d_41','conv2d_51','conv2d_61','conv2d_71','conv2d_81','conv2d_91','avg_pool']
z=0
test_scores =[]
layers=[str(sys.argv[1])]
print ("The passed argument is: ")
print (layers)
for layer in layers:
	print ("InceptionV3"," ",layer)
	print ("processing activations at" + str(time.time()) +".....")
	model = inception_v3.InceptionV3(include_top=True, weights='imagenet')
	#print model.summary()
	#raise Exception("")
	for i in range(5):
		paths = path_list[i]
		CNN_vector = [];ww=0
		for path in paths:
			vec= get_act_vector(path,model,layer)[0]
			vec=vec.flatten()
			vec = vec.tolist()
			CNN_vector.append(vec)
			#Get the Pearson Correlation
		word_vector = np.array(CNN_vector)
		length = word_vector.shape[0]
		ww+=1;print (ww);       
		# get the length of the word vector
		input_mat = np.empty((length, length))      
		input_mat.fill(0)                       # initialize the mattrix made by input word vecto
		# calculating correlation and generate the matrix
		kk=0
		for word1 in range (0,length):
			vector1 = word_vector[word1,:]
			for word2 in range (0,length):
				vector2 = word_vector[word2,:]
				input_mat[word1][word2] = pearsonr(vector1, vector2)[0]
		print (input_mat.shape)
		name= '/home/dhanushd/V3dump/inceptionv3_' + layer + '_'+ str(i)+'.npy'
		joblib.dump(input_mat,name)
		continue
		input_file_list =['Skip_gram','RNN','Cross_lingual','glove','Non-Distributional']
		for j in range(4):
			score=run_test(word_vector_objects[j],input_mat)
			test_scores.append([network,z,layer,input_file_list[j],score])


	#z= z+1
#joblib.dump(test_scores,"/home/dhanushd/test_scores_InceptionV3_"+layer+".pkl")
