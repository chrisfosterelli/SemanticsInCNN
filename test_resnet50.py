#This program focus on study of ResNet.

#import scipy.io as sio
import numpy as np
from scipy.stats.stats import pearsonr
import sys
import resnet50
#import inception_v3
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

vocab = joblib.load("./vocabSkipGram.pkl")
size_words = len(vocab)
print (size_words)
def get_act_vector(path,model,layer):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #preds = model.predict(x)
    #print('Predicted:', decode_predictions(preds))
    #model_inputs = x
    model_inputs = x
    act=get_activations(model, model_inputs, print_shape_only=True, layer_name=layer)
    #print act
    #print (type(act[0]))
    #print (act)
    return act



path_list =[]

path_list.append(joblib.load("./lpath1.pkl"))
path_list.append(joblib.load("./lpath2.pkl"))
path_list.append(joblib.load("./lpath3.pkl"))
path_list.append(joblib.load("./lpath4.pkl"))
path_list.append(joblib.load("./lpath5.pkl"))


#Total params: 143,667,240.0
network ='resnet50'

#,
layers =['conv1','res2a_branch2a','res2b_branch2a','res2c_branch2a','res3a_branch2a','res3b_branch2a','res3c_branch2a','res3d_branch2a','res4a_branch2a','res4b_branch2a','res4c_branch2a','res4d_branch2a','res4e_branch2a','res4f_branch2a','res5a_branch2a','res5b_branch2a','res5c_branch2a']

z=0
test_scores =[]
layers=['activation_']


layers = ['activation_'+str(i) for i in range(1,50)]

for layer in layers:
	print ("resnet50"," ",layer)
	model = resnet50.ResNet50(include_top=True, weights='imagenet')
	for i in range(5):
		paths = path_list[i]
		CNN_vector = []
		#w=0
		for path in paths:
			#print ("Image: %s" %(str(vocab[w])))
			#w+=1
			vec= get_act_vector(path,model,layer)[0]

			vec=vec.flatten()
			vec = vec.tolist()
			CNN_vector.append(vec)

		word_vector = np.array(CNN_vector)
		length = word_vector.shape[0]       
		# get the length of the word vector
		input_mat = np.empty((length, length))      
		input_mat.fill(0)                       # initialize the mattrix made by input word vector
		
		# calculating correlation and generate the matrix

		for word1 in range (0,length):
			vector1 = word_vector[word1,:]
			for word2 in range (0,length):
				vector2 = word_vector[word2,:]
				input_mat[word1][word2] = pearsonr(vector1, vector2)[0]	
		print (i)
		name= '/home/dhanushd/resdump/ResNet50_' + layer + '_'+ str(i)+'.npy'
		joblib.dump(input_mat,name)
		continue
		input_file_list =['Skip_gram','RNN','Cross_lingual','glove']
		for j in range(4):
			score=run_test(word_vector_objects[j],input_mat)
			print (score)
			test_scores.append([network,z,layer,input_file_list[j],score])


	z= z+1
	#print (test_scores)

#joblib.dump(test_scores,"/home/dhanushd/dump/test_scores_resnet50_Activations.pkl")

		
