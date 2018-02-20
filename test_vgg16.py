#This program focus on study of ResNet.

import scipy.io as sio
import numpy as np
from scipy.stats.stats import pearsonr
import sys
import vgg16
#import vgg19
#import resnet50
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
import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys

import time

import os

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # Fall back on pydot if necessary.
    try:
        import pydot
    except ImportError:
        pydot = None


def _check_pydot():
    if not (pydot and pydot.find_graphviz()):
        raise ImportError('Failed to import pydot. You must install pydot'
                          ' and graphviz for `pydotprint` to work.')


def model_to_dot(model, show_shapes=False, show_layer_names=True):
    """Converts a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """
    from ..layers.wrappers import Wrapper
    from ..models import Sequential

    _check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label,fillcolor="green")
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True):
    dot = model_to_dot(model, show_shapes, show_layer_names)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)

word_vec_in ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/"
vocab = joblib.load("./vocab.pkl")
#vocab = joblib.load("/Users/Dhanush/Desktop/Projects/CNNSTUDYDOCS/Pickels/vocab.pkl")
size_words = len(vocab)
print (len(vocab))
def get_act_vector(path,model,layer):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #preds = model.predict(x)
    model_inputs = x
    act=get_activations(model, model_inputs, print_shape_only=True, layer_name=layer)
    #print act
    #print('Predicted:', 
    #predictions = decode_predictions(preds)
    #print (type(act[0]))
    #print (act)
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
				input_words[word] = (map(float, tokens))
				added_word[word] = 0
	#print len(dictionary)
	# find words that is in dictionary but not in the input, record their indexs for making a mask
	for i, line in enumerate(vocab):
		if line not in input_words: 
			unavailable.append(i) 
	keylist = input_words.keys()
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
	PartA =[]
	PartB=[]
	Diff =[]
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
			PartB.append(part_b)
			PartA.append(part_a)
			if part_a > part_b:
				Diff.append(part_a - part_b)
				s += 1		

	print (total)
	raise Exception("")
	return s/float(total),PartA,PartB,Diff





def run_test(obj,cnn_mat):
	input_mat = obj['input_mat']
	mask = obj['mask']
	length = obj['length']
	
	cnn_mat = np.reshape(cnn_mat[mask], (length, length))
	score = two_vs_two(input_mat, cnn_mat, length)
	
	return score

path_list =[]
path_list.append(joblib.load("./path1.pkl"))
path_list.append(joblib.load("./path2.pkl"))
path_list.append(joblib.load("./path3.pkl"))
path_list.append(joblib.load("./path4.pkl"))
path_list.append(joblib.load("./path5.pkl"))
#word_vector_objects=joblib.load("/Users/Dhanush/Desktop/Projects/CNNSTUDYDOCS/Pickels/word_vectors.pkl")





#for j in range(4):
#	word_vector_objects[j]

#Total params: 143,667,240.0
network ='vgg16'

#,
#layers =['block1_conv1','block2_conv1','block3_conv1','block3_conv3','block4_conv1','block4_conv3','block5_conv1','block5_conv3','fc1','fc2']
z=0
test_scores =[]
ADUMP=[]
BDUMP=[]
CDUMP=[]
layers=[str(sys.argv[1])]
print ("The passed argument is: ")
print (layers)
for layer in layers:
	print ("vgg16"," ",layer)
	print ("processing activations at" + str(time.time()) +".....")
	model = vgg16.VGG16(include_top=True, weights='imagenet')

	for i in range(5):
		paths = path_list[i];print (i);
		CNN_vector = []
		ww=0
		for path in paths:
			vec= get_act_vector(path,model,layer)[0]
			vec=vec.flatten()
			vec = vec.tolist()
			CNN_vector.append(vec)
			#Get the Pearson Correlation
			ww+=1
			print (ww)
		word_vector = np.array(CNN_vector)
		length = word_vector.shape[0]       
		# get the length of the word vector
		input_mat = np.empty((length, length))      
		input_mat.fill(0)                       # initialize the mattrix made by input word vector
		
		# calculating correlation and generate the matrix
		print ("generating pearson correlations")
		for word1 in range (0,length):
			vector1 = word_vector[word1,:]
			for word2 in range (0,length):
				vector2 = word_vector[word2,:]
				input_mat[word1][word2] = pearsonr(vector1, vector2)[0]
		print (input_mat.shape)
		
		name = 'vgg16_'+layer+'_'+str(i)+'.npy'
		joblib.dump(input_mat,"/home/dhanushd/vgg16dump/"+name)
		"""
		input_mat=joblib.load("/Users/Dhanush/Desktop/dump/"+name)
		input_file_list =['Skip_gram','RNN','Cross_lingual','glove']#,'Non-Distributional']
		for j in range(4):
			score,aa,bb,cc=run_test(word_vector_objects[j],input_mat)
			aa.insert(0,input_file_list[j])
			bb.insert(0,input_file_list[j])
			cc.insert(0,input_file_list[j])
			aa.insert(0,layer)
			bb.insert(0,layer)
			cc.insert(0,layer)
			ADUMP.append(aa)
			BDUMP.append(bb)
			CDUMP.append(cc)
			test_scores.append([network,z,layer,input_file_list[j],score])
			#print (aa,bb,cc)
		"""

	#z= z+1

"""
#joblib.dump(test_scores,"/home/dhanushd/test_scores_vgg16.pkl")
#joblib.dump(ADUMP,"/home/dhanushd/dump/vgg16_parta.pkl")
#joblib.dump(BDUMP,"/home/dhanushd/dump/vgg16_partb.pkl")
#joblib.dump(CDUMP,"/home/dhanushd/dump/vgg16_diff.pkl")
joblib.dump(ADUMP,"/Users/Dhanush/Desktop/vgg16_parta.pkl")
joblib.dump(BDUMP,"/Users/Dhanush/Desktop/vgg16_partb.pkl")
"""
		
