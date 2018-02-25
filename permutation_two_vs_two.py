#This program performs the permutation test for selected layers of CNN's for the two vs two test.

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
import glob
import os
import random
from scipy.stats.stats import pearsonr
import numpy as np
import sys
import time


vocab = joblib.load("/Users/Dhanush/Desktop/Projects/CNNSTUDYDOCS/Pickels/vocab.pkl")
size_words = len(vocab)
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
				input_words[word] = list((map(float, tokens)))
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

	word_vector = np.array(word_vector)

	#Temp code to random shuffle an array
	np.random.shuffle(word_vector)
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





def run_test(cnn_mat):
	obj=get_matrix_and_mask(open("./wordvectors/Skip_gram_corrected_838.txt",'r'))
	input_mat = obj['input_mat']
	mask = obj['mask']
	length = obj['length']
	
	cnn_mat = np.reshape(cnn_mat[mask], (length, length))
	score = two_vs_two(input_mat, cnn_mat, length)
	
	return score
network=str(sys.argv[1])
layer=str(sys.argv[2])
print ("The passed argument is: ")
print (network,layer)

if network=='inception':
	origin='/home/dhanushd/V3dump/inceptionv3_'
elif network=='resnet':
	origin='/home/dhanushd/V3dump/ResNet50_'
elif network =='vgg16':
	origin='/home/dhanushd/V3dump/vgg16_'
else:
	print ("Error in Network name")
	sys.exit()

permutations=[]
for i in range(5):
	name= origin + layer + '_'+ str(i)+'.npy'
	cnn_mat = joblib.load(name)
	scores =[]
	for i in range(1000):
		score=run_test(cnn_mat)
		scores.append(score)
		print (i+1," ",score)
	permutations.append(scores)
dump="./PermutationScores/"+network+"_"+layer+"_permutation.pkl"
joblib.dump(permutations,dump)

