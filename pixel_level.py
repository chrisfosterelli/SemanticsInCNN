import sys
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import glob
import os
import random
from scipy.stats.stats import pearsonr
from PIL import image
import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import time

#layers =['mixed0']
#word_vec_in ="/home/dhanushd//Word_Vectors/"
vocab = joblib.load("./Cnn2Word2Vec/vocab.pkl")
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


def run_permutations(obj, cnn_mat, length):
	#Run permutation tests

	input_mat = obj['input_mat']
	mask = obj['mask']
	length = obj['length']
	score_permute=[]
	print ("Running Permutation test")
	cnn_mat = np.reshape(cnn_mat[mask], (length, length))
	actual_score = get_score(input_mat,  cnn_mat, mask, length)
	for i in range(1000):
		copy_wv=np.copy(input_mat)
		np.random.shuffle(copy_wv)
		np.random.shuffle(copy_wv.T)
		score_permute.append(two_vs_two(copy_wv, brain_mat, length))
	pvalue = sum(i > actual_score for i in score_permute)
	pvalue=pvalue/float(1000)
	print [actual_score,pvalue]

path_list =[]
path_list.append(joblib.load("./Cnn2Word2Vec/path1.pkl"))
path_list.append(joblib.load("./Cnn2Word2Vec/path2.pkl"))
path_list.append(joblib.load("./Cnn2Word2Vec/path3.pkl"))
path_list.append(joblib.load("./Cnn2Word2Vec/path4.pkl"))
path_list.append(joblib.load("./Cnn2Word2Vec/path5.pkl"))
word_vector_objects=joblib.load("./Cnn2Word2Vec/word_vector_objects.pkl")


for a  in range(1):

	#print model.summary()
	#raise Exception("")
	for i in range(5):
		print (i)
		paths = path_list[i]
		CNN_vector = []
		for path in paths:
			img = image.load_img(path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			x=x.flatten()
			CNN_vector.append(x)
		#Get the Pearson Correlation
		word_vector = np.array(CNN_vector)
		length = word_vector.shape[0]     
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
		name= '/Users/Dhanush/Desktop/pixel_' + str(i)+'.npy'
		joblib.dump(input_mat,name)
		continue
		input_file_list =['Skip_gram','glove','RNN','Cross_lingual']
		for j in range(4):
			print(input_file_list[j])
			#score=run_test(word_vector_objects[j],input_mat)
			run_permutations(input_mat,)
			#test_scores.append([network,z,layer,input_file_list[j],score])


	#z= z+1
#joblib.dump(test_scores,"/home/dhanushd/test_scores_InceptionV3_"+layer+".pkl")
