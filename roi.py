#This program would be used to study ROI's in Brain 

from sklearn.externals import joblib

data=joblib.load("/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/region_score_fmri200.p")



vectors=[ 'Skip_gram_corrected.txt', 'glove.6B.300d.txt','RNN.txt','Global_context.txt' , 'Cross_lingual.txt','Non-Distributional.txt']
rois= data['Cross_lingual.txt'].keys()
matrix={}
roi_dict={}

for roi in rois:
	roi_dict[roi] = []

for vector in vectors:
	for roi in rois:
		roi_dict[roi].append(sum(data[vector][roi])/9)


print (roi_dict)

print (len(roi_dict))

data_file=open("./data.txt",'w')

string1=" ".join(vectors)
data_file.write(string1+"\n")

for roi in rois:
	temp=[]
	temp.append(roi)
	temp.extend([str(i) for i in roi_dict[roi]])
	print (temp)
	string1=" ".join(temp)
	data_file.write(string1+"\n")
data_file.close()

	

