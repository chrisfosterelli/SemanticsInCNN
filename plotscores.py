#Load the test scores and plot them based on layers
# Runs in Python 3

from sklearn.externals import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math

"""
count=0
for layer in model.layers:
	if str(layer).split('.')[3].split(" ")[0] == 'Conv2D':
		count+=1
	if layer.name in resnet50_layers:
		resnet50_layer_no.append(count)
		print (count)
"""

#Inception V3

#inceptionV3Scores = joblib.load("./scores/test_scores_inceptionv3_activations.pkl")
inceptionV3Scores = joblib.load("./scores/test_scores_inceptionv3_mixed.pkl")
#vgg16Scores = joblib.load("./scores/test_scores_vgg16.pkl")




SG=[]
RNN = []
Glove = []
Cross =[]
dist=[]
print (inceptionV3Scores)
for row in inceptionV3Scores:
#for row in resnet50Scores:
#for row in inceptionV3Scores:
	#print (row)
	if row[2] == 'Skip_gram':
		SG.append(row[3])
	if row[2] == 'Non-Distributional':
		dist.append(row[3])
	if row[2] == 'RNN':
		RNN.append(row[3])
	if row[2] == 'glove':
		Glove.append(row[3])
	if row[2] == 'Cross_lingual':
		Cross.append(row[3])								




m_SG =[]
s_RNN =[]
s_glove =[]
s_cross =[]
s_dist=[]
for i in range(0,len(SG),5):
	m_SG.append(sum(SG[i:i+5])/5)
	s_RNN.append(sum(RNN[i:i+5])/5)
	s_glove.append(sum(Glove[i:i+5])/5)
	s_cross.append(sum(Cross[i:i+5])/5)
	s_dist.append(sum(dist[i:i+5])/5)
#print (m_SG)
#print (inceptionV3_layers)
#print (len(m_SG))
#raise Exception("Stop")
#filler=[0 for i in range(len(vgg16_layers))]
sns.set()
sns.set_style("dark")
sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.0)
plt.figure(figsize=(5,5))
sns.set_style("darkgrid",{"xtick.major.size": 5})
#plt.title('Layers of InceptionV3 against two vs two test')

#Lets Pickle Here
"""
dict1 ={}
dict1['network']='InceptionV3'
dict1['SkipGram'] = m_SG
dict1['RNN'] = s_RNN
dict1['Glove'] = s_glove
dict1['Lingual'] = s_cross
dict1['layers']=inceptionV3_layers
dict1['layers_no']=inceptionV3_layers_no
joblib.dump(dict1,'InceptionV3PICKLEDATA_concatenate.pkl')
"""
#Plot below for One Vs Two Test.
#sns.plt.title('1 vs 2 Accuracy Through Layers of VGG16').set_fontsize('12')
#sns.plt.ylabel('1 vs 2 Accuracy').set_fontsize('12')
#sns.plt.xlabel('Layers of VGG16').set_fontsize('12')
#X=['B1C1','B2C1','B3C1','B4C1','B5C1','FC2']
#X_no=[1,2,3,4,5,6]
#y=[18,19,21,23,23,19]
#scores = [i/float(36) for i in y]
#plt.plot(X_no,scores)
#plt.xticks(X_no,X,ha='right')
#plt.plot(inceptionV3_layers_no,m_SG,label='SkipGram')

#For VGG plotting
# vgg16_layers =['B1C1','B1C2','B2C1','B2C2','B3C1','B3C2','B3C3','B4C1','B4C2','B4C3','B5C1','B5C2','B5C3','FC1','FC2']
# vgg16_layer_no =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# plt.plot(vgg16_layer_no,m_SG,label='Skip Gram')
# plt.plot(vgg16_layer_no,s_RNN,label='RNN')
# plt.plot(vgg16_layer_no,s_glove,label='Glove')
# plt.plot(vgg16_layer_no,s_cross,label='Cross-Lingual')
# plt.plot(vgg16_layer_no,s_dist,label='Non-Distributional')
# plt.xticks(vgg16_layer_no,vgg16_layers,ha='right')
# sns.plt.title('2 vs 2 Accuracy Through Layers of VGG16').set_fontsize('12')
# sns.plt.ylabel('2 vs 2 Accuracy').set_fontsize('12')
# sns.plt.xlabel('Layers of VGG16').set_fontsize('12')

#For ResNet50
#resnet50_layer_no= [i for i in range(1,50)]
#resnet50_layer=['ACT'+ str(i) for i in range(1,50)]
#plt.plot(resnet50_layer_no,m_SG[0:49],label='Skip Gram')
#plt.xticks(resnet50_layer_no,resnet50_layer,ha='right')
#sns.plt.title('2 vs 2 Accuracy Through Layers of ResNet50').set_fontsize('12')
#sns.plt.xlabel('Activation Layers of ResNet50').set_fontsize('12')
#sns.plt.ylabel('2 vs 2 Accuracy').set_fontsize('12')

#For InceptionV3

#inceptionV3_layers=['activation_'+ str(i) for i in range(1,95)]
#inceptionV3_layers_no=[i for i in range(1,95)]


#Inception using concatenate and initial activations
inceptionV3_layers=['ACT'+ str(i) for i in range(1,6)]
inceptionV3_layers.extend(['Mixed'+str(i) for i in range(0,11)])
inceptionV3_layers.append('avg_pool')
inceptionV3_layers_no= [ i for i in range(1,len(inceptionV3_layers)+1)]

plt.plot(inceptionV3_layers_no,m_SG,label='Skip Gram')
plt.xticks(inceptionV3_layers_no,inceptionV3_layers,ha='right')
sns.plt.title('2 vs 2 Accuracy Through Layers of InceptionV3').set_fontsize('12')
sns.plt.xlabel('Layers of InceptionV3').set_fontsize('12')
sns.plt.ylabel('2 vs 2 Accuracy').set_fontsize('12')








plt.gcf().subplots_adjust(bottom=0.20)
sns.plt.legend()
plt.xticks(rotation=45)
plt.savefig("/Users/Dhanush/Desktop/InceptionV3Mixed.png", dpi=300)







