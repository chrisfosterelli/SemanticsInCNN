from sklearn.externals import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#This program analyze the One vs Two test and makes plot. The points on graphs where p value is
#less than 0.05 are marked. The results are also imported into a text file and finally into excel.
#The permutation test Histograms for one of the models will be plotted in a separate file.
loc='/Volumes/LLL/2vs2_cnn_data/OneVsTwo/'
#Inception V3

first='inceptionV3_'
layers=['activation_'+str(i) for i in range(1,95,1)]
layers_no=[i for i in range(1,95,1)]

"""
#This is for mixed layers
inceptionV3_layers=['ACT'+ str(i) for i in range(1,6)]
inceptionV3_layers.extend(['Mixed'+str(i) for i in range(0,11)])
inceptionV3_layers.append('avg_pool')
inceptionV3_layers_no= [ i for i in range(1,len(inceptionV3_layers)+1)]
inc_layers=['activation_'+str(i) for i in range(1,6,1)]
mixed_layers=['mixed'+str(i) for i in range(0,11,1)]
inc_layers.extend(mixed_layers)
inc_layers.append('avg_pool')
"""



#first='vgg16_'

#ResNet50
"""
first='resnet50_'
layers_no=[i for i in range(1,49,1)]
layers=['activation_'+str(i) for i in range(1,49,1)]
resnet_layers=[str(i) for i in range(1,49,1)]
"""
#VGG16
"""
first='vgg16_'
layers =['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3','fc1','fc2']
layers_no=[i for i in range(1,16,1)]
"""
scores=[]
pvalues=[]
alayer=[]
for layer in layers:
	path=loc+first+layer+'_stored.pkl'
	try:
		data=joblib.load(path)
	except:
		continue
	score=data[0]/float(data[1])
	scores.append(score)
	try:
		path=loc+first+layer+'_permutation.pkl'
		permute=joblib.load(path)
	except:
		continue
	pvalue = sum(i > score for i in permute)
	pvalue=pvalue/float(1000)
	pvalues.append(pvalue)
	alayer.append(layer)



for i in range(len(scores)):
	print (scores[i],pvalues[i],alayer[i])	

sns.set()
sns.set_style("dark")
sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.0)
plt.figure(figsize=(5,5))
sns.set_style("darkgrid",{"xtick.major.size": 5})


#For 
#plt.plot(inceptionV3_layers_no,scores,label='Skip Gram')
plt.plot(layers_no,scores,label='Skip Gram')
#plt.xticks(inceptionV3_layers_no,inceptionV3_layers,ha='right')
sns.plt.title('1 vs 2 Accuracy Through Layers of InceptionV3').set_fontsize('12')
sns.plt.xlabel('Layers of InceptionV3').set_fontsize('12')
sns.plt.ylabel('1 vs 2 Accuracy').set_fontsize('12')

#For ResNet50
"""
plt.plot(layers_no,scores,label='Skip Gram')
plt.xticks(layers_no,resnet_layers,ha='right')
sns.plt.title('1 vs 2 Accuracy Through Layers of ResNet50').set_fontsize('12')
sns.plt.xlabel('Layers of ResNet50').set_fontsize('12')
sns.plt.ylabel('1 vs 2 Accuracy').set_fontsize('12')
"""
"""
plt.plot(layers_no,scores,label='Skip Gram')
#plt.xticks(layers_no,resnet_layers,ha='right')
sns.plt.title('1 vs 2 Accuracy Through Layers of VGG16').set_fontsize('12')
sns.plt.xlabel('Layers of VGG16').set_fontsize('12')
sns.plt.ylabel('1 vs 2 Accuracy').set_fontsize('12')
"""

plt.gcf().subplots_adjust(bottom=0.20)
sns.plt.legend()
plt.xticks(rotation=45)
plt.show()

