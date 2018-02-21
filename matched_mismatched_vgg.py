#Load the test scores and plot them based on layers
# Runs in Python 3

from sklearn.externals import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
sns.set()
sns.set_style("dark")
sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.0)
plt.figure(figsize=(6,5))
sns.set_style("darkgrid",{"xtick.major.size": 5})
vgg16_layer_no =[1,3,5,7,8,10,11,13,14,15]
matched=joblib.load("/Users/Dhanush/Desktop/vgg16_parta.pkl")
mismatched=joblib.load("/Users/Dhanush/Desktop/vgg16_partb.pkl")
SG =[]
RNN = []
Glove = []
Cross =[]

SG1 =[]
RNN1 = []
Glove1 = []
Cross1 =[]

SG2 =[]
RNN2 = []
Glove2 = []
Cross2 =[]


SGF=[]
SGL=[]
flagF=True;flagL=True;
for row1,row2 in zip(matched,mismatched):
	MAT= row1[2:]
	MIS= row2[2:]
	temp1=[];temp2=[];temp3=[];
	for i,j in zip(MAT,MIS):
		if i >j:
			temp1.append(i)
			temp2.append(j)
			temp3.append(i-j)

	if row1[1] == 'Skip_gram':
		SG.append(sum(temp1)/len(temp1))
		SG1.append(sum(temp2)/len(temp2))
		SG2.append(sum(temp3)/len(temp3))
		if (row1[0]=='block1_conv1') and flagF:
			SGF.append(temp3)
			flagF=False

		if (row1[0]=='fc2') and flagL:
			SGL.append(temp3)
			flagL=False

	if row1[1] == 'RNN':
		RNN.append(sum(temp1)/len(temp1))
		RNN1.append(sum(temp2)/len(temp2))
		RNN2.append(sum(temp3)/len(temp3))
	if row1[1] == 'glove':
		Glove.append(sum(temp1)/len(temp1))
		Glove1.append(sum(temp2)/len(temp2))
		Glove2.append(sum(temp3)/len(temp3))
	if row1[1] == 'Cross_lingual':
		Cross.append(sum(temp1)/len(temp1))
		Cross1.append(sum(temp2)/len(temp2))
		Cross2.append(sum(temp3)/len(temp3))


m_SG =[]
s_RNN =[]
s_glove =[]
s_cross =[]
for i in range(0,len(SG),5):
	#m_SG.append(sum(SG[i:i+5])/5)
	m_SG.append(SG[i])
	#print (sum(SG[i:i+5])/5)
	#s_RNN.append(sum(RNN[i:i+5])/5)
	s_RNN.append(RNN[i])
	#s_glove.append(sum(Glove[i:i+5])/5)
	s_glove.append(Glove[i])
	#s_cross.append(sum(Cross[i:i+5])/5)
	s_cross.append(Cross[i])

#print (m_SG)



plt.plot(vgg16_layer_no,m_SG,label='Matched SkipGram',color='#641E16')
plt.plot(vgg16_layer_no,s_RNN,label='Matched RNN',color='#154360')
plt.plot(vgg16_layer_no,s_glove,label='Matched Glove',color='#1F618D')
plt.plot(vgg16_layer_no,s_cross,label='Matched Cross-Lingual',color='#D35400')
#plt.ylabel('Correlation of Vectors')
#plt.xlabel('Layers of VGG16')



m_SG =[]
s_RNN =[]
s_glove =[]
s_cross =[]

for i in range(0,len(SG1),5):
	#m_SG.append(sum(SG1[i:i+5])/5)
	m_SG.append(SG1[i])
	#s_RNN.append(sum(RNN1[i:i+5])/5)
	s_RNN.append(RNN1[i])
	#s_glove.append(sum(Glove1[i:i+5])/5)
	s_glove.append(Glove1[i])
	#s_cross.append(sum(Cross1[i:i+5])/5)
	s_cross.append(Cross1[i])



plt.plot(vgg16_layer_no,m_SG,'--',label='Mismatched SkipGram',color='#641E16')
plt.plot(vgg16_layer_no,s_RNN,'--',label='Mismatched RNN',color='#154360')
plt.plot(vgg16_layer_no,s_glove,'--',label='Mismatched Glove',color='#1F618D')
plt.plot(vgg16_layer_no,s_cross,'--',label='Mismatched Cross-Lingual',color='#D35400')
sns.plt.ylabel('Correlation of Vectors').set_fontsize('12')
sns.plt.xlabel('Layers of VGG16').set_fontsize('12')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.13)
sns.plt.title('Layers of VGG16 with Matched and Mismatched Correlation Pairs').set_fontsize('12')
#plt.show()
plt.savefig("/Users/Dhanush/Desktop/Matched_Mismatched.png", dpi=300)
"""
m_SG =[]
s_RNN =[]
s_glove =[]
s_cross =[]

for i in range(0,len(SG2),5):
	#m_SG.append(sum(SG2[i:i+5])/5)
	m_SG.append(SG2[i])
	#s_RNN.append(sum(RNN2[i:i+5])/5)
	s_RNN.append(RNN2[i])
	#s_glove.append(sum(Glove2[i:i+5])/5)
	s_glove.append(Glove2[i])
	#s_cross.append(sum(Cross2[i:i+5])/5)
	s_cross.append(Cross2[i])

plt.subplot(1, 2, 2)

plt.plot(vgg16_layer_no,m_SG,label='SkipGram',color='#641E16')
plt.plot(vgg16_layer_no,s_RNN,label='RNN',color='#154360')
plt.plot(vgg16_layer_no,s_glove,label='Glove',color='#1F618D')
plt.plot(vgg16_layer_no,s_cross,label='Cross-Lingual',color='#D35400')
plt.xlabel('layers of VGG16')
plt.legend()

plt.savefig("matched_mismatched_vgg161.png", dpi=300)

print (len(SGF))
print (len(SGF[0]))

joblib.dump(SGF,"SGF",protocol=2)
joblib.dump(SGL,"SGL",protocol=2)


plt.figure()
plt.subplot(2, 1, 1)
plt.title('Diff between matched and not-matched pairs for first layer of VGG16')
plt.hist(SGF, normed=True,histtype='bar',bins=50,color='#2980B9')
plt.ylabel('Difference ')
plt.xlabel('layers of VGG16')
plt.subplot(2, 1, 2)
plt.title('Difference between matched and not-matched pairs for last layer of VGG16')
plt.hist(SGL, normed=True,histtype='bar',bins=50,color='#2980B9')
print (max(SGL))
plt.ylabel('Difference')
plt.xlabel('layers of VGG16')
#plt.show()
plt.savefig("diff_matched_mismatched_vgg16_first_lastlayer.png", dpi=300)
"""