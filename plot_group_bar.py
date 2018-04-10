import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set()
sns.set_style("dark")
sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.2)
plt.figure(figsize=(2,6))
sns.set_style("darkgrid")

n_groups = 2

#Overall
"""
skip=[0.819335,0.818205,0.741207,0.675051]
skip.append(sum(skip)/len(skip))
Glove=[0.768675,0.835844,0.723647,0.732321]
Glove.append(sum(Glove)/len(Glove))
RNN=[0.768675,0.773007,0.707386,0.588186]
RNN.append(sum(RNN)/len(RNN))
Global=[0.759134,0.764281,0.617842,0.482993]
Global.append(sum(Global)/len(Global))
Cross=[0.768675,0.838481,0.723647,0.588186]
Cross.append(sum(Cross)/len(Cross))
Dist=[0.753107,0.755493,0.723647,0.654643]
Dist.append(sum(Dist)/len(Dist))
"""

# Concrete Vs Abstract
"""
skip=[0.680162,0.751187]
skip.append(sum(skip)/len(skip))
Glove=[0.669366,0.752453]
Glove.append(sum(Glove)/len(Glove))
RNN=[0.602436,0.722963]
RNN.append(sum(RNN)/len(RNN))
Global=[0.591693,0.646154]
Global.append(sum(Global)/len(Global))
Cross=[0.660519,0.747388]
Cross.append(sum(Cross)/len(Cross))
Dist=[0.614635,0.690725]
Dist.append(sum(Dist)/len(Dist))
"""
"""
temp=[0.819335,0.818205,0.741207,0.675051]
skip=[]
skip.append(sum(temp)/len(temp))
skip.extend([0.6392,0.6962,0.6094,0.6462,0.5679])



temp=[0.768675,0.773007,0.707386,0.588186]
RNN=[]
RNN.append(sum(temp)/len(temp))
RNN.extend([0.3675,0.4928,0.2925,0.4344,0.3990])
print (RNN)

temp=[0.759134,0.764281,0.617842,0.482993]
Global=[]
Global.append(sum(temp)/len(temp))
Global.extend([0.6231,0.7005,0.5566,0.3140,0.3153])

temp=[0.768675,0.838481,0.723647,0.588186]
Cross=[]
Cross.append(sum(temp)/len(temp))
Cross.extend([0.6803,0.7434,0.6464,0.7587,0.6467])
"""


skip=[0.651]
skip.append(sum([0.819335,0.818205])/len([0.819335,0.818205]))

Glove=[0.655]
Glove.append(sum([0.768675,0.835844])/len([0.768675,0.835844]))

RNN=[0.629]
RNN.append(sum([0.768675,0.773007])/len([0.768675,0.773007]))

Global=[0.635]
Global.append(sum([0.759134,0.764281])/len([0.759134,0.764281]))

Cross=[0.665]
Cross.append(sum([0.768675,0.838481])/len([0.768675,0.838481]))
Dist=[0.61]
Dist.append(sum([0.753107,0.755493])/len([0.753107,0.755493]))

#Comparion of Brain Bench Vs Rest
fig, ax = plt.subplots(figsize=(6,6))
index = np.arange(n_groups)
print (index)
bar_width = 0.05
opacity = 1.0
#('Global Context',	'Skip-Gram',	'RNN',	'Cross-Lingual',	'Glove',	'Non-Distributional','Skip-Gram-Italian')

rects1 = plt.bar([0.1, 0.5], skip, bar_width,
                 alpha=opacity,align='center',
                 color='#581845',
                 label='Skip-Gram')

rects2 = plt.bar([0.15, 0.55], Glove, bar_width,
                 alpha=opacity,align='center',
                 color='#FF5733',
                 label='Glove')


rects3 = plt.bar([0.20, 0.60], RNN, bar_width,
                 alpha=opacity,align='center',
                 color='#000080',
                 label='RNN')

rects4 = plt.bar([0.25, 0.65], Global, bar_width,
                 alpha=opacity,align='center',
                 color='#800080',
                 label='Global Context')

rects5 = plt.bar([0.30, 0.70], Cross, bar_width,
                 alpha=opacity,align='center',
                 color='#808000',
                 label='Cross-Lingual')

rects6 = plt.bar([0.35, 0.75], Dist, bar_width,
                 alpha=opacity,align='center',
                 color='#FFC300',
                 label='Non-Dist')

"""
rects1 = plt.bar(index, skip, bar_width,
                 alpha=opacity,
                 color='#581845',
                 label='Skip-Gram')

rects2 = plt.bar(index + bar_width , RNN, bar_width,
                 alpha=opacity,
                 color='#000080',
                 label='RNN')

rects3 = plt.bar(index + bar_width+bar_width, Global, bar_width,
                 alpha=opacity,
                 color='#800080',
                 label='Global Context')

rects4 = plt.bar(index + bar_width + bar_width+bar_width, Cross, bar_width,
                 alpha=opacity,
                 color='#808000',
                 label='Cross-Lingual')


"""






#algo = ('BrainBench','WS-353','WS-353-SIM', 'WS-353-REL','MEN','MTurk-771')

#plt.xticks(index + 0.15, ( 'BrainBench','WS-353','WS-353-SIM', 'WS-353-REL','MEN','MTurk-771'))
plt.xticks([0.21, 0.61], ( 'BrainBench V1.0','BrainBench V2.0'))
plt.ylim(0.50,0.99)
#sns.plt.title('2 vs. 2 Accuracy for Concrete/Abstract Words in Italian fMRI').set_fontsize('12')
sns.plt.title('Comparison of BrainBench Versions').set_fontsize('12')
sns.plt.ylabel('Correlation').set_fontsize('12')




plt.gcf().subplots_adjust(bottom=0.10)
plt.legend()
plt.tight_layout()

# plt.xticks(rotation=15)
plt.savefig("/Users/Dhanush/Desktop/Versions.png", dpi=300)




