import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set()
sns.set_style("dark")
sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.2)
plt.figure(figsize=(3,3))
sns.set_style("darkgrid")

n_groups = 5

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

temp=[0.819335,0.818205,0.741207,0.675051]
skip=[]
skip.append(sum(temp)/len(temp))
skip.extend([0.7607969546793074, 0.4423644329526677,0.7673906485671189,0.5593966817496223])

temp=[0.768675,0.835844,0.723647,0.732321]
Glove=[]
Glove.append(sum(temp)/len(temp))
Glove.extend([0.7448,0.3847,0.6556,0.5616])

temp=[0.768675,0.773007,0.707386,0.588186]
RNN=[]
RNN.append(sum(temp)/len(temp))
RNN.extend([0.4382,0.298,0.543,0.327])
print (RNN)

temp=[0.759134,0.764281,0.617842,0.482993]
Global=[]
Global.append(sum(temp)/len(temp))
Global.extend([0.358,0.0925,0.685,0.520])

temp=[0.768675,0.838481,0.723647,0.588186]
Cross=[]
Cross.append(sum(temp)/len(temp))
Cross.extend([0.710,0.3668,0.675,0.5800])

temp=[0.753107,0.755493,0.723647,0.654643]
Dist=[]
Dist.append(sum(temp)/len(temp))
Dist.extend([0.359,0.486,0.554,0.2063])


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
"""
#Comparion of Brain Bench Vs Rest
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.10
opacity = 1.0
#('Global Context',	'Skip-Gram',	'RNN',	'Cross-Lingual',	'Glove',	'Non-Distributional','Skip-Gram-Italian')

rects1 = plt.bar(index, skip, bar_width,
                 alpha=opacity,
                 color='#581845',
                 label='Skip-Gram')

rects2 = plt.bar(index + bar_width, Glove, bar_width,
                 alpha=opacity,
                 color='#FF5733',
                 label='Glove')


rects3 = plt.bar(index + bar_width + bar_width, RNN, bar_width,
                 alpha=opacity,
                 color='#000080',
                 label='RNN')

rects4 = plt.bar(index + bar_width + bar_width+bar_width, Global, bar_width,
                 alpha=opacity,
                 color='#800080',
                 label='Global Context')

rects5 = plt.bar(index + bar_width + bar_width+bar_width+bar_width, Cross, bar_width,
                 alpha=opacity,
                 color='#808000',
                 label='Cross-Lingual')

rects6 = plt.bar(index + bar_width + bar_width+bar_width+bar_width+bar_width, Dist, bar_width,
                 alpha=opacity,
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






algo = ('BrainBench','MEN','SimLex','WS-353-SIM' ,'WS-353-REL')
#algo = ( 'BrainBench V1.0','BrainBench V2.0')
algo_len = np.arange(len(algo))
plt.xticks(index + 0.25, ( 'BrainBench','MEN','SimLex','WS-353-SIM', 'WS-353-REL'))
#plt.xticks(index+bar_width, ( 'BrainBench V1.0','BrainBench V2.0'))
plt.ylim(0.00,1.0)
#sns.plt.title('2 vs. 2 Accuracy for Concrete/Abstract Words in Italian fMRI').set_fontsize('12')
sns.plt.title('Comparison of BrainBench With Word Similarity Datasets').set_fontsize('12')
sns.plt.ylabel('Correlation').set_fontsize('12')




plt.gcf().subplots_adjust(bottom=0.20)
plt.legend()
plt.tight_layout()
# plt.xticks(rotation=15)
#plt.show()
plt.savefig("/Users/Dhanush/Desktop/Comparison.png", dpi=300)

