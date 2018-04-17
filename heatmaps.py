import numpy as np; np.random.seed(12322)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from matplotlib import colors


temp = np.random.rand(10, 10)
temp=temp[0,:]
data=np.empty([1, 10])
np.copyto(data,temp)
#print (data.shape)

labels=['C'+str(i) for i in range(1,11)]

fig, ax = plt.subplots()
sns.heatmap(data,cbar=False,cmap='magma',xticklabels=labels,yticklabels=['C1'])
ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
plt.imshow(data, cmap='afmhot', interpolation='nearest')
ax.xaxis.tick_top()
plt.savefig("wwC1.png",dpi=300)
