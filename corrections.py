#This program will correct the vocab of skipgram removing duplicate classes
from sklearn.externals import joblib



vocab = joblib.load("./vocabSkipGram.pkl")
mapping =joblib.load('ImagenetMapping.pkl')
paths=joblib.load("./lpath3.pkl")


reverse_mapping={}
for item in mapping:
	word=item
	no=mapping[item][0]
	if word in vocab:
		if no not in reverse_mapping:
			reverse_mapping[no]=word

new_vocab =[]
new_lpath=[]

for cls in reverse_mapping:
	word =reverse_mapping[cls]
	new_vocab.append(word)

print (len(new_vocab))
new_vocab.sort()

for i in range(len(vocab)):
	if vocab[i] in new_vocab:
		new_lpath.append(paths[i])

print (len(new_vocab))
print (len(new_lpath))
#joblib.dump(new_vocab,"./vocabSkipGram1.pkl")
joblib.dump(new_lpath,"./lpath3.pkl")
