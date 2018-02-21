from sklearn.externals import joblib

#This program merges the inception V3 Data.

data1=joblib.load('./scores/test_scores_inceptionv3_activation20.pkl')
data2=joblib.load('./scores/test_scores_inceptionv3_activation40.pkl')
data3=joblib.load('./scores/test_scores_inceptionv3_activation60.pkl')
data4=joblib.load('./scores/test_scores_inceptionv3_activation80.pkl')
data5=joblib.load('./scores/test_scores_inceptionv3_activation94.pkl')
mixed=joblib.load('./scores/test_scores_inceptionv3_mixed.pkl')

merged=[]
merged.extend(data1)
merged.extend(data2)
merged.extend(data3)
merged.extend(data4)
merged.extend(data5)


joblib.dump(merged,'./scores/test_scores_inceptionv3_activations.pkl')

mixed_out =merged[0:25]
mixed_out.extend(mixed)


print (mixed_out)

joblib.dump(mixed_out,'./scores/test_scores_inceptionv3_mixed.pkl')