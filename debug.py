from sklearn.externals import joblib

vocab=joblib.load("ImagenetMapping_withFiles.pkl")

print (vocab['snowplow'][0])
print (vocab['snowplough'][0])