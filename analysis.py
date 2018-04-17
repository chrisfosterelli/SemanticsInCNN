# This program aims to do a Quantitative analysis of mistakes made by VGG16, Inception V3 and ResNet 50
 
from sklearn.externals import joblib

# Find out how many in top 5 it got correct


data=joblib.load("/Users/Dhanush/Desktop/Cnn2Word2Vec/vgg16_predictions.pkl")

print (data)


# Find out cosine similarity of predicted and actual class.





#Print 6 images each for VGG16, Inception V3 n Resnet