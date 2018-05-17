#!/usr/bin/env python3

""" Generate 1 vs. 2 results for adversarial images """

import os
import csv
import joblib
import argparse
import numpy as np
from os.path import join
from keras.preprocessing import image
from scipy.stats.stats import pearsonr
from Activations import get_activations
from inception_v3 import InceptionV3, decode_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--layer', default='mixed4')
parser.add_argument('--target-class', default=-1, type=int)
parser.add_argument('--cache-dir', default='./activation_cache')
parser.add_argument('adversarial_images', metavar='adversarial-images')
parser.add_argument('valid_images', metavar='valid-images')
parser.add_argument('map_file', metavar='map-file')

class SkipgramVectors:

    source = './Skip_gram.txt'

    def __init__(self):
        self.words = {}
        with open(self.source, 'r') as src:
            reader = csv.reader(src, delimiter=' ')
            for line in reader: 
                vector = [ float(v) for v in line[1:] ]
                self.words[line[0]] = vector
        
    def get(self, word):
        return self.words[word]

def preprocess_input(x):
    """ Model weights were trained expecting this preprocessing """
    return (x / 127.5) - 1.

if __name__ == '__main__':
    
    args = parser.parse_args()

    correct = 0
    total = 0

    model = InceptionV3(include_top=True, weights='imagenet')
    #print([layer.name for layer in model.layers])
    layer_name = args.layer

    valid_concepts = [ fname for fname in os.listdir(args.valid_images) ]

    try:
        cnn_valid = joblib.load(join(args.cache_dir, 'activation_cache' + layer_name + '.pkl'))
        print('Using cached file...')
    except FileNotFoundError:
        cnn_valid = []
        for valid_concept in valid_concepts: 
            dir_path = join(args.valid_images, valid_concept)
            image_filename = [ fname for fname in os.listdir(dir_path) ][0]
            image_path = join(dir_path, image_filename)
            img = image.load_img(image_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y = model.predict(x)
            preds = decode_predictions(y)
            if preds[0][0][1] != valid_concept:
                print('Valid image for', image_path, 'did not give', valid_concept)
                print(preds)
            else:
                print('OK:', image_path)
            activations = get_activations(model, x, layer_name=layer_name)
            cnn_valid.append(activations)
        cnn_valid = np.array(cnn_valid)
        os.makedirs(args.cache_dir, exist_ok=True)
        joblib.dump(cnn_valid, join(args.cache_dir, 'activation_cache' + layer_name + '.pkl'))

    print(cnn_valid.shape)

    wv_valid = []
    vectors = SkipgramVectors()
    for valid_concept in valid_concepts:
        wv_valid.append(vectors.get(valid_concept))
    print('loaded word vectors')
        
    assert len(cnn_valid) == 100

    map_data = joblib.load(args.map_file)
    classes = list(range(len(map_data[0])))
    assert len(map_data[0]) == len(map_data[1])

    for i in classes:
        source_class = map_data[0][i]
        target_class = map_data[1][i][args.target_class][0]
        dir_path = join(args.adversarial_images, source_class, target_class)
        image_filename = [ fname for fname in os.listdir(dir_path) ][0]
        image_path = join(dir_path, image_filename)
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = decode_predictions(model.predict(x))
        # TODO: Should be an assert
        if preds[0][0][1] != target_class:
            print('Adversarial image for', image_path, 'did not give', target_class)
            print('Skipping...')
            continue
        print('OK:', image_path)
        act = get_activations(model, x, layer_name=layer_name)
        act = np.array(act)
        i_adversarial = []
        assert len(cnn_valid[0].flatten()) == len(act.flatten())


        for activations in cnn_valid:
            i_adversarial.append(pearsonr(activations.flatten(), act.flatten())[0])

        v_i = vectors.get(source_class)
        w_valid =  []
        for vector in wv_valid:
            w_valid.append(pearsonr(vector, v_i)[0])

        a_i = vectors.get(target_class)
        w_adversarial =  []
        for vector in wv_valid:
            w_adversarial.append(pearsonr(vector, a_i)[0])

        a = pearsonr(w_adversarial, i_adversarial)[0]
        b = pearsonr(w_valid, i_adversarial)[0]

        if a < b: correct += 1
        total += 1

    print(correct)
    print(total)
    print('RESULT:', correct / total)
