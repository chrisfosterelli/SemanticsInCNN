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
parser.add_argument('--skipgram', default='./Skip_gram.txt')
parser.add_argument('--cache-dir', default='./activation_cache')
parser.add_argument('adversarial_images', metavar='adversarial-images')
parser.add_argument('valid_images', metavar='valid-images')
parser.add_argument('map_file', metavar='map-file')

class SkipgramParser:

    """ Skipgram weight parser """

    def __init__(self, source):
        self.words = {}
        with open(source, 'r') as src:
            reader = csv.reader(src, delimiter=' ')
            for line in reader: 
                vector = [ float(v) for v in line[1:] ]
                self.words[line[0]] = vector
        
    def get(self, word):
        return self.words[word]

def preprocess_input(x):
    """ Model weights were trained expecting this preprocessing """
    return (x / 127.5) - 1.

def run_model(model, path, layer):
    """ Return the activations for a layer and the net's prediction """
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    preds = decode_predictions(y)
    activations = get_activations(model, x, layer_name=layer)
    return preds, np.array(activations).flatten()

if __name__ == '__main__':
    
    args = parser.parse_args()

    correct, total = 0, 0
    model = InceptionV3(include_top=True, weights='imagenet')
    layer_name = args.layer

    valid_concepts = [ fname for fname in os.listdir(args.valid_images) ]

    cache_filename = 'activation_cache' + args.layer + '.pkl'
    cache_path = join(args.cache_dir, cache_filename)

    try:

        cnn_valid = joblib.load(cache_path)
        print('Using cached activations')

    except FileNotFoundError:

        cnn_valid = []

        for valid_concept in valid_concepts: 
            concept_dir = join(args.valid_images, valid_concept)
            image_filename = [ fname for fname in os.listdir(concept_dir) ][0]
            image_path = join(concept_dir, image_filename)
            _, activations = run_model(model, image_path, args.layer)
            cnn_valid.append(activations)
            print('OK:', image_path)

        cnn_valid = np.array(cnn_valid)
        os.makedirs(args.cache_dir, exist_ok=True)
        joblib.dump(cnn_valid, cache_path)

    wv_valid = []
    vectors = SkipgramParser(args.skipgram)
    print('Loaded Word Vectors')

    for valid_concept in valid_concepts:
        wv_valid.append(vectors.get(valid_concept))

    map_data = joblib.load(args.map_file)
    classes = list(range(len(map_data[0])))
    assert len(map_data[0]) == len(map_data[1])

    for class_i in classes:

        source_class = map_data[0][class_i]
        target_class = map_data[1][class_i][args.target_class][0]
        concept_dir = join(args.adversarial_images, source_class, target_class)
        image_filename = [ fname for fname in os.listdir(concept_dir) ][0]
        image_path = join(concept_dir, image_filename)
        preds, adv_activations = run_model(model, image_path, args.layer)
        assert len(cnn_valid[0]) == len(adv_activations)

        if preds[0][0][1] != target_class:
            print('WARN: Attack did not produce', target_class)
            print('WARN: Skipping image at', image_path)
            continue

        print('OK:', image_path)

        i_adversarial = []
        v_i = vectors.get(source_class)
        a_i = vectors.get(target_class)
        w_valid, w_adversarial = [], []

        for valid_activations in cnn_valid:
            corr = pearsonr(valid_activations, adv_activations)
            i_adversarial.append(corr[0])

        for vector in wv_valid:
            corr = pearsonr(vector, v_i)
            w_valid.append(corr[0])

        for vector in wv_valid:
            corr = pearsonr(vector, a_i)
            w_adversarial.append(corr[0])

        a = pearsonr(w_adversarial, i_adversarial)[0]
        b = pearsonr(w_valid, i_adversarial)[0]

        if a < b: correct += 1
        total += 1

    print('TOTAL:', total)
    print('CORRECT:', correct)
    print('RESULT:', correct / total)
