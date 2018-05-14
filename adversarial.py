#!/usr/bin/env python3

""" Run an adversarial attack """

import os
import json
import scipy
import joblib
import argparse
import numpy as np
from os.path import join
from keras import backend as K
from inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import MomentumIterativeMethod
from keras.applications.imagenet_utils import decode_predictions

CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/' + \
                   'image-models/imagenet_class_index.json'

ATTACK_PARAMS = {
    'eps': 8.0 / 255.0,
    'clip_min': -1.,
    'clip_max': 1.
}

def get_imagenet_index():
    """ Load imagenet class map """
    int_to_str, str_to_int = {}, {}
    fpath = get_file('imagenet_class_index.json',
                     CLASS_INDEX_PATH,
                     cache_subdir='models',
                     file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
        data = json.load(f)
    for k, v in data.items():
        int_to_str[int(k)] = v[1]
        str_to_int[v[1]] = int(k)
    return int_to_str, str_to_int

def preprocess_input(x):
    """ Model weights were trained expecting this preprocessing """
    return (x / 127.5) - 1.

def postprocess_input(x):
    """ Undo the preprocessing in preprocess_input to get an image back """
    return (x + 1.) * 127.5

def adversarial_attack(x, target):
    """ Take an input x and generate an adversarial x """

def save_adv(attack, source, dest, target):
    """ Load an image and save an adversarial image """
    img = image.load_img(source, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y_target = np.zeros([ 1, 1000 ])
    y_target[0, target] = 1
    adv_x = attack.generate_np(x, y_target=y_target, **ATTACK_PARAMS)
    img = postprocess_input(adv_x[0])
    scipy.misc.imsave(dest, img)

parser = argparse.ArgumentParser()
parser.add_argument('--output-images', default='./output')
parser.add_argument('--classes', nargs='+', default=[], type=int)
parser.add_argument('source_images', metavar='source-images')
parser.add_argument('map_file', metavar='map-file')

if __name__ == '__main__':

    args = parser.parse_args()
    model = InceptionV3(include_top=True, weights='imagenet')
    _, imagenet_map = get_imagenet_index()
    map_data = joblib.load(args.map_file)

    sess = K.get_session()
    wrap = KerasModelWrapper(model)
    attack = MomentumIterativeMethod(wrap, sess=sess)

    classes = list(range(len(map_data[0])))
    if args.classes == []: args.classes = classes
    assert len(map_data[0]) == len(map_data[1])

    for i in classes:
        source_class = map_data[0][i]
        target_classes = map_data[1][i]
        if i not in args.classes: continue
        for target_class, _ in target_classes:
            target_class_i = imagenet_map[target_class]
            src_dir_path = join(args.source_images, source_class)
            dst_dir_path = join(args.output_images, source_class, target_class)
            image_filenames = [ fpath for fpath in os.listdir(src_dir_path) ]
            for filename in image_filenames:
                src_path = join(src_dir_path, filename)
                dst_path = join(dst_dir_path, filename)
                os.makedirs(dst_dir_path, exist_ok=True)
                save_adv(attack, src_path, dst_path, target_class_i)
                print(src_path, '->', dst_path)
