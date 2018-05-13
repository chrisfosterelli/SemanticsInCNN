
#
# Adversarial attack
#

import scipy
import numpy as np
from keras import backend as K
from inception_v3 import InceptionV3
from keras.preprocessing import image
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import MomentumIterativeMethod
from keras.applications.imagenet_utils import decode_predictions

def preprocess_input(x):
    """ Model weights were trained expecting this preprocessing """
    return (x / 127.5) - 1.

def postprocess_input(x):
    """ Undo the preprocessing in preprocess_input to get an image back """
    return (x + 1.) * 127.5

if __name__ == '__main__':

    model = InceptionV3(include_top=True, weights='imagenet')

    img_path = '/Users/chrisfosterelli/Desktop/acc.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    print('Predicted:', decode_predictions(preds))

    target = 420

    sess = K.get_session()
    wrap = KerasModelWrapper(model)
    attack = MomentumIterativeMethod(wrap, sess=sess)
    y_target = np.zeros([ 1, 1000 ])
    y_target[0, target] = 1

    params = {
        'eps': 8.0/255.0,
        'clip_min': -1.,
        'clip_max': 1.,
        'y_target': y_target
    }

    adv_x = attack.generate_np(x, **params)
    preds = model.predict(adv_x)

    print('Predicted:', decode_predictions(preds))

    img = postprocess_input(adv_x[0])
    scipy.misc.imsave('/Users/chrisfosterelli/Desktop/acc_adv.jpg', img)
