import os
import glob
import numpy as np
import argparse
import keras
from keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint
)
from keras.datasets import cifar100
from sklearn.utils import shuffle
from keras.layers import (
    Activation,
    Input,
    Dense,
    Flatten
)
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
#from keras.utils.visualize_util import plot
from keras.utils import np_utils

from fractalnet import fractal_net

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

NB_CLASSES = 100
NB_EPOCHS = 800
LEARN_START = 0.02
BATCH_SIZE = 100
MOMENTUM = 0.9

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
#permute the labels randomly
np.random.shuffle(y_train)
np.random.shuffle(y_test)

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train,Y_train=shuffle(X_train, Y_train, random_state=0)

X_train /= 255
X_test /= 255





print X_train.shape

# Drop by 10 when we halve the number of remaining epochs (200, 300, 350, 375)
def learning_rate(epoch):
    if epoch < 400:
        return 0.002
    if epoch < 500:
        return 0.0002
    if epoch < 700:
        return 0.00002
    if epoch < 800:
        return 0.000002
    return 0.0000002

def build_network(deepest=False):
    dropout = [0., 0.1, 0.2, 0.3, 0.4]
    conv = [(64, 3, 3), (128, 3, 3), (256, 3, 3), (512, 3, 3), (512, 2, 2)]
    input= Input(shape=(32, 32,3))
    output = fractal_net(
        c=3, b=5, conv=conv,
        drop_path=0.15, dropout=dropout,
        #drop_path=0.15, dropout=None,
        deepest=deepest)(input)
    output = Flatten()(output)
    #output = Dense(NB_CLASSES, init='he_normal')(output)
    output=Dense(NB_CLASSES, kernel_initializer='he_normal')(output)
    output = Activation('softmax')(output)
    #model = Model(input=input, output=output)
    model = Model(inputs=input, outputs=output)
    optimizer = SGD(lr=LEARN_START, momentum=MOMENTUM)
    #optimizer = RMSprop(lr=LEARN_START)
    #optimizer = Adam()
    #optimizer = Nadam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    #plot(model, to_file='model.png')
    return model

def train_network(net):
    print("Training network")
    history = LossHistory()
    snapshot = ModelCheckpoint(
        filepath="/home/dhanushd/scratch/Rsnapshots/weights.{epoch:04d}.h5",
        monitor="val_loss",save_weights_only=False,
        save_best_only=False)
    learn = LearningRateScheduler(learning_rate)
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    csv_logger=keras.callbacks.CSVLogger('Rtraining.log', separator=',', append=False)
    net.fit(
        x=X_train, y=Y_train, batch_size=BATCH_SIZE,
        epochs=NB_EPOCHS, validation_data=(X_test, Y_test),
        callbacks=[history,learn, snapshot,csv_logger]
        #callbacks=[snapshot]
    )

def test_network(net, weights):
    print("Loading weights from '{}' and testing".format(weights))
    net.load_weights(weights)
    ret = net.evaluate(x=X_test, y=Y_test, batch_size=BATCH_SIZE)
    print('Test:', ret)

def main():
    parser = argparse.ArgumentParser(description='FractalNet on CIFAR-100')
    parser.add_argument('--load', nargs=1,
                        help='Test network with weights file')
    parser.add_argument('--deepest', help='Build with only deepest column activated',
                        action='store_true')
    parser.add_argument('--test-all', nargs=1,
                        help='Test all the weights from a folder')
    parser.add_argument('--summary',
                        help='Print a summary of the network and exit',
                        action='store_true')
    args = parser.parse_args()
    net = build_network(deepest=args.deepest)
    if args.load:
        weights = args.load[0]
        test_network(net, weights)
    elif args.test_all:
        folder = args.test_all[0]
        for weights in glob.glob(os.path.join(folder, 'weigh*')):
            test_network(net, weights)
    elif args.summary:
        net.summary()
    else:
        train_network(net)

main()
