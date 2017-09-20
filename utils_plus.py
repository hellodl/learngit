from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import save_model, load_model
import numpy as np
from math import ceil
from dc_utils import onehot,get_batch,save_array,load_array

def create_vgg16():
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))

    def vgg_preprocess(x):
        x = x - vgg_mean
        return x[:, ::-1]

    def ConvBlock(model, nb_block, nb_layer, nb_filter, activation):
        for i in range(nb_block):
            for j in range(nb_layer[i]):
                model.add(ZeroPadding2D((1, 1)))
                model.add(Conv2D(nb_filter[i], (3, 3), activation=activation))
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(model, classes, nb_neuron, dropout, activation):
        for i in range(len(nb_neuron)):
            model.add(Dense(nb_neuron[i], activation=activation[i], name='dense_%s' % i))
            model.add(Dropout(dropout[i], name='dropout_%s' % i))

        model.add(Dense(classes, activation='softmax', name='dense_out'))

    model = Sequential()
    model.add(
        Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    ConvBlock(model,
              nb_block=5,
              nb_layer=[2, 2, 3, 3, 3],
              nb_filter=[64, 128, 256, 512, 512],
              activation='relu')

    model.add(Flatten())
    FCBlock(model,
            classes=1000,
            nb_neuron=[4096, 4096],
            dropout=[0.5, 0.5],
            activation=['relu', 'relu', 'softmax'])

    model.load_weights('vgg16.h5')
    return model


def create_vgg16_bn():
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))

    def vgg_preprocess(x):
        x = x - vgg_mean
        return x[:, ::-1]  # reverse axis rgb->bgr

    def ConvBlock(model, nb_block, nb_layer, nb_filter, activation):
        for i in range(nb_block):
            for j in range(nb_layer[i]):
                model.add(ZeroPadding2D((1, 1)))
                model.add(Conv2D(nb_filter[i], (3, 3), activation=activation))
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(model, classes, nb_neuron, dropout, activation):
        for i in range(len(nb_neuron)):
            model.add(Dense(nb_neuron[i], activation=activation[i], name='dense_%s' % i))
            model.add(BatchNormalization())  # before dropout
            model.add(Dropout(dropout[i], name='dropout_%s' % i))

        model.add(Dense(classes, activation='softmax', name='dense_out'))

    model = Sequential()
    model.add(
        Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    ConvBlock(model,
              nb_block=5,
              nb_layer=[2, 2, 3, 3, 3],
              nb_filter=[64, 128, 256, 512, 512],
              activation='relu')

    model.add(Flatten())
    FCBlock(model,
            classes=1000,
            nb_neuron=[4096, 4096],
            dropout=[0.5, 0.5],
            activation=['relu', 'relu', 'softmax'])

    model.load_weights('vgg16_bn.h5')
    return model


def gen_feat_lbs(model, datapath, batch_size, img_size=(224,224)):
    batches = get_batch(datapath,
                        target_size=img_size,
                        class_mode='categorical',
                        shuffle=False,
                        batch_size=batch_size)
    labels = onehot(batches.classes)
    features = model.predict_generator(batches,
                                        ceil(batches.samples / batch_size),
                                        verbose=1)

    return features, labels


def gen_feat_ids(model, datapath, batch_size, img_size=(224,224)):
    batches = get_batch(
        datapath,
        target_size=img_size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size)

    ids = np.array([int(f[8:f.find('.')]) for f in batches.filenames])
    features = model.predict_generator(batches,
                                       ceil(batches.samples / batch_size),
                                       verbose=1)
    return features, ids


def spilt_model(model):
    layers = model.layers
    last_conv_idx = [index for index, layer in enumerate(layers)
                     if type(layer) is Conv2D][-1]

    conv_layers = layers[:last_conv_idx + 1]
    conv_model = Sequential(conv_layers)
    fc_layers = layers[last_conv_idx + 1:]

    return conv_model, fc_layers, last_conv_idx


if __name__ == '__main__':
    vgg16_mdl = create_vgg16()
    conv_model,_,_ = spilt_model(vgg16_mdl)
    trn_feat, trn_lbs = gen_feat_lbs(conv_model,
                                     './data/train/',
                                     batch_size=64)
    val_feat, val_lbs = gen_feat_lbs(conv_model,
                                     './data/valid/',
                                     batch_size=64)

    save_array('./data/feature/trn_cnv_feat.bc', trn_feat)
    save_array('./data/feature/val_cnv_feat.bc', val_feat)
