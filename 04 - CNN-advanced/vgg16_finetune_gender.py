from __future__ import print_function

from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils.np_utils import to_categorical
import numpy as np

K.set_image_dim_ordering('tf')


def load(test_split=0.2):
    # TODO: implementirati ucitavanje WIKI dataseta
    # parametar test_split: udeo podataka koji treba uzeti za testni skup
    # rezultat treba da bude (x_train, y_train), (x_test, y_test)
    # x su slike, y su labele pola (0=female, 1=male)
    raise NotImplementedError('TODO: implementirati ucitavanje WIKI dataseta')


def load_data():
    (x_train, y_train), (x_test, y_test) = load()

    # preprocesiranje podataka
    # vrednosti piksela slika se centriraju predefinisanim vrednostima za ImageNet dataset
    # pogledati implementaciju preprocess_input
    x_train = preprocess_input(x_train.astype(np.float32))
    x_test = preprocess_input(x_test.astype(np.float32))
    # izlazi se transformisu iz celobrojnih vrednosti (0,1) u kategoricke vrednosti ([1, 0], [0, 1])
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return (x_train, y_train), (x_test, y_test)


def get_model():
    # bazni model je VGG16 obucen na ImageNet datasetu
    base_model = VGG16(include_top=True, weights='imagenet')
    # koristimo poslednji FC sloj i na njega dodamo ono sto nama treba
    final = base_model.get_layer(name='fc2').output
    final = Dropout(0.5)(final)
    final = Dense(2, activation='softmax')(final)  # dva izlaza, female/male
    model = Model(input=base_model.input, output=final)

    return base_model, model


def train():
    (x_train, y_train), (x_test, y_test) = load_data()
    base_model, model = get_model()

    # augmentacija podataka
    gen = ImageDataGenerator(rotation_range=10.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True)

    # "zamrzavanje" svih slojeva osim poslednjeg
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # callback koji snima tezine modela
    mc = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    # callback koji prekida obucavanje u slucaju overfitting-a
    es = EarlyStopping(monitor='val_loss', patience=10)

    # malo istreniramo samo poslednji sloj kao neku vrstu inicijalizacije novog modela
    model.fit_generator(gen.flow(x_train, y_train, batch_size=32, shuffle=True),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=100, verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[mc, es])

    # ucitavanje tezina sa najboljim rezultatom
    model.load_weights('weights.h5')

    # "odmrzavanje" svih slojeva, pusticemo sada i njih da se prilagode
    for layer in base_model.layers:
        layer.trainable = True

    # obratite paznju da je learning rate veina mali
    model.compile(optimizer=SGD(1e-4, momentum=0.9, decay=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    mc = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=20)

    # sada radimo fine-tuning celog modela
    model.fit_generator(gen.flow(x_train, y_train, batch_size=32, shuffle=True),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=200, verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[mc, es])


if __name__ == '__main__':
    train()
