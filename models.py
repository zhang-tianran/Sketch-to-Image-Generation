# Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/context_encoder/context_encoder.py
import argparse
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import KLDivergence as kl
from keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate of optimizer")
parser.add_argument("--b", type=float, default=0.5, help="beta of optimizer")
parser.add_argument("--lam1", type=float, default=0.1, help="coefficient of perceptual loss")
parser.add_argument("--lam2", type=float, default=0.1, help="coefficient of contextual loss")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--gf_dim", type=int, default=64, help="dimension of gen filters in first conv layer.")
parser.add_argument("--df_dim", type=int, default=64, help="dimension of discrim filters in first conv layer.")
opt = parser.parse_args()
print(opt)


class ContextualGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.mask_height = 8
        self.mask_width = 8
        self.channels = 3
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(opt.lr, opt.b)

        def contextual_loss(y_true, y_pred):
            y_pred = tf.image.rgb_to_grayscale(tf.slice(y_pred, [0,0,0,0], [opt.batch_size, self.img_rows, self.img_cols, self.channels]))
            y_true = tf.image.rgb_to_grayscale(tf.slice(y_true, [0,0,0,0], [opt.batch_size, self.img_rows, self.img_cols, self.channels]))
            
            y_pred = tf.divide(tf.add(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), 1), 2)
            y_true = tf.divide(tf.add(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), 1), 2)
                
            # normalize sum to 1
            y_pred = tf.divide(y_true, tf.tile(tf.expand_dims(tf.reduce_sum(y_true, axis=1), 1), [1,tf.shape(y_true)[1]]))
            y_true = tf.divide(y_pred, tf.tile(tf.expand_dims(tf.reduce_sum(y_pred, axis=1), 1), [1,tf.shape(y_pred)[1]]))
            
            return kl(y_pred, y_true)

        def total_loss(y_true, y_pred):
            p_loss = K.binary_crossentropy(y_true, y_pred)
            c_loss = contextual_loss(y_true, y_pred)
            return opt.lam1 * p_loss + opt.lam2 * c_loss

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=[total_loss, 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # TODO
        model.add(Input((8192*2)))
        model.add(Dense(4*8*512))
        model.add(Reshape((4, 8, 512)))
        model.add(Conv2DTranspose(64*8, 1, 1, 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64*4, 5, 2, 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64*2, 5, 2, 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 5, 2, 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(3, 5, 2, 'same'))
        model.add(LeakyReLU())
        model.add(Activation('tanh'))

        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        # TODO
        model.add(Input((64, 128, 3)))
        model.add(Conv2D(64, 5, 2, 'same'))
        model.add(Conv2D(64*2, 5, 2, 'same'))
        model.add(Conv2D(64*4, 5, 2, 'same'))
        model.add(Conv2D(64*8, 5, 2, 'same'))
        model.add(Dense(1, 'softmax'))

        model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)