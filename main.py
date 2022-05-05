# Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/context_encoder/context_encoder.py

import argparse
import numpy as np
# import cv2

from matplotlib import pyplot as plt

from models import *
from evaluation import *
from preprocess import *
# from visualizer import * 

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=3, help="interval between image sampling")
parser.add_argument("--z_dim", type=int, default=100, help="dimension of z sampler.")
opt = parser.parse_args()

def train(model, X_train):

    d_loss_list = []
    g_loss_list = []

    # Adversarial ground truths
    valid = np.ones((opt.batch_size, 1))
    fake = np.zeros((opt.batch_size, 1))

    X_train = X_train.astype('float32') / 127.5 - 1.

    for epoch in range(opt.epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], opt.batch_size)
        imgs = X_train[idx]

        # img = imgs[1].astype('float32')[:,:,::-1]
        # plt.imshow(img)
        # plt.show()

        # masked_imgs, missing_parts, _ = model.mask_randomly(imgs)
        sketch = mask_image(imgs)

        z_sample = np.random.uniform(-1, 1, size=(opt.batch_size , opt.z_dim))

        # Generate a batch of new images
        gen = model.generator.predict(sketch)

        # Train the discriminator
        d_loss_real = model.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = model.discriminator.train_on_batch(gen, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator 
        # ---------------------

        g_loss = model.combined.train_on_batch(sketch, [valid, imgs])

        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss[0])

        # Plot the progress
        print ("%d [D loss: %f] [G loss: %f, perceptual loss: %f, contextual loss: %f]" % (epoch, d_loss, g_loss[0], g_loss[1], g_loss[2]))

        if epoch % opt.sample_interval == 0:
            img = gen[1].astype('float32')[:,:,::-1]
            plt.imshow(img)
            plt.savefig(f'{epoch}.png')
            plt.close()
            visualize_loss(d_loss_list, g_loss_list)


        # If at save interval => save generated image samples
        # if epoch % opt.sample_interval == 2:
            # idx = np.random.randint(0, X_train.shape[0], 6)
            # imgs = X_train[idx]

            # img = gen[1].astype('float32')[:,:,::-1]
            # img = 0.5 * img + 0.5
            # plt.imshow(img)
            # plt.savefig(f'saved_img/{epoch}.png')
            # plt.close()
            # sample_images(model, epoch, imgs)

def mask_image(imgs):
    # change mask shape
    mask_shape = imgs.shape[2]
    sketches = np.copy(imgs)
    sketches[:, :, mask_shape // 2:, :] = 1.0
    return sketches

def sample_images(model, epoch, imgs):
    r, c = 3, 6

    # masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_image(imgs)
    masked_imgs = mask_image(imgs)
    gen = model.generator.predict(masked_imgs)

    # imgs = 0.5 * imgs + 0.5
    # masked_imgs = 0.5 * masked_imgs + 0.5
    # gen = 0.5 * gen_missing + 0.5

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0,i].imshow(imgs)
        axs[0,i].axis('off')
        axs[1,i].imshow(gen)
        axs[1,i].axis('off')

        # axs[0,i].imshow(imgs[i, :,:])
        # axs[0,i].axis('off')
        # axs[1,i].imshow(masked_imgs[i, :,:])
        # axs[1,i].axis('off')
        # filled_in = imgs[i].copy()
        # filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
        # axs[2,i].imshow(filled_in)
        # axs[2,i].axis('off')
    #fig.savefig("images/%d.png" % epoch)
    plt.close()

def visualize_loss(d_loss, g_loss): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    """
    x = [i for i in range(len(d_loss))]
    plt.plot(x, d_loss, color="blue", label="d_loss")
    plt.plot(x, g_loss, color="green", label="g_loss")
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def save_model(model):
    """
    input param: model is the trained GAN model including both generator and discrinminator 
    input param: train_X is the training dataset, used to sample randomly and generate sketch
    """
    model.discriminator.save("saved_model/discriminator")
    model.generator.save("saved_model/generator")


def load_model(model): 
    model.discriminator = tf.keras.models.load_model("saved_model/discriminator")
    model.generator = tf.keras.models.load_model("saved_model/generator")

if __name__ == '__main__':
    model = ContextualGAN()
        
    # save_model(model)
    # load_model(model)

    # eval = evaluation()

    # Import data from preprocess
    train_input = get_data("sample_data/test")

    train(model, train_input)
    # print(eval.test(model, test_input, test_labels))