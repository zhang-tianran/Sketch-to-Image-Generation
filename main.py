# Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/context_encoder/context_encoder.py

import argparse
import numpy as np
import cv2

from models import *
from evaluation import *
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

def train(model, X_train):

    # Adversarial ground truths
    valid = np.ones((opt.batch_size, 1))
    fake = np.zeros((opt.batch_size, 1))

    for epoch in range(opt.epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], opt.batch_size)
        imgs = X_train[idx]

        # masked_imgs, missing_parts, _ = model.mask_randomly(imgs)
        sketch = mask_image(imgs)

        z_sample = tf.Variable(tf.random.truncated_normal((1, 100)))

        # Generate a batch of new images
        gen = model.generator.predict(sketch)

        # Train the discriminator
        d_loss_real = model.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = model.discriminator.train_on_batch(gen, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator 
        # ---------------------

        g_loss = model.combined.train_on_batch(sketch, [imgs, valid])

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

        # If at save interval => save generated image samples
        if epoch % opt.sample_interval == 0:
            idx = np.random.randint(0, X_train.shape[0], 6)
            imgs = X_train[idx]
            sample_images(model, epoch, imgs)

def mask_image(imgs):
    # change mask shape
    mask_shape = imgs.shape[2]
    sketches = np.copy(imgs)
    sketches[:, :, mask_shape // 2:, :] = 0.0
    return sketches

def sample_images(model, epoch, imgs):
    r, c = 3, 6

    # masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_image(imgs)
    masked_imgs = mask_image(imgs)
    gen_missing = model.generator.predict(masked_imgs)

    imgs = 0.5 * imgs + 0.5
    masked_imgs = 0.5 * masked_imgs + 0.5
    gen_missing = 0.5 * gen_missing + 0.5

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0,i].imshow(imgs[i, :,:])
        axs[0,i].axis('off')
        axs[1,i].imshow(masked_imgs[i, :,:])
        axs[1,i].axis('off')
        filled_in = imgs[i].copy()
        filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
        axs[2,i].imshow(filled_in)
        axs[2,i].axis('off')
    fig.savefig("images/%d.png" % epoch)
    plt.close()

def save_model(model):

    def save(model, model_name):
        model_path = "saved_model/%s.json" % model_name
        weights_path = "saved_model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(model.generator, "generator")
    save(model.discriminator, "discriminator")


if __name__ == '__main__':
    model = ContextualGAN()
    # eval = evaluation()
    # TODO Import data from preprocess
    train_input = get_data("sample_data/apple_sketch")
    train(model, train_input)
    # print(eval.test(model, test_input, test_labels))
    # save_model(model)