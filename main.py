# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import IPython.display


if "X0" not in globals():
    (X0, L0), (_, _) = tf.keras.datasets.mnist.load_data()
    X0 = tf.cast(X0, tf.float32) / 255.0
    X0 = tf.expand_dims(X0, -1)

def log_if_unsafe(func, msg="unsafe operation"):
    '''Function wrapper. Warns if function is not called with safe param'''
    def output_fn(*args, safe=False, **kwargs):
        if not safe: print(f"Warning: {msg}")
        return func(*args, **kwargs)
    return output_fn


class GAN_Core(tf.keras.Model):

    def __init__(self, dis_model, gen_model, z_dims, z_sampler=tf.random.normal, **kwargs):

        super().__init__(**kwargs)
        self.z_dims = z_dims
        self.z_sampler = log_if_unsafe(z_sampler, "z_sampler(shape, ...) is an internal function. Consider using sample_z(n)")
        self.gen_model = gen_model
        self.dis_model = dis_model
        


    def sample_z(self, num_samples, **kwargs):
        '''generates a z realization from the z sampler'''
        return self.z_sampler([num_samples, *self.z_dims[1:]], safe=True)
    
    def discriminate(self, inputs, **kwargs):
        '''predict whether input input is a real entry from the true dataset'''
        return self.dis_model(inputs, **kwargs)

    def generate(self, z, **kwargs):
        '''generates an output based on a specific z realization'''
        return self.gen_model(z, **kwargs)


    def call(self, inputs, **kwargs):
        b_size = tf.shape(inputs)[0]
        ## TODO: Implement call process per instructions
        z_samp = self.sample_z(b_size)   ## Generate a z sample
        g_samp = self.generate(z_samp)   ## Generate an x-like image
        d_samp = self.discriminate(inputs)   ## Predict whether x-like is real
        print(f'Z( ) Shape = {z_samp.shape}')
        print(f'G(z) Shape = {g_samp.shape}')
        print(f'D(x) Shape = {d_samp.shape}\n')
        return d_samp

    def build(self, input_shape, **kwargs):
        super().build(input_shape=self.z_dims, **kwargs)


bce_func = tf.keras.backend.binary_crossentropy ## optional
acc_func = tf.keras.metrics.binary_accuracy     ## optional


# TODO: fill in loss functions!
def g_loss(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    l = bce_func(tf.ones_like(d_fake), d_fake)
    return tf.reduce_mean(l)

def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    l = bce_func(tf.zeros_like(d_fake), d_fake) 
    l += bce_func(tf.ones_like(d_real), d_real)
    return tf.reduce_mean(l)

# TODO: fill in accuracy functions!
def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return acc_func(tf.zeros_like(d_fake), d_fake)

def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.ones_like(d_fake), d_fake)

def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.ones_like(d_real), d_real)

################################################################################

def test_metric(loss_fn, fake, real, expected):
    value = loss_fn(tf.constant(fake), tf.constant(real)).numpy()
    print(f'{loss_fn.__name__:11} with ' + 
        f'fake {str(fake):15} and real {str(real):15} = ' + 
        f'{value:6.3f}\t[{expected:6.3f}]')

test_metric(d_loss,     [0., 0., 0.], [1., 1., 1.], -0.0)
test_metric(d_loss,     [0., 0., 0.], [0., 0., 0.], 15.425)
test_metric(d_loss,     [1., 1., 1.], [1., 1., 1.], 15.333)
test_metric(d_loss,     [1., 1., 1.], [0., 0., 0.], 30.758)
print()
test_metric(g_loss,     [1., 1., 1.], [],           -0.0)
test_metric(g_loss,     [0., 0., 0.], [],           15.425)
print()
test_metric(g_acc,      [.1, .9, .9], [],           0.667)
test_metric(d_acc_fake, [.1, .9, .9], [],           0.333)
test_metric(d_acc_real, [], [.1, .9, .9],           0.667)

#@markdown You don't really have to look at it, so just run the block with the play button.

#@markdown * If you open up the code, double-click the right side of the cell to hide it again 
import matplotlib.pyplot as plt
from PIL import Image
import io

class EpochVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_inputs, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.sample_inputs = sample_inputs
        self.imgs = [] 

    def on_epoch_end(self, epoch, logs=None):
        x_real, z_samp = self.sample_inputs
        x_fake = self.model.gen_model(z_samp)
        d_real = tf.nn.sigmoid(self.model.dis_model(x_real))
        d_fake = tf.nn.sigmoid(self.model.dis_model(x_fake))
        outputs = tf.concat([x_real, x_fake], axis=0)
        labels  = [f"D(true x) = {np.round(100 * d, 0)}%" for d in d_real] 
        labels += [f"D(fake x) = {np.round(100 * d, 0)}%" for d in d_fake]

        self.add_to_imgs(
            outputs = outputs,
            labels = labels,
            epoch = epoch
        )

    def add_to_imgs(self, outputs, labels, epoch, nrows=1, ncols=8, figsize=(16, 5)):
        '''
        Plot the image samples in outputs in a pyplot figure and add the image 
        to the 'imgs' list. Used to later generate a gif. 
        '''
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == 1: axs = np.array([axs])
        fig.suptitle(f'Epoch {epoch+1}')
        axs[0][0].set_title(f'Epoch {epoch+1}')
        for i, ax in enumerate(axs.reshape(-1)):
            out_numpy = np.squeeze(outputs[i].numpy(), -1)
            ax.imshow(out_numpy, cmap='gray')
            ax.set_title(labels[i])
        self.imgs += [self.fig2img(fig)]
        plt.close(fig)

    @staticmethod
    def fig2img(fig):
        """
        Convert a Matplotlib figure to a PIL Image and return it
        https://stackoverflow.com/a/61754995/5003309
        """
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)
    
    def save_gif(self, filename='mnist_recon', loop=True, duration=500):
        imgs = self.imgs
        self.imgs[0].save(
            filename+'.gif', save_all=True, append_images=self.imgs[1:], 
            loop=loop, duration=duration)


leaky_relu = tf.keras.layers.LeakyReLU(0.01)

def get_dis_model(name="dis_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256), 
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(256), 
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name=name)

def get_gen_model(name="gen_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000), 
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(1000), 
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(784, activation="tanh"), 
        tf.keras.layers.Reshape((28, 28, 1))
    ], name=name)

gan_model = GAN_Core(    
    dis_model = get_dis_model(), 
    gen_model = get_gen_model(), 
    z_dims = (None, 96),
    name="gan"
)

gan_model.build(input_shape = X0.shape)
# gan_model.summary()
gan_model.gen_model.summary()
gan_model.dis_model.summary()

class GAN(GAN_Core):

    def compile(self, optimizers, losses, accuracies, **kwargs):
        super().compile(
            loss        = losses.values(),
            optimizer   = optimizers.values(),
            metrics     = accuracies.values(),
            **kwargs
        )
        self.loss_funcs = losses
        self.optimizers = optimizers
        self.acc_funcs  = accuracies


    def fit(self, *args, d_steps=1, g_steps=1, **kwargs):
        self.g_steps = g_steps
        self.d_steps = d_steps
        super().fit(*args, **kwargs)


    def test_step(self, data): 
        x_real, l_real = data
        batch_size = tf.shape(x_real)[0]

        ## TODO: Generate the inputs for the loss functions: 
        ## - x_real: Real Images from dataset
        ## - d_real: The discriminator's prediction of the reals
        ## - x_fake: Images generated by generator
        ## - d_fake: The discriminator's prediction of the fakes
        z = self.sample_z(l_real)
        x_fake = self.gen_model(z)
        d_fake = self.dis_model(x_fake)
        d_real = self.dis_model(x_real)

        ########################################################################

        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }

    
    def train_step(self, data):
        x_real, l_real = data
        batch_size = tf.shape(x_real)[0]

        ## TODO: Sample z realization to maintain some batch-level consistency
        z = self.sample_z(l_real)
          
        ## TODO: Train the discriminator for `self.d_steps` steps
        with tf.GradientTape() as tape:
          x_fake = self.gen_model(z, trainable=False)
          d_fake = self.dis_model(x_fake, trainable=True)
          d_real = self.dis_model(x_real, trainable=True)
          loss_fn  = self.loss_funcs['d_loss'](d_fake, d_real) ## HINT
        
        optimizer = self.optimizers['d_opt']  ## HINT
        train_vars = self.dis_model.trainable_variables
        gradients = tape.gradient(loss_fn, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

        ## TODO: Train the generator for `self.g_steps` steps
        with tf.GradientTape() as tape:
          x_fake = self.gen_model(z, trainable=True)
          d_fake = self.dis_model(x_fake, trainable=False)
          d_real = self.dis_model(x_real, trainable=False)
          loss_fn  = self.loss_funcs['g_loss'](d_fake, d_real) ## HINT
        
        optimizer = self.optimizers['g_opt']  ## HINT
        train_vars = self.gen_model.trainable_variables
        gradients = tape.gradient(loss_fn, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

        ## TODO: Compute final states for metric computation (if necessary)
        x_fake = self.gen_model(z, trainable=False)
        d_fake = self.dis_model(x_fake, trainable=False)
        d_real = self.dis_model(x_real, trainable=False)

        ########################################################################

        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }

gan_model = GAN(   
    dis_model = get_dis_model(), 
    gen_model = get_gen_model(), 
    z_dims    = (None, 96),
    name      = "gan"
)

gan_model.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.Adam(1e-3, beta_1=0.5), 
        'g_opt' : tf.keras.optimizers.Adam(1e-3, beta_1=0.5), 
    },
    losses = {
        'd_loss' : d_loss,
        'g_loss' : g_loss,
    },
    accuracies = {
        'd_acc_real' : d_acc_real,
        'd_acc_fake' : d_acc_fake,
        'g_acc'      : g_acc,
    }
)

train_num = 10000       ## Feel free to bump this up to 50000 when your architecture is done
true_sample = X0[train_num-2:train_num+2]       ## 4 real images
fake_sample = gan_model.sample_z(4)             ## 4 z realizations
viz_callback = EpochVisualizer(gan_model, [true_sample, fake_sample])

gan_model.fit(
    X0[:train_num], L0[:train_num], 
    d_steps    = 5, 
    g_steps    = 5, 
    epochs     = 10, ## Feel free to bump this up to 20 when your architecture is done
    batch_size = 50,
    callbacks  = [viz_callback]
)

viz_callback.save_gif('generation')
IPython.display.Image(open('generation.gif','rb').read())