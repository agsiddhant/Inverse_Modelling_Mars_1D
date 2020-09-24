"""
A Mixture Density Layer for Keras

cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer

Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
for a starting point for this code.

----------------------------------------------------------------------
Modified by Siddhant Agarwal 2020 (German Aerospace Center, Berlin)
----------------------------------------------------------------------

Provided under MIT License
"""
from .version import __version__
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import layers
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp


class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Mixture weights (pi) are trained in as logits, not in the softmax space.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus', dtype="float64")  # mix*output vals, no activation
            self.mdn_sigmas = layers.Dense(self.num_mix * int((self.output_dim*(self.output_dim-1)/2 +self.output_dim)), name='mdn_sigmas', dtype="float64") 
            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi', dtype="float64")
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                         name='mdn_outputs')
        return mdn_out

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        sigNum = int(output_dim*(output_dim-1)/2 +output_dim)
        
        y_pred = tf.reshape(tf.cast(y_pred,"float64"), [-1, num_mixes*output_dim + num_mixes*sigNum + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(tf.cast(y_true,"float64"), [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * sigNum,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits_mu = [output_dim] * num_mixes
        component_splits_sig = [sigNum] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits_mu, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits_sig, axis=1)
        low_diag = tfp.bijectors.FillTriangular(upper=False)

        sigs = [low_diag.forward(scale) for scale in sigs]
        sigs = [tf.matrix_set_diag(scale,tf.exp(tf.linalg.diag_part(scale))) for scale in sigs]
        
        covs = [tf.matmul(scale,tf.transpose(scale, perm=[0,2,1])) for scale in sigs]
        
        #for i in range(len(covs)):
        #    Kf = (covs[i] + tf.transpose(covs[i],perm=[0,2,1]))/2.
        #    e,v = tf.self_adjoint_eig(Kf)
        #    e = tf.where(e > 1e-5, e, 1e-5*tf.ones_like(e))
        #    covs[i] = tf.matmul(tf.matmul(v,tf.matrix_diag(e),transpose_a=True),v)
        
        coll = [tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=scale) for loc, scale
                in zip(mus, covs)]
        
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func


def get_mixture_log_likelihood(y_true, y_pred, num_mixes, output_dim):
    sigNum = int(output_dim*(output_dim-1)/2 +output_dim)
    out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                     num_mixes * sigNum,
                                                                     num_mixes],
                                         axis=-1, name='mdn_coef_split')
    # Construct the mixture models
    cat = tfd.Categorical(logits=out_pi)
    component_splits_mu = [output_dim] * num_mixes
    component_splits_sig = [sigNum] * num_mixes
    mus = tf.split(out_mu, num_or_size_splits=component_splits_mu, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=component_splits_sig, axis=1)
    low_diag = tfp.bijectors.FillTriangular(upper=False)

    sigs = [low_diag.forward(scale) for scale in sigs]
    sigs = [tf.matrix_set_diag(scale,tf.exp(tf.linalg.diag_part(scale))) for scale in sigs]

    covs = [tf.matmul(scale,tf.transpose(scale, perm=[0,2,1])) for scale in sigs]

    coll = [tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=scale) for loc, scale
            in zip(mus, covs)]

    mixture = tfd.Mixture(cat=cat, components=coll)
    log_l = mixture.log_prob(y_true)
    return log_l


def softmax(w):
    """Softmax function for a list or numpy array of logits

    Arguments:
    w -- a list or numpy array of logits
    """
    e = np.array(w) 
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist