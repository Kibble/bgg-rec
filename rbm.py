import pandas as pd

import tensorflow as tf
import utils1
import numpy as np
import matplotlib.pyplot as plt

# Define a function to return only the generated hidden states
def hidden_layer(v0_state, W, hb):
    # Probabilities of the hidden units
    h0_prob = tf.nn.sigmoid(tf.linalg.matmul([v0_state], W) + hb)
    # sample_h_given_X
    h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob))))
    return h0_state

# Back Propagation
# Define a function to return the reconstructed output
def reconstructed_ouptput(h0_state, W, vb):
    v1_prob = tf.nn.sigmoid(tf.linalg.matmul(h0_state, tf.transpose(W)) + vb)
    # sample_v_given_h
    v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random.uniform(tf.shape(v1_prob))))[0]
    return v1_state

# v1 = reconstructed_ouptput(h0, W, visible_layer_bias)

# Mean Absolute Error
def error(v0_state, v1_state):
    return tf.reduce_mean(tf.square(v0_state - v1_state))

def train(X, hp, verbose=1):
    errors = []
    weights = []
    train_ds = tf.data.Dataset.from_tensor_slices((np.float32(X))).batch(hp['batchsize'])

    for epoch in range(hp['epochs']):
        batch_number = 0
        for batch_x in train_ds:
            for i_sample in range(len(batch_x)):
                for k in range(hp['K']):
                    v0_state = batch_x[i_sample]
                    h0_state = hidden_layer(v0_state, W, hidden_layer_bias)
                    v1_state = reconstructed_ouptput(h0_state, W, visible_layer_bias)
                    h1_state = hidden_layer(v1_state, W, hidden_layer_bias)

                    delta_W = tf.linalg.matmul(tf.transpose([v0_state]), h0_state) \
                                - tf.linalg.matmul(tf.transpose([v1_state]), h1_state)
                    
                    # Update weights
                    W = W + hp['alpha'] * delta_W

                    # Update biases
                    visible_layer_bias = visible_layer_bias \
                        + hp['alpha'] * tf.reduce_mean(v0_state - v1_state, 0)
                    hidden_layer_bias = hidden_layer_bias \
                        + hp['alpha'] * tf.reduce_mean(h0_state - h1_state, 0)

                    v0_state = v1_state

                if i_sample == len(batch_x) - 1:
                    err = error(batch_x[i_sample], v1_state)
                    errors.append(err)
                    weights.append(W)
                    if verbose > 0:
                        print('Epoch: %d' % (epoch + 1),
                            'batch #: %i ' % (batch_number + 1), 'of %i' % (len(X) / hp['batchsize'] + 1),
                            'sample #: %i ' % (i_sample + 1),
                            'reconstruction error: %f' % err)
                    
            batch_number += 1