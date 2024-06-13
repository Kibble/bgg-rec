import pandas as pd

import tensorflow as tf
import utils1
import numpy as np
import matplotlib.pyplot as plt

class Model:
    X, hiddenUnits, visibleUnits, W, errors, weights = [], [], [], [], [], []
    v0, visible_layer_bias, hidden_layer_bias = [], [], []

    def __init__(self, X, visibleUnits, hiddenUnits):
        self.initialize(X, visibleUnits, hiddenUnits)

    def initialize(self, X, visibleUnits, hiddenUnits):
        self.X = X
        self.hiddenUnits = hiddenUnits
        self.visibleUnits = visibleUnits
        self.v0 = tf.Variable(X[0])
        # Set the bias of visible layer to 0
        self.visible_layer_bias = tf.Variable(tf.zeros([self.visibleUnits]), tf.float32)

        # Set the bias of hidden layer to 0
        self.hidden_layer_bias = tf.Variable(tf.zeros([hiddenUnits]), tf.float32)

        # Set the weights to 0
        self.W = tf.Variable(tf.zeros([self.visibleUnits, hiddenUnits]), tf.float32)

        self.errors = []
        self.weights = []

    # Define a function to return only the generated hidden states
    def hidden_layer(self, v0_state, W, hb):
        # Probabilities of the hidden units
        h0_prob = tf.nn.sigmoid(tf.linalg.matmul([v0_state], W) + hb)
        # sample_h_given_X
        h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob))))
        return h0_state

    # Back Propagation
    # Define a function to return the reconstructed output
    def reconstructed_ouptput(self, h0_state, W, vb):
        v1_prob = tf.nn.sigmoid(tf.linalg.matmul(h0_state, tf.transpose(W)) + vb)
        # sample_v_given_h
        v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random.uniform(tf.shape(v1_prob))))[0]
        return v1_state

    # Mean Absolute Error
    def error(self, v0_state, v1_state):
        return tf.reduce_mean(tf.square(v0_state - v1_state))

    def train(self, hp, verbose=1):
        train_ds = tf.data.Dataset.from_tensor_slices((np.float32(self.X))).batch(hp['batchsize'])

        for epoch in range(hp['epochs']):
            batch_number = 0
            for batch_x in train_ds:
                for i_sample in range(len(batch_x)):
                    for k in range(hp['K']):
                        v0_state = batch_x[i_sample]
                        h0_state = self.hidden_layer(v0_state, self.W, self.hidden_layer_bias)
                        v1_state = self.reconstructed_ouptput(h0_state, self.W, self.visible_layer_bias)
                        h1_state = self.hidden_layer(v1_state, self.W, self.hidden_layer_bias)

                        delta_W = tf.linalg.matmul(tf.transpose([v0_state]), h0_state) \
                                    - tf.linalg.matmul(tf.transpose([v1_state]), h1_state)
                        
                        # Update weights
                        self.W = self.W + hp['alpha'] * delta_W

                        # Update biases
                        self.visible_layer_bias = self.visible_layer_bias \
                            + hp['alpha'] * tf.reduce_mean(v0_state - v1_state, 0)
                        self.hidden_layer_bias = self.hidden_layer_bias \
                            + hp['alpha'] * tf.reduce_mean(h0_state - h1_state, 0)

                        v0_state = v1_state

                    if i_sample == len(batch_x) - 1:
                        err = self.error(batch_x[i_sample], v1_state)
                        self.errors.append(err)
                        self.weights.append(self.W)
                        if verbose > 0:
                            print('Epoch: %d' % (epoch + 1),
                                'batch #: %i ' % (batch_number + 1), 'of %i' % (len(self.X) / hp['batchsize'] + 1),
                                'sample #: %i ' % (i_sample + 1),
                                'reconstruction error: %f' % err)
                        
                batch_number += 1

    def plot_results(self):
        plt.plot(self.errors)
        plt.ylabel('Error')
        plt.xlabel('Batch')
        plt.show()

    def predict(self, row, labels, scale=10):
        input = tf.convert_to_tensor(row, 'float32')
        v0 = input
        hh0 = tf.nn.sigmoid(tf.linalg.matmul([v0], self.W) + self.hidden_layer_bias)
        vv1 = tf.nn.sigmoid(tf.linalg.matmul(hh0, tf.transpose(self.W)) + self.visible_layer_bias)
        rec = vv1

        tf.maximum(rec, 1)

        pred_df = pd.DataFrame({'Game': labels})
        pred_df = pred_df.assign(RecommendationScore = rec[0])
        pred_df = pred_df.assign(Actual = row)
        pred_df['RecommendationScore'] = pred_df['RecommendationScore'].apply(
            lambda x: round(scale * x, 1)
        )
        pred_df['Actual'] = pred_df['Actual'].apply(
            lambda x: round(scale * x, 1)
        )
        pred_df.sort_values(['RecommendationScore'], ascending=False, inplace=True)

        return pred_df
    
    def test(self, pred):
        pred = pred.loc[pred['Actual'] != 0].copy()
        # print(pred)
        pred['Error'] = pred.apply(lambda r: (r['RecommendationScore'] - r['Actual'])**2, axis=1)
        print(f"Mean Squared Error: {pred[['Error']].mean()}")
        # return pred[['Error']].mean()

    def save(self, filename):
        df_W = pd.DataFrame(data=self.W.numpy())
        df_W.to_csv(f"saved_models/{filename}_weights.csv", index=False)
        df_hidden_bias = pd.DataFrame(data=self.hidden_layer_bias.numpy())
        df_hidden_bias.to_csv(f"saved_models/{filename}_hidden_layer_bias.csv", index=False)
        df_visible_bias = pd.DataFrame(self.visible_layer_bias.numpy())
        df_visible_bias.to_csv(f"saved_models/{filename}_visible_layer_bias.csv", index=False)
    
    def load(self, filename):
        df_W = pd.read_csv(f"saved_models/{filename}_weights.csv")
        self.W = tf.constant(df_W.values, tf.float32)
        df_hidden_bias = pd.read_csv(f"saved_models/{filename}_hidden_layer_bias.csv")
        self.hidden_layer_bias = tf.constant(np.ravel(df_hidden_bias), tf.float32)
        df_visible_bias = pd.read_csv(f"saved_models/{filename}_visible_layer_bias.csv")
        self.visible_layer_bias = tf.constant(np.ravel(df_visible_bias), tf.float32)