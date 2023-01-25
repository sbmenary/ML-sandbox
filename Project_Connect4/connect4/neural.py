###
###  connect4.neural.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of neural networks.
"""

import logging

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers     import Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Softmax
from tensorflow.keras.losses     import Loss, MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam


##  Global logger for this module
logger = logging.getLogger(__name__)



###======================###
###   Method defitions   ###
###======================###


def create_model(board_size=(7, 6), num_filters=40, num_conv_blocks=4, batch_norm=False, max_pool=False, skip_connect=True,
                 kernel_size=3, dropout=0.1, dense_width=400, num_dense=3, learning_rate=0.001, name=None) -> Model :
    """
    Create a neural policy/value model with a configurable architecture.
    """
    
    ##  Create input layer
    inp = Input((board_size[0], board_size[1], 1), name="game_board_input")
    
    ##  Pass through a number of Conv blocks made up of [Conv2d, skip_connect, batch_norm, dropout, max_pool] as configured
    x = inp
    for block_idx in range(num_conv_blocks) :
        block_in = x
        x = Conv2D(num_filters, kernel_size=kernel_size, padding="same", 
                   activation="tanh", name=f"conv{block_idx}_conv2d")(x)
        if skip_connect : x = Concatenate(name=f"conv{block_idx}_skipconnect")([x, block_in])
        if batch_norm   : x = BatchNormalization(name=f"conv{block_idx}_batchnorm")(x)
        if dropout > 0  : x = Dropout(dropout, name=f"conv{block_idx}_dropout")(x)
        if max_pool     : x = MaxPooling2D(name=f"conv{block_idx}_maxpool")(x)
    
    ##  Flatten output of conv blocks
    x = Flatten(name="flatten")(x)
    
    ##  Pass through a number of Dense blocks made up of [Dense, batch_norm, dropout] as configured
    for block_idx in range(num_dense) :
        x = Dense(dense_width, activation="relu", name=f"dense{block_idx}_feedforward")(x)
        if batch_norm  : x = BatchNormalization(name=f"dense{block_idx}_batchnorm")(x)
        if dropout > 0 : x = Dropout(dropout, name=f"dense{block_idx}_dropout")(x)
        
    ##  Pass through one further dense layer, then connect to Softmax output for output policy
    xp = Dense(dense_width, activation="relu", name="policy_process")(x)
    xp = Dense(board_size[0], activation="linear", name="policy_logit")(xp)
    xp = Softmax(name="policy")(xp)
    
    ##  Pass through one further dense layer, then connect to single neuron for output value, constrained to [-1, 1] by tanh activation
    xv = Dense(dense_width, activation="relu", name="value_process")(x)
    xv = Dense(1, activation="tanh", name="value")(xv)
        
    ##  Create Model with input game_board (dims = [h, v, 1]) and outputs [policy vector (dims = [h]), value (dims = [1])]
    x = Model(inp, [xp, xv], name=name)

    ##  Compile model with loss function CrossEntropy(policy) + MSE(value)
    ##  -  use Adam optimizer with configured learning rate
    x.compile(loss=["categorical_crossentropy", "mse"], optimizer=Adam(learning_rate=learning_rate))

    ##  Return model
    return x



def load_model(model_name:str) -> Model :
    """
    Load policy/value model from the directory provided.
    """
    return tf.keras.models.load_model(model_name)




##======================##
##  Deprecated methods  ##
##======================##

##
##  Deprecated: methods for defining a custom loss function as the sum of 
##  -  might be needed again if we decide to introduce a Lagrange multiplier
##
'''
def load_model(model_name) :
    #return tf.keras.models.load_model(model_name, custom_objects={'Loss_NeuralMCTS':Loss_NeuralMCTS})
    return tf.keras.models.load_model(model_name)


def Loss_NeuralMCTS(Loss) :
    
    def call(self, y_true, y_pred):
        y_true_p, y_true_v = y_true
        y_pred_p, y_pred_v = y_pred
        return MeanSquaredError(y_true_v, y_pred_v) + CategoricalCrossentropy(y_true_p, y_pred_p)
'''
