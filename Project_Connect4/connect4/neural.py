###
###  connect4.neural.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of neural networks.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Softmax
from tensorflow.keras.losses import Loss, MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.models import Model



def create_model(board_size=(7, 6), num_filters=10, num_conv_blocks=3, batch_norm=True, max_pool=False, skip_connect=True,
                 dropout=0.1, dense_width=100, num_dense=2,  optimizer="sgd", name=None) :
    
    inp = Input((board_size[0], board_size[1], 1))
    
    x = inp
    for block_idx in range(num_conv_blocks) :
        block_in = x
        x = Conv2D(num_filters, kernel_size=(2,2), padding="same")(x)
        if skip_connect : x = Add()([x, block_in])
        if batch_norm   : x = BatchNormalization()(x)
        if dropout > 0  : x = Dropout(dropout)(x)
        if max_pool     : x = MaxPooling2D()(x)
    
    x = Flatten()(x)
    
    for dense_idx in range(num_dense) :
        x = Dense(dense_width, activation="relu")(x)
        if batch_norm  : x = BatchNormalization()(x)
        if dropout > 0 : x = Dropout(dropout)(x)
        
    xp = Dense(board_size[0], activation="linear")(x)
    xp = Softmax(name="policy")(xp)
    
    xv = Dense(1, activation="linear", name="value")(x)
    
    x = Model(inp, [xp, xv], name=name)
    x.compile(loss=["categorical_crossentropy", "mse"], optimizer=optimizer)
    return x



def load_model(model_name) :
	#return tf.keras.models.load_model(model_name, custom_objects={'Loss_NeuralMCTS':Loss_NeuralMCTS})
	return tf.keras.models.load_model(model_name)



'''def Loss_NeuralMCTS(Loss) :
    
    def call(self, y_true, y_pred):
        y_true_p, y_true_v = y_true
        y_pred_p, y_pred_v = y_pred
        return MeanSquaredError(y_true_v, y_pred_v) + CategoricalCrossentropy(y_true_p, y_pred_p)'''
