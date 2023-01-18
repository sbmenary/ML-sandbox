###
###  connect4.neural.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of neural networks.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers     import Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Softmax
from tensorflow.keras.losses     import Loss, MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam



def create_model(board_size=(7, 6), num_filters=40, num_conv_blocks=4, batch_norm=False, max_pool=False, skip_connect=True,
                 kernel_size=3, dropout=0.1, dense_width=400, num_dense=3, learning_rate=0.001, name=None) :
    
    inp = Input((board_size[0], board_size[1], 1), name="game_board_input")
    
    x = inp
    for block_idx in range(num_conv_blocks) :
        block_in = x
        x = Conv2D(num_filters, kernel_size=kernel_size, padding="same", 
                   activation="tanh", name=f"conv{block_idx}_conv2d")(x)
        if skip_connect : x = Concatenate(name=f"conv{block_idx}_skipconnect")([x, block_in])
        if batch_norm   : x = BatchNormalization(name=f"conv{block_idx}_batchnorm")(x)
        if dropout > 0  : x = Dropout(dropout, name=f"conv{block_idx}_dropout")(x)
        if max_pool     : x = MaxPooling2D(name=f"conv{block_idx}_maxpool")(x)
    
    x = Flatten(name="flatten")(x)
    
    for block_idx in range(num_dense) :
        x = Dense(dense_width, activation="relu", name=f"dense{block_idx}_feedforward")(x)
        if batch_norm  : x = BatchNormalization(name=f"dense{block_idx}_batchnorm")(x)
        if dropout > 0 : x = Dropout(dropout, name=f"dense{block_idx}_dropout")(x)
        
    xp = Dense(dense_width, activation="relu", name="policy_process")(x)
    xp = Dense(board_size[0], activation="linear", name="policy_logit")(xp)
    xp = Softmax(name="policy")(xp)
    
    xv = Dense(dense_width, activation="relu", name="value_process")(x)
    xv = Dense(1, activation="tanh", name="value")(xv)
        
    x = Model(inp, [xp, xv], name=name)
    x.compile(loss=["categorical_crossentropy", "mse"], optimizer=Adam(learning_rate=learning_rate))
    return x



def load_model(model_name) :
	#return tf.keras.models.load_model(model_name, custom_objects={'Loss_NeuralMCTS':Loss_NeuralMCTS})
	return tf.keras.models.load_model(model_name)



'''def Loss_NeuralMCTS(Loss) :
    
    def call(self, y_true, y_pred):
        y_true_p, y_true_v = y_true
        y_pred_p, y_pred_v = y_pred
        return MeanSquaredError(y_true_v, y_pred_v) + CategoricalCrossentropy(y_true_p, y_pred_p)'''
