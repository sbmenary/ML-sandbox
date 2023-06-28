###
###  mathsformer.tf_objects.py
###  author: S. Menary [sbmenary@gmail.com]
###  FeedForward and Transformer-type layers modified from tf keras tutorial: https://www.tensorflow.org/text/tutorials/transformer
###
"""
Definition of custom keras objects.
"""

from __future__ import annotations

import logging, math

import numpy      as np
import tensorflow as tf
import tensorflow.keras.backend as K

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers    import (Add, Average, BatchNormalization, Concatenate, Dense, Dropout, Layer, 
                                        LayerNormalization, LeakyReLU,  MultiHeadAttention)
from tensorflow.keras.models    import Model, Sequential



##=============##
##   Globals   ##
##=============##

##  Module logger
logger  = logging.getLogger(__name__)

##  Numerical constants
pi, two_pi = math.pi, 2*math.pi


##=============##
##   Methods   ##
##=============##

def create_custom_objects_dict(*layers, model:Model=None) :
    """
    Parse a list of custom keras layers and create a dictionary of "LayerName":LayerName for all custom keras objects in the inheritance tree.
    This is needed when loading a model with custom keras layers from file.

    Inputs:

        >  layers, list
           A list of the required custom layer classes, e.g. [PositionalEncoding, EncoderBlock,...]

        >  model, Model, default=None
           Optionally, use keyword argument to provide a model whose layers will be queried to see whether they are instances of CustomObject
    """
    ##  If model was provided then add its layers to the list
    if type(model) != type(None) : layers += tuple([l.__class__ for l in model.layers])

    ##  Create list of sets of custom objects found in layers
    list_of_sets  = [l.get_custom_objects() if hasattr(l,"get_custom_objects") else {} for l in layers]

    ##  Combine into a single set
    union_of_sets = set().union(*list_of_sets)

    ##  Return custom objects a dictionary of __name__:__class__
    return {l.__name__:l for l in union_of_sets}



def get_nested_sublayers(layer) :
    """
    Return a list of all the nested sublayers found within layer.
    
    Searches for sublayers/submodels within the object attributes.
    Additionally searches for a list called layer.layers if one exists.
    Search is recursive to pick up all nested sublayers.
    
    Inputs:
    
        >  layer, any type but often keras.Model or keras.Layer
           An object containing nested sublayers / submodels to be searched for.
    """

    ##  Initialise list of layers
    layers = []
        
    ##  Check for model.layers
    if hasattr(layer, "layers") and type(layer.layers) is list :
        for sublayer in layer.layers :
            if not (isinstance(sublayer, Layer) or isinstance(sublayer, Model)) : continue
            if sublayer.name in [l.name for l in layers] : continue
            layers.append(sublayer)
            layers += get_nested_sublayers(sublayer)
    
    ##  Check for other attributes that are instances of Layer or Model
    for attr_name, attr in layer.__dict__.items() :
        if not (isinstance(attr, Layer) or isinstance(attr, Model)) : continue
        if attr.name in [l.name for l in layers] : continue
        layers.append(attr)
        layers += get_nested_sublayers(attr)

    ##  Return container
    return layers


def scalar_masked_categorical_accuracy(y, y_pred, mask_value=0) :
    """
    Computes the sparse categorical cross-entropy over only unmasked tokens
    May be used as a tf loss function
    B is batch size, S is sequence length, V is vocab size
    
    Inputs:
    
        >  y, Tensor of shape [B, S]
           Ground truth token ID
    
        >  y_pred, Tensor of shape [B, S, V]
           Predicted token logits for every token in dictionary
           
        >  mask_value, int, default=0
           Token ID to be interpreted as masked
    """
    ##  Create the mask wherever the label is zero
    mask = y != mask_value
    
    ##  Get the predicted token-id using an argmax along the final axis
    y_pred = tf.argmax(y_pred, axis=-1)
    
    ##  Determine whether the predicted token matches the label
    y_pred = tf.cast(y_pred, y.dtype)
    match = y == y_pred
    
    ##  Mask the matches
    match = match & mask
    
    ##  Cast matches and mask to float, and compute masked average
    match = tf.cast(match, dtype=tf.float32)
    mask  = tf.cast(mask , dtype=tf.float32)
    acc   = tf.reduce_sum(match) / tf.reduce_sum(mask)
    
    ##  Return accuracy
    return acc


def scalar_masked_sparse_categorical_crossentropy(y, y_pred, mask_value:int=0, weight_seq_by_length:bool=False) :
    """
    Computes the sparse categorical cross-entropy over only unmasked tokens
    May be used as a tf loss function
    B is batch size, S is sequence length, V is vocab size
    
    Inputs:
    
        >  y, Tensor of shape [B, S]
           Ground truth token ID
    
        >  y_pred, Tensor of shape [B, S, V]
           Predicted token logits for every token in dictionary
           
        >  mask_value, int, default=0
           Token ID to be interpreted as masked
           
        >  weight_seq_by_length, bool, default=False
           If True then a sequence with S' unmasked values receives a weight of S' in the loss combination
    """
    ##  Create the mask wherever the label is zero
    mask = y != mask_value

    logger.info(f"{y.shape, y_pred.shape})")
        
    ##  Calculate the loss for every token, including masked tokens
    loss = tf.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)
    
    ##  Cast the mask to the same dtype as the loss values
    mask = tf.cast(mask, dtype=loss.dtype)
    
    ##  Calculate sum loss over sequence, excluding the masked values
    loss *= mask
    loss  = tf.reduce_sum(loss)
    
    ##  Calculate average loss over the unmasked values if configured
    if not weight_seq_by_length :
        loss /= tf.reduce_sum(mask)
    
    ##  Return loss value
    return loss



##=============================##
##   CustomLayer keras layer   ##
##=============================##
##
class CustomLayer(Layer) :

    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return set()


    @classmethod
    def get_custom_objects(cls) :
        """
        Return a set of the custom objects used by this class and its ancestors, required when loading a model from file.
        """
        list_of_sets = [l.get_custom_objects() if hasattr(l,"get_custom_objects") else set() for l in cls.get_custom_layer_types()]
        return set([cls,]).union(*list_of_sets)



##=========================================##
##   AdaptiveLearningRate keras callback   ##
##=========================================##
##
class AdaptiveLearningRate(Callback) :
    
    def __init__(self, decay_factor:float, patience:int=1, monitor:str=None, mode:str='min', variable=None, logger=None, log_lvl:int=logging.DEBUG) :
        """
        class AdaptiveLearningRate
        
        Reduces learning rate when a given metric stops improving during training
        
        Inputs:
        
            >  decay_factor, float
               Update factor for the target variable
               
            >  patience, int, default=1
               Number of epochs we wait for the metric to improve before updating
               
            >  monitor, str, None
               Which metric to monitor, if None then fall back to first metric in model.metric_names
               
            >  mode, str, default='min'
               Direction in which we want the metric to improve, choose from ['min', 'max']
               
            >  variable, tf.Variable, default=None
               Tensorflow variable to update, if None then fall back to model.optimizer.learning_rate
               
            >  logger, logging.Logger, default=None
               If provided then use to log state changes such as variable updates or fall backs
               
            >  log_lvl, int, default=DEBUG
               Log-level for messages to logger.log if provided
        """
        ##  Check that a valid mode was provided
        mode = mode.lower()
        if mode not in ["min", "max"] :
            raise ValueError(f"mode must be 'min' or 'max', but '{mode}' provided")
            
        ##  Initialise constant variables
        self.decay_factor = decay_factor
        self.patience     = patience
        self.monitor      = monitor
        self.mode         = mode
        self.variable     = variable
        self.logger       = logger
        self.log_lvl      = log_lvl
        
        
    def on_epoch_end(self, epoch_idx:int, logs:dict) :
        """
        Retrieve latest metric value from logs, use to update best value, and update variable if needed.
        Method is run automatically at the end of an epoch during training.
        
        Inputs:
        
            >  epoch_idx, int
               Index of just-completed epoch
        
            >  logs, dict
               Dictionary of all metric names:values evaluated over the epoch
        """
        ##  Get latest value
        monitor_val = logs[self.monitor]
        
        ##  Update run variables
        if np.isnan(self.best_val) or (self.mode == 'min' and  monitor_val < self.best_val) or (self.mode == 'max' and  monitor_val > self.best_val) :
            self.best_val = monitor_val
            self.num_itr  = 0
        else :
            self.num_itr += 1
            
        ##  Update learning rate
        if self.num_itr == self.patience :
            self.update()
            self.reset_run_variables()
            self.epoch_update_indcs.append(epoch_idx)
        
        
    def on_train_begin(self, logs:dict) :
        """
        Initialise state variables:
            >  self.variable, retrieved from model if not already provided
            >  self.monitor, retrieved from model if not already provided
            >  run variables (self.best_val, self.num_itr, self.epoch_update_indcs)
        Method is run automatically at the start of training.
        
        Inputs:
        
            >  logs:dict
               Empty dictionary of logs
        """
        
        ##  If no variable provided then fall back to model.optimizer.learning_rate
        if not self.variable :
            self.variable = self.model.optimizer.learning_rate
            if self.logger :
                self.logger.log(self.log_lvl, f"Setting variable to {self.variable.name}")
            
        ##  If no metric provided then fall back to self.model.metric_names[0]
        if not self.monitor :
            self.monitor = self.model.metric_names[0]
            if self.logger :
                self.logger.log(self.log_lvl, f"Setting monitor to {self.monitor}")
                
        ##  Initialise run variables
        self.reset_run_variables()
        
    
    def reset_run_variables(self) :
        """
        Reset run variables 
          >  self.best_val = np.nan
          >  self.num_itr = 0
          >  self.epoch_update_indcs = []
        """
        self.best_val = np.nan
        self.num_itr  = 0
        self.epoch_update_indcs = []
                
        
    def update(self) :
        """
        Update the variable by a factor of self.decay_factor
        """
        ##  Derive new values
        current_value = self.variable.value()
        new_value     = self.decay_factor * current_value
        
        ##  Log
        if self.logger :
            self.logger.log(self.log_lvl, f"Updating variable {self.variable.name} from {current_value.numpy()} to {new_value.numpy()}")
        
        ##  Assign new value
        self.variable.assign(new_value)


##================================##
##   AttentionBlock keras layer   ##
##================================##
##
class AttentionBlock(CustomLayer) :

    
    def __init__(self, num_heads:int, ndim_hidden:int, ndim_out:int, dropout:float=0, self_attention:bool=False, use_causal_mask:bool=False, skip_connect:bool=True, mixture_skip_connect:bool=False, pre_layer_norm:bool=True, post_layer_norm:bool=False, _mha:MultiHeadAttention=None, **kwargs) :
        '''
        class AttentionBlock

        A keras layer for applying multi-head attention.

        Inputs:

            >  num_heads, int
               Number of heads

            >  ndim_hidden, int
               Number of neurons in the query / key dimension being contracted over

            >  ndim_out, int, default=None
               Number of neurons in the output feature vector

            >  dropout, float, default=0
               Dropout rate

            >  self_attention, bool, default=False
               If True then only provide one sequence which will be used for both query and key

            >  use_causal_mask, bool, default=False
               If True then only attend to previous elements in the sequence

            >  skip_connect, bool, default=True
               Whether to use a skip connection

            >  mixture_skip_connect, bool, default=False
               Whether to implement skip connection using a LearnableMixture

            >  pre_layer_norm , bool, default=True
               Whether to use a layer normalisation on the inputs

            >  post_layer_norm , bool, default=False
               Whether to use a layer normalisation on the outputs

            >  _mha, keras MultiHeadAttention layer, default=None
               Pre-existing MultiHeadAttention layer, perhaps deserialised after loading from file
        '''

        ##  Base class constructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.num_heads            = num_heads
        self.ndim_hidden          = ndim_hidden
        self.ndim_out             = ndim_out
        self.dropout              = dropout
        self.self_attention       = self_attention
        self.use_causal_mask      = use_causal_mask
        self.skip_connect         = skip_connect
        self.mixture_skip_connect = mixture_skip_connect
        self.pre_layer_norm       = pre_layer_norm
        self.post_layer_norm      = post_layer_norm

        ##  Create keras layers
        base_name = self.name
        if _mha : self._mha  = _mha
        else    : self._mha  = MultiHeadAttention(name=f"{base_name}_mha", num_heads=self.num_heads, key_dim=self.ndim_hidden, value_dim=self.ndim_out, dropout=self.dropout, dtype=self.dtype)
        self._mixture        = LearnableMixture  (name=f"{base_name}_mixture"        , dtype=self.dtype) if self.skip_connect and     self.mixture_skip_connect else None
        self._add            = Add               (name=f"{base_name}_add"            , dtype=self.dtype) if self.skip_connect and not self.mixture_skip_connect else None
        self._pre_layernorm  = LayerNormalization(name=f"{base_name}_pre_layer_norm" , dtype=self.dtype) if self.pre_layer_norm  else None
        self._post_layernorm = LayerNormalization(name=f"{base_name}_post_layer_norm", dtype=self.dtype) if self.post_layer_norm else None
        

    def call(self, x, training=None, mask=None) :
        '''
        Apply attention block to inputs x.
        x is a single Tensor if configured to use self attention, otherwise it is a list of two Tensors (the query and reference sequences).
        If using TensorFlow 2.10+ then we should inherit masks from q and v correctly (TO-DO: check in unit tests)
        '''

        ##  Resolve query (current sequence) and value (reference sequence) depending on whether we are configured to use self attention
        if self.self_attention : 
            if type(x) is list :
                raise TypeError(f"With self-attention configured, expect input to be a single Tensor object (not a list)")
            if self.pre_layer_norm : 
                x = self._pre_layernorm(x, training=training)
            q, v = x, x
        else : 
            if type(x) != list :
                raise TypeError(f"Not configured for self-attention, therefore expect input to be a list of two Tensors (query and reference)")
            q, v = x[0], x[1]
            if self.pre_layer_norm : 
                q = self._pre_layernorm(q, training=training)

        ##  Execute attention, skip-connection and layer-normalisation
        y = self._mha(query=q, value=v, use_causal_mask=self.use_causal_mask, training=training)
        if self.skip_connect : 
            q_dims, y_dims = q.shape[-1], y.shape[-1]
            if q_dims != y_dims : raise RuntimeError(f"Cannot apply skip-connection combining tensors of different dimensions {q_dims} and {y_dims}")
            if self.skip_connect and     self.mixture_skip_connect : y = self._mixture([q, y])
            if self.skip_connect and not self.mixture_skip_connect : y = self._add([q, y])
        if self.post_layer_norm : y = self._post_layernorm(y, training=training)
        return y


    def compute_mask(self, *args, **kwargs) :
        """
        Compute the mask generated by the multi-head attention layer
        """
        return self._mha.compute_mask(*args, **kwargs)

    
    @classmethod
    def from_config(cls, config) :
        """
        Create a new AttentionBlock layer from a dictionary generated by the get_config() method.
        """

        ##  Deserialise the MultiHeadAttention layer inside config
        if "_mha" in config :
            config['_mha'] = tf.keras.layers.deserialize(config['_mha'])

        ##  Load from config
        return super().from_config(config)
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain AttentionBlock layers. 
        This dict stores all values we need to pass to __init__ to create a AttentionBlock layer with the same configuration.
        The config dict includes a serialised copy of the MultiHeadAttention layer that must be deserialised upon loading.
        """
        config = super().get_config()
        config.update(
            {
                "num_heads"            : self.num_heads, 
                "ndim_hidden"          : self.ndim_hidden, 
                "ndim_out"             : self.ndim_out, 
                "dropout"              : self.dropout, 
                "self_attention"       : self.self_attention, 
                "use_causal_mask"      : self.use_causal_mask, 
                "skip_connect"         : self.skip_connect, 
                "mixture_skip_connect" : self.mixture_skip_connect,
                "pre_layer_norm"       : self.pre_layer_norm, 
                "post_layer_norm"      : self.post_layer_norm, 
                "_mha"                 : tf.keras.layers.serialize(self._mha),
            })
        return config


    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return (MultiHeadAttention, LearnableMixture,)




##==============================##
##   DecoderBlock keras layer   ##
##==============================##
##
class DecoderBlock(CustomLayer) :


    def __init__(self, ndim_out:int, num_heads:int, ndim_hidden_mha:int, ndim_hidden_ff:int, num_hidden_layers_ff:int=1, dropout_mha:float=0, dropout_ff:float=0, skip_connect:bool=True, mixture_skip_connect:bool=False, pre_layer_norm:bool=True, 
                 post_layer_norm:bool=False, use_causal_mask:bool=True, activation:str="leakyrelu", leakyrelu_gradient:float=0.1, _self_att_block:AttentionBlock=None, _cross_att_block:AttentionBlock=None, **kwargs) :
        """
        A keras layer for applying causally-masked multi-head self-attention, followed by cross-attention to an encoded sequence
        encoded sequence

        Inputs:

            >  ndim_out, int
               Number of neurons in the output layer of attention and feed-forward blocks

            >  num_heads, int
               Number of attention heads to run in parallel

            >  ndim_hidden_mha, int
               Number of neurons in the hidden dimensions of each attention head

            >  ndim_hidden_ff, int
               Number of neurons in the hidden layer(s) of the feed-forward block

            >  num_hidden_layers_ff, int, default=1
               Number of hidden layers in the feed-forward block

            >  dropout_mha, float, default=0
               Dropout rate in the multi-head attention block

            >  dropout_ff, float, default=0
               Dropout rate in the feed-forward block

            >  skip_connect, bool, default=True
               Whether to use skip-connections in the attention and feed-forward blocks

            >  mixture_skip_connect, bool, default=False
               Whether to implement skip connections as a LearnableMixture

            >  pre_layer_norm, bool, default=True
               Whether to use input layer normalisation in the attention and feed-forward blocks

            >  post_layer_norm, bool, default=False
               Whether to use output layer normalisation in the attention and feed-forward blocks

            >  use_causal_mask, bool, default=True
               Whether to apply a causal mask in the multi-head attention block

            >  activation, str, default="relu"
               Activation function for non-linear layers

            >  leakyrelu_gradient, float, default=0.1
               Gradient for the negative side of the leakyrelu activation function

            >  _self_att_block, AttentionBlock, default=None
               Pre-existing AttentionBlock layer for self-attention step, perhaps deserialised after loading from file

            >  _cross_att_block, AttentionBlock, default=None
               Pre-existing AttentionBlock layer for cross-attention step, perhaps deserialised after loading from file
        """
        ##  Initialise base class
        super().__init__(**kwargs)
        
        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.ndim_out             = ndim_out
        self.num_heads            = num_heads
        self.ndim_hidden_mha      = ndim_hidden_mha
        self.ndim_hidden_ff       = ndim_hidden_ff
        self.num_hidden_layers_ff = num_hidden_layers_ff
        self.dropout_mha          = dropout_mha
        self.dropout_ff           = dropout_ff
        self.skip_connect         = skip_connect
        self.mixture_skip_connect = mixture_skip_connect
        self.pre_layer_norm       = pre_layer_norm
        self.post_layer_norm      = post_layer_norm
        self.use_causal_mask      = use_causal_mask
        self.activation           = activation
        self.leakyrelu_gradient   = leakyrelu_gradient
        
        ##  Create keras layers
        base_name = self.name
        if   _self_att_block  : self._self_att_block  = _self_att_block
        else                  : self._self_att_block  = AttentionBlock(name=f"{base_name}_self_attention_block" , ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, mixture_skip_connect=mixture_skip_connect, pre_layer_norm=pre_layer_norm, post_layer_norm=post_layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=use_causal_mask, self_attention=True)
        if   _cross_att_block : self._cross_att_block = _cross_att_block
        else                  : self._cross_att_block = AttentionBlock(name=f"{base_name}_cross_attention_block", ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, mixture_skip_connect=mixture_skip_connect, pre_layer_norm=pre_layer_norm, post_layer_norm=post_layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=False, self_attention=False)
        self._ff_block = FeedForwardBlock(name=f"{base_name}_feedfwd_block", ndim_out=ndim_out, ndim_hidden=ndim_hidden_ff, dropout=dropout_ff, skip_connect=skip_connect, mixture_skip_connect=mixture_skip_connect, pre_layer_norm=pre_layer_norm, post_layer_norm=post_layer_norm, dtype=self.dtype, batch_norm=False, activation=activation, num_hidden_layers=num_hidden_layers_ff)
                    
        
    def call(self, x, training=False, mask=None) :
        """
        Apply a multi-head self-attention and feed-forward block to a sequence.
        """
        seq_dec , seq_enc  = x[0]   , x[1]
        mask_dec, mask_enc = mask[0], mask[1]
        seq_dec = self._self_att_block (seq_dec, training=training)                  # N.B uses internal keras_mask so no mask argument needed
        seq_dec = self._cross_att_block([seq_dec, seq_enc], training=training)       # N.B uses internal keras_mask so no mask argument needed
        seq_dec = self._ff_block       (seq_dec, mask=mask_dec, training=training)   
        return seq_dec


    def compute_mask(self, *args, **kwargs) :
        """
        Compute the mask generated by the multi-head attention layer
        """
        return self._cross_att_block.compute_mask(*args, **kwargs)
    

    @classmethod
    def from_config(cls, config) :
        """
        Create a new DecoderBlock layer from a dictionary generated by the get_config() method.
        """

        ##  Deserialise the AttentionBlock layers inside config
        if "_self_att_block" in config :
            config['_self_att_block'] = tf.keras.layers.deserialize(config['_self_att_block'])
        if "_cross_att_block" in config :
            config['_cross_att_block'] = tf.keras.layers.deserialize(config['_cross_att_block'])

        ##  Load from config
        return super().from_config(config)
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain DecoderBlock layers. 
        This dict stores all values we need to pass to __init__ to create a DecoderBlock layer with the same configuration.
        The config dict includes a serialised copy of the AttentionBlock layer that must be deserialised upon loading.
        """
        config = super().get_config()
        config.update(
            {
                "ndim_out"             : self.ndim_out, 
                "num_heads"            : self.num_heads, 
                "ndim_hidden_mha"      : self.ndim_hidden_mha, 
                "ndim_hidden_ff"       : self.ndim_hidden_ff, 
                "num_hidden_layers_ff" : self.num_hidden_layers_ff, 
                "dropout_mha"          : self.dropout_mha, 
                "dropout_ff"           : self.dropout_ff, 
                "skip_connect"         : self.skip_connect, 
                "mixture_skip_connect" : self.mixture_skip_connect,
                "pre_layer_norm"       : self.pre_layer_norm, 
                "post_layer_norm"      : self.post_layer_norm, 
                "use_causal_mask"      : self.use_causal_mask, 
                "activation"           : self.activation,
                "leakyrelu_gradient"   : self.leakyrelu_gradient,
                "_self_att_block"      : tf.keras.layers.serialize(self._self_att_block),
                "_cross_att_block"     : tf.keras.layers.serialize(self._cross_att_block),
            })
        return config


    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return (AttentionBlock, FeedForwardBlock)
    


##==============================##
##   EncoderBlock keras layer   ##
##==============================##
##
class EncoderBlock(CustomLayer) :


    def __init__(self, ndim_out:int, num_heads:int, ndim_hidden_mha:int, ndim_hidden_ff:int, num_hidden_layers_ff:int=1, dropout_mha:float=0, dropout_ff:float=0, skip_connect:bool=True, mixture_skip_connect:bool=False, pre_layer_norm:bool=True, post_layer_norm:bool=False, use_causal_mask:bool=False, activation:str="leakyrelu", leakyrelu_gradient:float=0.1, _att_block:AttentionBlock=None, **kwargs) :
        """
        A keras layer for applying a multi-head self-attention and feed-forward block to a sequence.

        Inputs:

            >  ndim_out, int
               Number of neurons in the output layer of both the multi-head attention and feed-forward blocks

            >  num_heads, int
               Number of attention heads to run in parallel

            >  ndim_hidden_mha, int
               Number of neurons in the hidden dimensions of each attention head

            >  ndim_hidden_ff, int
               Number of neurons in the hidden layer(s) of the feed-forward block

            >  num_hidden_layers_ff, int, default=1
               Number of hidden layers in the feed-forward block

            >  dropout_mha, float, default=0
               Dropout rate in the multi-head attention block

            >  dropout_ff, float, default=0
               Dropout rate in the feed-forward block

            >  skip_connect, bool, default=True
               Whether to use skip-connections in both the multi-head attention and feed-forward blocks

            >  mixture_skip_connect, bool, default=False
               Whether to implement skip-connections using a LearnableMixture

            >  pre_layer_norm, bool, default=True
               Whether to use input layer normalisation in both the multi-head attention and feed-forward blocks

            >  post_layer_norm, bool, default=False
               Whether to use output layer normalisation in both the multi-head attention and feed-forward blocks

            >  use_causal_mask, bool, default=False
               Whether to apply a causal mask in the multi-head attention block

            >  activation, str, default="leakyrelu"
               Activation function for non-linear layers

            >  leakyrelu_gradient, float, default=0.1
               Gradient for the negative side of the leakyrelu function

            >  _att_block, AttentionBlock, default=None
               Pre-existing AttentionBlock layer, perhaps deserialised after loading from file
        """
        ##  Initialise base class
        super().__init__(**kwargs)
        
        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.ndim_out             = ndim_out
        self.num_heads            = num_heads
        self.ndim_hidden_mha      = ndim_hidden_mha
        self.ndim_hidden_ff       = ndim_hidden_ff
        self.num_hidden_layers_ff = num_hidden_layers_ff
        self.dropout_mha          = dropout_mha
        self.dropout_ff           = dropout_ff
        self.skip_connect         = skip_connect
        self.mixture_skip_connect = mixture_skip_connect
        self.pre_layer_norm       = pre_layer_norm
        self.post_layer_norm      = post_layer_norm
        self.use_causal_mask      = use_causal_mask
        self.activation           = activation
        self.leakyrelu_gradient   = leakyrelu_gradient
        
        ##  Create keras layers
        base_name = self.name
        if _att_block : 
            self._att_block = _att_block
        else : 
            self._att_block = AttentionBlock  (name=f"{base_name}_attention_block", ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, mixture_skip_connect=mixture_skip_connect, pre_layer_norm=pre_layer_norm, post_layer_norm=post_layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=use_causal_mask, self_attention=True)
        self._ff_block      = FeedForwardBlock(name=f"{base_name}_feedfwd_block"  , ndim_out=ndim_out, ndim_hidden=ndim_hidden_ff , dropout=dropout_ff , skip_connect=skip_connect, mixture_skip_connect=mixture_skip_connect, pre_layer_norm=pre_layer_norm, post_layer_norm=post_layer_norm, dtype=self.dtype, batch_norm=False, activation=activation, leakyrelu_gradient=leakyrelu_gradient, num_hidden_layers=num_hidden_layers_ff)
                    

        
    def call(self, x, training=False, mask=None) :
        """
        Apply a multi-head self-attention and feed-forward block to a sequence.
        """
        y = x
        y = self._att_block(y, training=training, mask=mask)
        y = self._ff_block (y, training=training, mask=mask)   
        return y


    def compute_mask(self, *args, **kwargs) :
        """
        Compute the mask generated by the multi-head attention layer
        """
        return self._att_block.compute_mask(*args, **kwargs)
    

    @classmethod
    def from_config(cls, config) :
        """
        Create a new EncoderBlock layer from a dictionary generated by the get_config() method.
        """

        ##  Deserialise the AttentionBlock layer inside config
        if "_att_block" in config :
            config['_att_block'] = tf.keras.layers.deserialize(config['_att_block'])

        ##  Load from config
        return super().from_config(config)
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain EncoderBlock layers. 
        This dict stores all values we need to pass to __init__ to create a EncoderBlock layer with the same configuration.
        The config dict includes a serialised copy of the AttentionBlock layer that must be deserialised upon loading.
        """
        config = super().get_config()
        config.update(
            {
                "ndim_out"             : self.ndim_out, 
                "num_heads"            : self.num_heads, 
                "ndim_hidden_mha"      : self.ndim_hidden_mha, 
                "ndim_hidden_ff"       : self.ndim_hidden_ff, 
                "num_hidden_layers_ff" : self.num_hidden_layers_ff, 
                "dropout_mha"          : self.dropout_mha, 
                "dropout_ff"           : self.dropout_ff, 
                "skip_connect"         : self.skip_connect, 
                "mixture_skip_connect" : self.mixture_skip_connect,
                "pre_layer_norm"       : self.pre_layer_norm, 
                "post_layer_norm"      : self.post_layer_norm, 
                "use_causal_mask"      : self.use_causal_mask, 
                "activation"           : self.activation,
                "leakyrelu_gradient"   : self.leakyrelu_gradient,
                "_att_block"           : tf.keras.layers.serialize(self._att_block),
            })
        return config


    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return (AttentionBlock, FeedForwardBlock)



##===========================##
##   Enumerate keras layer   ##
##===========================##
##
class Enumerate(CustomLayer) :

    def __init__(self, index_from_zero:bool=True, **kwargs) :
        '''
        Class Enumerate

        A keras layer that creates a new tensor enumerating the right-most indices of the input.
        '''
        ##  Dtype should fall back to tf.int32
        kwargs["dtype"] = kwargs.get("dtype", tf.int32)

        ##  Base class contructor
        super().__init__(**kwargs)
        
        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.index_from_zero = index_from_zero
    

    def call(self, x, training=False, mask=None, minimal_dims:bool=True) :
        '''
        Create a tensor which enumerates the positions of the right-most column of x, with length N.
        If minimal_dims=True then output shape is [1, ..., N], otherwise output dims match those of x.
        '''
        ##  Get shape of input tensor
        shape    = tf.shape(x)
        num_idcs = len(shape)     # Number of tensor indices
        length   = shape[-1]      # Dim of right-most index
        
        ##  Create new tensor of enumerations with shape [1, ..., N]
        offset = 0 if self.index_from_zero else 1
        arange = tf.range(offset, offset + length)
        for i in range(num_idcs-1) :
            arange = tf.expand_dims(arange, axis=0)
            
        ##  If configured for minimal dimensionality output then return this new tensor
        if minimal_dims :
            return arange
             
        ##  Otherwise tile the new tensor to match the input dimensions and return that
        tile_size = [shape[i] for i in range(num_idcs-1)]
        tile_size.append(1)            
        return tf.tile(arange, tile_size)

    
    def compute_mask(self, x, mask=None):
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask

    
    def compute_output_shape(self, input_shape):
        """
        Return the expected shape of the output tensor for a given input shape
        """
        return input_shape
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain Enumerate layers. 
        This dict stores all values we need to pass to __init__ to create a Enumerate layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "index_from_zero" : self.index_from_zero, 
            })
        return config



##==================================##
##   FeedForwardBlock keras layer   ##
##==================================##
##
class FeedForwardBlock(CustomLayer) :

    def __init__(self, ndim_out:int, ndim_hidden:int, num_hidden_layers:int=1, dropout:float=0, activation='leakyrelu', activation_out='linear', 
                 leakyrelu_gradient:float=0.1, skip_connect:bool=False, mixture_skip_connect:bool=False, pre_layer_norm:bool=False, post_layer_norm:bool=False, batch_norm:bool=True, **kwargs) :
        '''
        class FeedForwardBlock

        A keras layer for grouping together a short feed-forward block with optional skip connection and layer normalisation. Modified from 
        https://www.tensorflow.org/text/tutorials/transformer.

        Supports propagation of keras mask, as well as model saving and loading. 

        Inputs:

            >  ndim_out, int
               Number of neurons in the linear output layer

            >  ndim_hidden, int
               Number of neurons in the hidden layers

            >  num_hidden_layers, int, default=1
               Number of hidden layers

            >  dropout, float, default=0
               Dropout rate, if <=0 then no dropout is applied

            >  activation, str, default='leakyrelu'
               Activation function for the hidden layers

            >  activation_out, str, default='linear'
               Activation function for the output layers

            >  leakyrelu_gradient, float, default=0.1
               Gradient of the leakyrelu activation function

            >  skip_connect, bool, default=False
               Whether to apply skip connection between the input and output; only possible when ndim_out is equal to the input size, or
               when they can be broadcast to the same size (warning: check that output size is what you expect!)

            >  mixture_skip_connect, bool, default=False
               Whether to implement skip connections using a LearnableMixture

            >  pre_layer_norm, bool, default=False
               Whether to apply layer normalisation on the inputs

            >  post_layer_norm, bool, default=False
               Whether to apply layer normalisation on the outputs

            >  batch_norm, bool, default=True
               Whether to apply batch normalisation on the outputs
        '''

        ##  Base class contructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.ndim_out             = ndim_out
        self.ndim_hidden          = ndim_hidden
        self.num_hidden_layers    = num_hidden_layers
        self.dropout              = dropout
        self.activation           = activation
        self.activation_out       = activation_out
        self.leakyrelu_gradient   = leakyrelu_gradient
        self.skip_connect         = skip_connect
        self.mixture_skip_connect = mixture_skip_connect
        self.pre_layer_norm       = pre_layer_norm
        self.post_layer_norm      = post_layer_norm
        self.batch_norm           = batch_norm

        ##  Create keras sublayers
        self.initialise_layers()


    def call(self, x, training:bool=None, mask=None) :
        '''
        Operate this feed forward block on Tensor object x. The training flag is passed to all sublayers.
        '''
        y = x
        y = self._dense_block(y, training=training)
        y = self._dense_out  (y, training=training)
        if self._leakyrelu_out is not None :
            y = self._leakyrelu_out(y)
        if self.skip_connect : 
            x_dims, y_dims = x.shape[-1], y.shape[-1]
            if x_dims != y_dims : raise RuntimeError(f"Cannot apply skip-connection combining tensors of different dimensions {x_dims} and {y_dims}")
            if self.mixture_skip_connect :
                y = self._mixture([x, y], training=training)
            else :
                y = self._add([x, y], training=training)
        return y


    def compute_mask(self, x, mask=None):
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain FeedForwardBlock layers. 
        This dict stores all values we need to pass to __init__ to create a FeedForwardBlock layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "ndim_out"             : self.ndim_out, 
                "ndim_hidden"          : self.ndim_hidden, 
                "num_hidden_layers"    : self.num_hidden_layers,
                "dropout"              : self.dropout, 
                "activation"           : self.activation, 
                "activation_out"       : self.activation_out, 
                "leakyrelu_gradient"   : self.leakyrelu_gradient, 
                "skip_connect"         : self.skip_connect,
                "mixture_skip_connect" : self.mixture_skip_connect,
                "pre_layer_norm"       : self.pre_layer_norm,
                "post_layer_norm"      : self.post_layer_norm,
                "batch_norm"           : self.batch_norm,
            })
        return config


    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return (LearnableMixture,)


    def initialise_layers(self) :
        '''
        Create new sublayer objects using their default initialisers
        '''
        dtype, base_name, dense_block_layers = self.dtype, self.name, []
        if self.pre_layer_norm  :
            dense_block_layers.append(LayerNormalization(dtype=dtype, name=f"{base_name}_pre_layer_norm"))
        for l_idx in range(self.num_hidden_layers) :
            if self.activation.lower() == "leakyrelu" :
                dense_block_layers.append(Dense(self.ndim_hidden, dtype=dtype, name=f"{base_name}_dense_hidden_{l_idx+1}", activation="linear"))
                dense_block_layers.append(LeakyReLU(self.leakyrelu_gradient, name=f"{base_name}_leakyrelu_{l_idx+1}"))
            else :
                dense_block_layers.append(Dense(self.ndim_hidden, dtype=dtype, name=f"{base_name}_dense_hidden_{l_idx+1}", activation=self.activation))
            if self.dropout > 0 :
                dense_block_layers.append(Dropout(self.dropout, dtype=dtype, name=f"{base_name}_dropout_{l_idx+1}"))
        self._dense_block  = Sequential(dense_block_layers, name=f"{base_name}_dense_block")
        if self.activation_out.lower() == "leakyrelu" :
            self._dense_out     = Dense(self.ndim_out, dtype=dtype, name=f"{base_name}_dense_out", activation="linear")
            self._leakyrelu_out = LeakyReLU(self.leakyrelu_gradient, name=f"{base_name}_leakyrelu_out")
        else :
            self._dense_out     = Dense(self.ndim_out, dtype=dtype, name=f"{base_name}_dense_out", activation=self.activation_out)
            self._leakyrelu_out = None
        if self.skip_connect :
            if self.mixture_skip_connect :
                self._mixture = LearnableMixture(dtype=dtype, name=f"{base_name}_mixture")
                self._add     = None
            else :
                self._mixture = None
                self._add     = Add(dtype=dtype, name=f"{base_name}_add")
        if self.batch_norm : 
            dense_block_layers.append(BatchNormalization(dtype=dtype, name=f"{base_name}_batch_norm"))
        if self.post_layer_norm  :
            dense_block_layers.append(LayerNormalization(dtype=dtype, name=f"{base_name}_post_layer_norm"))



##==========================================##
##   LayerActivationRecord keras callback   ##
##==========================================##
##
class LayerActivationRecord(Callback) :
    
    def __init__(self, batch_frequency:int, val_input, val_output=None) :
        """
        class LayerActivationRecord
        
        Tracks the mean & std dev of the activations from all layers in model.layers during training
        Uses val_input as model input
        
        Note that it is not easy to access the activations of sublayers. This is because we want to access the
        activations through layer.output, which generally exists as part of the computational graph. However,
        layer.sublayer.output is not tracked in the computational graph. This means that layer.sublayer.output
        does not exist even after the layers have been built, and we cannot create keras functions to access 
        them. This means that we currently do not support tracking the activations of sublayers.
        
        Inputs:
            
            >  batch_frequency, int
               Batch frequency with which to measure layer activations
               
            >  val_input, Tensor
               Model input from which layer activations will be derived
        """
        
        ##  Base class constructor
        super().__init__()
        
        ##  Store arguments
        self.batch_frequency = batch_frequency
        self.val_input       = val_input
        
        ##  Initialise containers and variables
        self.batch_indices = []
        self.layer_means   = {}
        self.layer_stds    = {}
        self.layers        = []

    
    @classmethod
    def from_config(cls, config) :
        """
        Create a new LayerActivationRecord layer from a dictionary generated by the get_config() method.
        """

        ##  Deserialise the val_input tensor
        config['val_input'] = tf.io.serialize_tensor(config['val_input'])

        ##  Load from config
        return super().from_config(config)
    

    def get_config(self) :
        """
        Create config dict storing all values needed for __init__ to create a LayerActivationRecord layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "batch_frequency" : self.batch_frequency, 
                "val_input"       : tf.io.serialize_tensor(self.val_input), 
            })
        return config
        
        
    def on_batch_end(self, batch_idx:int, logs:dict=None) :
        """
        Processing to be run at the end of each batch.
        With the given batch frequency, we pass self.val_input through the model and measure the layer activations.
        We record the mean and std for each layer.
        
        Inputs:
        
            >  batch_idx, int
               Index of the batch having just been processed
        """
        
        ##  Only proceed with the given batch frequency
        if (batch_idx != 0) and ((batch_idx+1) % self.batch_frequency != 0) : return
        
        ##  Store the batch index, using self.batch_offset to ensure continuation over epochs
        self.batch_indices.append(self.batch_offset + batch_idx)
        
        ##  Define + run a keras function for calculating a list of the layer activations
        layer_eval_func = tf.keras.backend.function(
            [self.model.input], 
            [layer.output for layer in self.layers]
        )
        layer_outs = layer_eval_func([self.val_input])[0]
        
        ##  Loop over layers and store the summary statistics
        for layer, layer_out in zip(self.layers, layer_outs) :
            self.layer_means[layer.name].append(np.mean(layer_out))
            self.layer_stds [layer.name].append(np.std (layer_out))
    
    
    def on_epoch_begin(self, epoch_idx:int, logs:dict=None) :
        """
        Processing to be run at the start of each epoch.
        Sets self.batch_offset equal to the number of batches already processed.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        
        ##  Set self.batch_offset to the number of batches already processed
        self.batch_offset = epoch_idx * self.num_steps
    
    
    def on_train_begin(self, logs:dict=None) :
        """
        Processing to be run at the start of training.
        
        Re-initialises internal variables and storage containers.
        """
        ##  Initialise variables
        self.num_steps    = self.params["steps"]
        self.batch_offset = 0 
        
        ##  Initialise containers
        self.batch_indices = []
        self.layers        = [l for l in self.model.layers if "tf.__operators__" not in l.name]
        for layer in self.layers :
            self.layer_means[layer.name] = []
            self.layer_stds [layer.name] = []


    def plot(self, num_col:int=3, show:bool=True, savefig:str=None, close:bool=True, dpi:int=200) :
        """
        Create plot showing the spread of layer activations (mean and std) throughout training

        Inputs:
        
            >  num_col, int, default=3
               Number of axis columns

            >  show, bool, default=True
               Whether to call plt.show(fig) with the figure created

            >  savefig, str, default=None
               Optional filename to save the figure to

            >  close, bool, default=True
               Whether to call plt.close(fig) at the end

            >  dpi, int, default=200
               Pixels-per-inch when saving to file, for formats that require this

        Returns:

            >  fig, plt.Figure object, the figure created
        """

        ##  Get names of all layers for which activations have been recorded, in alphabetical order
        layer_names = [layer_name for layer_name, layer_mean in self.layer_means.items() if len(layer_mean) > 0]
        layer_names = sorted(layer_names)

        ##  Calculate number of rows needed
        num_row = math.ceil(len(layer_names) / num_col)

        ##  Create figure object
        fig = plt.figure(figsize=(4*num_col, 4*num_row))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ## Iterate over selected layers
        for layer_idx, layer_name in enumerate(layer_names) :

            ##  Add axis for layer
            ax  = fig.add_subplot(num_row, num_col, 1+layer_idx)
            ax.tick_params(which="both", axis="both", direction="in", left=True, top=True, labelsize=8)
            ax.set_title(layer_name, fontsize=6)

            ##  Pull data from records
            x  = np.array(self.batch_indices)
            y  = np.array(self.layer_means[layer_name])
            ey = np.array(self.layer_stds [layer_name])

            ##  Plot line tracking the layer mean activation, and shade region between std devs
            ax.plot(x, y, "-", lw=3, c='k')
            ax.fill_between(x, y-ey, y+ey, fc="darkblue", alpha=0.2, lw=0)

            ##  Draw text label on the first axis only
            if layer_idx == 0 :
                ax.text(0, 1.2, "Layer activations vs batch index", weight="bold", ha="left", va="bottom", fontsize=16, transform=ax.transAxes)


        ##  Save figure
        if savefig :
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)

        ##  Show figure
        if show :
            plt.show(fig)

        ##  Close figure
        if close :
            plt.close(fig)

        ##  Return figure
        return fig



##=======================================##
##   LayerWeightsRecord keras callback   ##
##=======================================##
##
class LayerWeightsRecord(Callback) :
    
    def __init__(self, batch_frequency:int, recursive:bool=False) :
        """
        class LayerWeightsRecord
        
        Tracks the mean & std dev of the weights from all layers in a model during training.
        
        Inputs:
            
            >  batch_frequency, int
               Batch frequency with which to measure layer weights
               
            >  recursive, bool, default=False
               Whether to recursively search for all nested sublayers
        """
        
        ##  Base class constructor
        super().__init__()
        
        ##  Store arguments
        self.batch_frequency = batch_frequency
        self.recursive       = recursive
        
        ##  Initialise containers and variables
        self.batch_indices = []
        self.layer_means   = {}
        self.layer_stds    = {}
        self.batch_offset  = 0
    

    def get_config(self) :
        """
        Create config dict storing all values needed for __init__ to create a LayerWeightsRecord layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "batch_frequency" : self.batch_frequency, 
                "recursive"       : self.recursive, 
            })
        return config
        
        
    def on_batch_end(self, batch_idx:int, logs:dict=None) :
        """
        Processing to be run at the end of each batch.
        With the given batch frequency, access the current weights for each layer, and record the mean & std dev
        
        Inputs:
        
            >  batch_idx, int
               Index of the batch having just been processed
        """
        
        ##  Only proceed with the given batch frequency
        if (batch_idx != 0) and ((batch_idx+1) % self.batch_frequency != 0) : return
        
        ##  Store the batch index, using self.batch_offset to ensure continuation over epochs
        self.batch_indices.append(self.batch_offset + batch_idx)
        
        ##  Loop over layers and store the summary statistics
        for layer in self.layers :
            weights = layer.get_weights()
            if len(weights) == 0 : continue
            weights = np.concatenate([w.flatten() for w in weights])
            self.layer_means[id(layer)].append(np.mean(weights))
            self.layer_stds [id(layer)].append(np.std (weights))
    
    
    def on_epoch_begin(self, epoch_idx:int, logs:dict=None) :
        """
        Processing to be run at the start of each epoch.
        Sets self.batch_offset equal to the number of batches already processed.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        
        ##  Set self.batch_offset to the number of batches already processed
        self.batch_offset = epoch_idx * self.num_steps
    
    
    def on_train_begin(self, logs:dict=None) :
        """
        Processing to be run at the start of training.
        
        Re-initialises internal variables and storage containers.
        """
        ##  Initialise variables
        self.num_steps    = self.params["steps"]
        self.batch_offset = 0 
        
        ##  Initialise layers
        self.layers = get_nested_sublayers(self.model) if self.recursive else self.model.layers
        self.layers = [l for l in self.layers if "tf.__operators__" not in l.name]
        
        ##  Initialise containers
        self.batch_indices = []
        for layer in self.layers :
            self.layer_means[id(layer)] = []
            self.layer_stds [id(layer)] = []


    def plot(self, num_col:int=3, show:bool=True, savefig:str=None, close:bool=True, dpi:int=200) :
        """
        Create plot showing the spread of layer weights (mean and std) throughout training

        Inputs:
        
            >  num_col, int, default=3
               Number of axis columns

            >  show, bool, default=True
               Whether to call plt.show(fig) with the figure created

            >  savefig, str, default=None
               Optional filename to save the figure to

            >  close, bool, default=True
               Whether to call plt.close(fig) at the end

            >  dpi, int, default=200
               Pixels-per-inch when saving to file, for formats that require this

        Returns:

            >  fig, plt.Figure object, the figure created
        """

        ##  Get names and IDs of all layers for which weights have been recorded
        layer_names, layer_ids = [], []
        for layer in self.layers :
            layer_id, layer_name = id(layer), layer.name
            layer_mean = self.layer_means[layer_id]
            if not len(layer_mean) : continue
            layer_names.append(layer_name)
            layer_ids  .append(layer_id  )

        ##  Calculate number of rows needed
        num_row = math.ceil(len(layer_ids) / num_col)

        ##  Create figure object
        fig = plt.figure(figsize=(4*num_col, 4*num_row))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ## Iterate over selected layers
        for layer_idx, (layer_id, layer_name) in enumerate(zip(layer_ids, layer_names)) :

            ##  Add axis for layer
            ax  = fig.add_subplot(num_row, num_col, 1+layer_idx)
            ax.tick_params(which="both", axis="both", direction="in", left=True, top=True, labelsize=8)
            ax.set_title(layer_name, fontsize=6)

            ##  Pull data from records
            x  = np.array(self.batch_indices)
            y  = np.array(self.layer_means[layer_id])
            ey = np.array(self.layer_stds [layer_id])

            ##  Plot line tracking the layer mean weight, and shade region between std devs
            ax.plot(x, y, "-", lw=3, c='k')
            ax.fill_between(x, y-ey, y+ey, fc="darkblue", alpha=0.2, lw=0)

            ##  Draw text label on the first axis only
            if layer_idx == 0 :
                ax.text(0, 1.2, "Layer weights vs batch index", weight="bold", ha="left", va="bottom", fontsize=16, transform=ax.transAxes)


        ##  Save figure
        if savefig :
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)

        ##  Show figure
        if show :
            plt.show(fig)

        ##  Close figure
        if close :
            plt.close(fig)

        ##  Return figure
        return fig
        


##==================================##
##   LearnableMixture keras layer   ##
##==================================##
##
class LearnableMixture(CustomLayer) :

    def __init__(self, **kwargs) :
        '''
        Class LearnableMixture

        A keras layer for creating a linear combination of several inputs with learnable coefficients.
        Linear coefficients are calculated as softmax(self._weights) with trainable parameters self._weights.
        Number of inputs not known until the build() method is executed during the first call().
        '''

        ##  Base class contructor
        super().__init__(**kwargs)

        ##  Create internal keras layers
        self._add = Add(name=f"{self.name}_add", dtype=self.dtype)


    def build(self, x:tuple) :
        '''
        Configure object and create class from list of inputs
        '''

        ##  Check input is list
        if not isinstance(x, list) :
            raise ValueError(f"A LearnableMixture layer should be called on a list of inputs. Received: {type(x)}")

        ##  Get number of inputs
        self.num_inputs = len(x)

        ##  Create trainable coefficients
        self._weights = self.add_weight(f"{self.name}_weights", shape=(self.num_inputs,), initializer="random_normal", trainable=True, dtype=self.dtype)
    

    def call(self, x) :
        '''
        Create a linear combination of several inputs with learnable coefficients.
        '''
        w = tf.nn.softmax(self._weights)
        y = [tf.math.scalar_mul(w[x_idx], xp) for (x_idx, xp) in enumerate(x)]
        return self._add(y)


    def compute_mask(self, x, mask=None):
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask



##===================================##
##   LoggerCallback keras callback   ##
##===================================##
##
class LoggerCallback(Callback) :
    
    def __init__(self, logger, loglvl:int=logging.INFO) :
        """
        class LoggerCallback
        
        At the end of each epoch, passes the log dictionary on the the logger object.
        
        Inputs:
            
            >  logger, logging.Logger
               Logger object to call
               
            >  loglvl, int, default=logging.INFO
               Log-level for the log calls at the end of each epoch
        """
        
        ##  Base class constructor
        super().__init__()
        
        ##  Store arguments
        self.logger = logger
        self.loglvl = loglvl
    
    
    def on_epoch_end(self, epoch_idx:int, logs:dict=None) :
        """
        Processing to be run at the end of each epoch.
        Prints contents of log dictionary to logger.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        self.logger.log(self.loglvl, f"Training reached the end of epoch at index {epoch_idx}")
        for log_name, log_val in logs.items() :
            self.logger.log(self.loglvl, f"    with metric {log_name}: {log_val:.5}")



##===============================##
##   MaskedMetric keras metric   ##
##===============================##
##
class MaskedMetric : 
    
    def __init__(self, 
                 scalar_output      : bool = True, 
                 equal_token_weight : bool = True, 
                 use_keras_mask     : int  = False, 
                 mask_value         : int  = None, 
                 **kwargs) :
        """
        class MaskedMetric
        
        Calculates metric values excluding masked values. 
        
        If mask_value is not None, we mask datapoints with this truth label
        If use_keras_mask is True, we inherit a _keras_mask from the model outputs
        If both then they are combined
        
        if scalar_output and equal_token_weight:
            -->  returns "mean metric per unmasked token"
            -->  By returning a scalar, we do not allow for use of sample_weight
        
        if scalar_output and not equal_token_weight:
            -->  returns "mean masked metric per sequence"
            -->  By returning a scalar, we do not allow for use of sample_weight
        
        if not scalar_output and equal_token_weight:
            -->  returns "mean masked metric for each row", scaled s.t. reduction-by-avg will return the 
                 "mean metric per unmasked token"
            -->  By returning a tensor, we allow for use of sample_weight - however, this will no longer combine
                 to "mean metric per unmasked token" because we cannot correctly account for the sample_weights 
                 when calculating the scaling factor! This is because we have access "sum-of-mask-over-all-data",
                 whereas we would need access to "weighted-sum-of-mask", which is unavailable
        
        if not scalar_output and not equal_token_weight:
            -->  returns "mean masked metric for each row"
            -->  By returning a tensor, we allow for use of sample_weight - in this case, the reduction will be
                 correct, and we may always interpret the result as "mean masked metric per weighted row"
            -->  This configuration is therefore recommended when using sample_weight to avoid unexpected
                 behaviour, although it means that we weight the tokens of short sequences much more highly than
                 those of long sequences, which is often undesired
        
        N.B. When we output a vector instead of a scalar/tensor, we often find a GraphExecutionError as keras
             fails to squeeze the dimensions. Therefore we output a tensor of values instead.
        N.B. There may be a better way that allows native correct calculation of loss, e.g. using a Masking?
        N.B. Reading the docs for keras.losses.Loss and keras.loss.LossWrapper, it seems like the mask and
             sample_weight should be combined correctly, but this appears to not be so
        """
        ##  Store self config
        self.scalar_output      = scalar_output
        self.equal_token_weight = equal_token_weight
        self.use_keras_mask     = use_keras_mask
        self.mask_value         = mask_value
        self.kwargs             = kwargs
        
        ##  Store additional arguments as self attributes
        for kwarg, val in kwargs.items() :
            setattr(self, kwarg, val)
            
            
    @abstractmethod
    def calculate_metric(self, y_true:tf.Tensor, y_pred:tf.Tensor) :
        """
        Calculate the metric value on per-element basis. Must be implemented by derived class.
        """
        raise NotImplementedError()
        
        
    def __call__(self, y_true:tf.Tensor, y_pred:tf.Tensor) :
        """
        Calculate the metric, masking by value or _keras_mask as configured
        """
        ##  Get base loss
        loss = self.calculate_metric(y_true, y_pred)
        
        ##  Set the dtype as whatever we got from the base loss
        dtype = loss.dtype
        
        ##  Create mask and cast to dtype
        mask = self.get_mask(y_true, y_pred)
        mask = tf.cast(mask, dtype=dtype)
        
        ##  Mask loss
        masked_loss = loss * mask
        
        ##  Return scalar loss assigning equal weight to all unmaksed tokens
        if self.scalar_output and self.equal_token_weight :
            return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        
        ##  Return scalar loss assigning equal weight to all sequences, regardless of num. unmasked tokens
        if self.scalar_output and not self.equal_token_weight :
            ##  First calculate the masked means over each sequence
            seq_means = tf.reduce_sum(masked_loss, axis=-1) / tf.reduce_sum(mask, axis=-1)
            ##  Then return the mean over sequences
            return tf.reduce_mean(seq_means)
        
        ##  Vector outputs are based on sum masked loss across each row
        num_rows    = tf.cast(tf.reduce_prod(tf.shape(y_true)[:-1]), dtype)
        row_width   = tf.cast(tf.shape(y_true)[-1], dtype)
        
        ##  Return vector accuracy with equal weight given to each sequence
        if not self.equal_token_weight :
            loss_per_row = tf.reduce_sum(masked_loss, axis=-1)
            loss_per_row = tf.expand_dims(loss_per_row, axis=-1)
            tile_shape   = [1]*(len(tf.shape(loss_per_row)) - 1) + [row_width]
            tiled_acc    = tf.tile(loss_per_row, tile_shape)
            return tiled_acc 
        
        ##  Return accuracies scaled by number
        return masked_loss * num_rows * row_width / tf.reduce_sum(mask)
        
        '''##  Vector outputs are based on sum masked loss across each row
        loss_per_row = tf.reduce_sum(masked_loss, axis=-1)
        tensor_loss  = loss_per_row[..., tf.newaxis]
        num_rows     = tf.cast(tf.reduce_prod(tf.shape(y_true)[:-1]), dtype)
        row_width    = tf.cast(tf.shape(y_true)[-1], dtype)
        
        ##  Return vector loss with equal weight given to each sequence
        if not self.equal_token_weight :
            return tensor_loss / tf.reduce_sum(mask, axis=-1)[..., tf.newaxis]
        
        ##  Calculate a per-row scale factor that causes avg-over-vector to return avg-per-unmasked-token
        row_sfs  = num_rows / tf.reduce_sum(mask)
        
        ##  Return vector loss assigning equal weight to all unmased tokens
        return tensor_loss * row_sfs[..., tf.newaxis]'''
        
        
    def get_config(self) :
        """
        Return dictionary of config arguments:values needed to recreate object
        """
        config = {}
        config["scalar_output"     ] = self.scalar_output
        config["equal_token_weight"] = self.equal_token_weight
        config["use_keras_mask"    ] = self.use_keras_mask
        config["mask_value"        ] = self.mask_value
        for kwarg, val in self.kwargs.items() :
            config[kwarg] = val
        return config

 
    def get_mask(self, y_true:tf.Tensor=None, y_pred:tf.Tensor=None) :
        """
        Create masking tensor for inputs y_true and y_pred
        
        If configured, a mask is created where y_true entries are equal to self.mask_value
        If configured, a second mask is taken from y_pred._keras_mask
        The up-to-two masks are combined
        
        Inputs:
        
            >  y_true, tf.Tensor, default=None
               Tensor of truth labels, masked according to their values
        
            >  y_pred, tf.Tensor, default=None
               Tensor of model outputs, masked according to its _keras_mask
        """
        ##  Initialise True mask
        mask = True 
        
        ##  Combine with value mask
        if self.mask_value is not None :
            mask = mask & (y_true != self.mask_value)
        
        ##  Combine with keras mask
        if self.use_keras_mask :
            mask = mask & y_pred._keras_mask
                    
        ##  Return
        return mask
    


##============================================##
##   MaskedCategoricalAccuracy keras metric   ##
##============================================##
##
class MaskedCategoricalAccuracy(MaskedMetric) :
    """
    class MaskedCategoricalAccuracy
        
    Calculates categorical accuracy excluding masked values. 
    """
    
    def calculate_metric(self, y_true:tf.Tensor, y_pred:tf.Tensor) :
        """
        Returns the accuracy.
        """
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    


##======================================================##
##   MaskedSparseCategoricalCrossentropy keras metric   ##
##======================================================##
##
class MaskedSparseCategoricalCrossentropy(MaskedMetric) :
    """
    class MaskedSparseCategoricalCrossentropy
        
    Calculates sparse categorical cross-entropy loss excluding masked values. 
        
    Value masks may also be achieved using the basic keras loss function with "ignore_class" argument, but 
    this may lead to unexpected behaviour when using certain reduction schemes along with sample_weight. By
    performing the reduction by hand, we achieve predictable behaviour when using sample_weight - although
    this behaviour might not be what you naively expect - see below!
    """
    
    def calculate_metric(self, y_true:tf.Tensor, y_pred:tf.Tensor) :
        """
        Returns the sparse categorical cross-entropy loss.
        """
        return tf.keras.metrics.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)


##=================================##
##   MetricRecord keras callback   ##
##=================================##
##
class MetricRecord(Callback) :
    
    def __init__(self, batch_frequency:int, data_input, data_output, validation_data=None, label:str="Partial\nval. loss", func=None, 
                 num_bootstrap:int=-1, plot_on_train_end:bool=False, plot_on_epoch_end:bool=False, plot_frequency:int=-1, yscale:str="log", 
                 logger=None, log_lvl:int=logging.DEBUG) :
        """
        class MetricRecord
        
        Tracks the value of a scale function over a validation dataset during training
        
        Inputs:
            
            >  batch_frequency, int
               Batch frequency with which to measure layer activations
               
            >  data_input, Tensor
               Truth inputs
               
            >  data_output, Tensor
               Truth labels

            >  validation_data, (Tensor, Tensor), default=None
               Optional validation data

            >  label, str, default='Partial\\nval. loss'
               Label of function to be tracked

            >  func, tf function, default=None
               Function to be tracked, if None then use model.loss
               Model signature is func(y, y_pred) -> [B,] where y are labels, y_pred is the model output and B is batch size

            >  num_bootstrap, int, default=-1
               If >0 then we use this many bootstraps to estimate the uncertainty on the loss

            >  plot_on_train_end, bool, default=False
               If True then show a plot of the loss curve at the end of training

            >  plot_on_epoch_end, bool, default=False
               If True then show a plot of the loss curve at the end of each epoch

            >  plot_frequency, int, default=-1
               If > 0 then show a plot every time we have generated this many datapoints

            >  yscale, str, default="log"
               Type of yscale to use for plotting

            >  logger, Logger, default=None
               If provided then use this to log the loss curve as it is generated

            >  log_lvl, int, default=DEBUG
               Log level if a logger is being used
        """
        
        ##  Base class constructor
        super().__init__()
        
        ##  Store arguments
        self.batch_frequency   = batch_frequency
        self.data_input        = data_input
        self.data_output       = data_output
        self.validation_data   = validation_data
        self.label             = label
        self.func              = func
        self.num_bootstrap     = num_bootstrap
        self.plot_on_train_end = plot_on_train_end
        self.plot_on_epoch_end = plot_on_epoch_end
        self.plot_frequency    = plot_frequency
        self.yscale            = yscale
        self.logger            = logger
        self.log_lvl           = log_lvl
        
        ##  Initialise containers and variables
        self.batch_indices    = []
        self.epoch_starts     = []
        self.values           = []
        self.values_11pct     = []
        self.values_50pct     = []
        self.values_89pct     = []
        self.val_values       = []
        self.val_values_11pct = []
        self.val_values_50pct = []
        self.val_values_89pct = []
        self.batch_offset     = 0

        ##  Initialise the bootstrap weights
        self.bootstrap_indices = None
        if num_bootstrap > 0 :
            num_data = len(data_output)
            self.bootstrap_indices = np.random.choice(num_data, size=(num_bootstrap, num_data))

        ##  Initialise bootstrap weights for validation data
        self.val_bootstrap_indices = None
        if num_bootstrap > 0 and validation_data is not None :
            num_data = len(validation_data[1])
            self.val_bootstrap_indices = np.random.choice(num_data, size=(num_bootstrap, num_data))

        
    def on_batch_end(self, batch_idx:int, logs:dict=None) :
        """
        Processing to be run at the end of each batch.
        With the given batch frequency, we pass self.data_input through the model and measure the loss.
        
        Inputs:
        
            >  batch_idx, int
               Index of the batch having just been processed
        """
        
        ##  Only proceed with the given batch frequency
        #if (batch_idx != 0) and ((batch_idx+1) % self.batch_frequency != 0) : return
        if (batch_idx == 0) or ((batch_idx+1) % self.batch_frequency != 0) : return
        
        ##  Store the batch index, using self.batch_offset to ensure continuation over epochs
        offset_batch_idx = batch_idx + self.batch_offset
        self.batch_indices.append(offset_batch_idx)

        ##  Update values
        self.update_values(offset_batch_idx, self.data_input, self.data_output, self.bootstrap_indices)
        if self.validation_data is not None :
            self.update_values(offset_batch_idx, self.validation_data[0], self.validation_data[1], self.val_bootstrap_indices, validation=True)

        ##  Plot if configured
        if batch_idx > 0 and self.plot_frequency > 0 and ((batch_idx+1) % self.plot_frequency == 0) :
            self.plot(show=True, close=True)

    
    def on_epoch_begin(self, epoch_idx:int, logs:dict=None) :
        """
        Processing to be run at the start of each epoch.
        Sets self.batch_offset equal to the number of batches already processed.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        
        ##  Set self.batch_offset to the number of batches already processed
        self.batch_offset = epoch_idx * self.num_steps
        self.epoch_starts.append(self.batch_offset)
    
    
    def on_epoch_end(self, epoch_idx:int, logs:dict=None) :
        """
        Processing to be run at the end of each epoch.
        Creates a plot of the loss curve if configured to do so
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        if self.plot_on_epoch_end : 
            self.plot(show=True, close=True)
    
    
    def on_train_begin(self, logs:dict=None) :
        """
        Processing to be run at the start of training.
        
        Re-initialises internal variables and storage containers.
        """
        ##  Initialise variables
        self.num_steps    = self.params["steps"]
        self.batch_offset = 0 

        ##  If no func provided then fallback to model loss function
        if type(self.func) == type(None) :
            self.func = self.model.loss
        
        ##  Initialise containers
        self.batch_indices    = []
        self.epoch_starts     = []
        self.values           = []
        self.values_11pct     = []
        self.values_50pct     = []
        self.values_89pct     = []
        self.val_values       = []
        self.val_values_11pct = []
        self.val_values_50pct = []
        self.val_values_89pct = []
    
    
    def on_train_end(self, logs:dict=None) :
        """
        Processing to be run at the end of training.
        Creates a plot of the loss curve if configured to do so
        """
        if not self.plot_on_train_end : return
        self.plot(show=True, close=True)


    def plot(self, show:bool=True, savefig:str=None, close:bool=True, dpi:int=200) :
        """
        Create plot showing the loss throughout training

        Inputs:

            >  show, bool, default=True
               Whether to call plt.show(fig) with the figure created

            >  savefig, str, default=None
               Optional filename to save the figure to

            >  close, bool, default=True
               Whether to call plt.close(fig) at the end

            >  dpi, int, default=200
               Pixels-per-inch when saving to file, for formats that require this

        Returns:

            >  fig, plt.Figure object, the figure created
        """
    
        ##  Create and format figure object
        fig = plt.figure(figsize=(8, 4.5))
        fig.subplots_adjust(hspace=0, wspace=0)

        ##  Create and format upper axes for linear y-axis
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.tick_params(axis="both", which="both", top=True, right=True, direction="in")
        ax1.grid(which="both")
        ax1.set_yscale(self.yscale)
        if self.yscale == "log" :
            ax1.set_ylabel(f"{self.label}\n[log]", ha="right", fontsize=14, labelpad=20, rotation=0)
        else :
            ax1.set_ylabel(f"{self.label}\n[linear]", ha="right", fontsize=14, labelpad=20, rotation=0)
        ax1.set_xlabel("Batch index", va="top", fontsize=14, labelpad=20)

        ##  Pull data as np arrays
        x    , y    , y_lo    , y_mid    , y_hi     = np.array(self.batch_indices), np.array(self.values    ), np.array(self.values_11pct    ), np.array(self.values_50pct    ), np.array(self.values_89pct    )
        val_x, val_y, val_y_lo, val_y_mid, val_y_hi = np.array(self.batch_indices), np.array(self.val_values), np.array(self.val_values_11pct), np.array(self.val_values_50pct), np.array(self.val_values_89pct)

        ##  Plot metric curve
        ax1.plot(x, y, "x-", lw=2.5, c="k", label="metric")

        ##  If we have calculated std then use to plot error band
        if len(y_lo) :
            ax1.fill_between(x, y_lo , y_mid, lw=0, fc="darkblue", alpha=0.3)
            ax1.fill_between(x, y_mid, y_hi , lw=0, fc="darkblue", alpha=0.3)

        ##  Plot validation curve
        if len(val_y) :
            ax1.plot(val_x, val_y, "X--", lw=1, c="darkred", label="validation")
            if len(y_lo) :
                ax1.fill_between(val_x, val_y_lo , val_y_mid, lw=0, fc="darkred", alpha=0.3)
                ax1.fill_between(val_x, val_y_mid, val_y_hi , lw=0, fc="darkred", alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), frameon=False, fontsize=11)
        
        ##  Plot vertical lines at epoch transitions
        for epoch_start in self.epoch_starts :
            ax1.axvline(epoch_start-0.5, ls="-", lw=2, c="k")

        ##  Save plot
        if savefig :
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)

        ##  Show plot
        if show :
            plt.show(fig)

        ##  Close plot
        if close :
            plt.close(fig)


    def update_values(self, batch_idx, x, y, bootstrap_indices=None, validation:bool=False) :
        """
        """
        
        ##  Calculate + store the loss
        y_pred = self.model(x, training=False)
        if type(y_pred) in [list, set, tuple] :
            y_pred = y_pred[0]
        batch_vals = self.func(y=y, y_pred=y_pred).numpy()
        if len(batch_vals.shape) == 2 and batch_vals.shape[-1] == 1 : batch_vals = batch_vals[:,0]
        value = np.mean(batch_vals)
        if validation :
            self.val_values.append(value)
        else :
            self.values.append(value)

        ##  Find std dev on loss using bootstraps if configured
        v11, v89 = None, None
        if bootstrap_indices is not None : 

            ##  Do bootstraps
            bs_vals = []
            for indcs in bootstrap_indices :
                x_bs       = [tf.gather(xp, indcs) for xp in x] if type(x) is list else tf.gather(x, indcs)
                y_bs       = tf.gather(y, indcs)
                y_pred_bs  = self.model(x_bs, training=False)
                if type(y_pred_bs) in [list, set, tuple] :
                    y_pred_bs = y_pred_bs[0]
                batch_vals = self.func(y=y_bs, y_pred=y_pred_bs).numpy()
                if len(batch_vals.shape) == 2 and batch_vals.shape[-1] == 1 : batch_vals = batch_vals[:,0]
                bs_vals.append(np.mean(batch_vals))

            ##  Store std dev
            v11, v50, v89 = np.percentile(bs_vals, [11, 50, 89])
            if validation :
                self.val_values_11pct.append(v11)
                self.val_values_50pct.append(v50)
                self.val_values_89pct.append(v89)
            else :
                self.values_11pct.append(v11)
                self.values_50pct.append(v50)
                self.values_89pct.append(v89)

        ##  Log if configured
        if self.logger :
            flat_label = self.label.replace('\n',' ') + (' (validation)' if validation else '')
            self.logger.log(self.log_lvl, f"Metric {flat_label} after {batch_idx} batches is {value:.5}{f' [68% @ {v11:.5} - {v89:.5}]' if v11 else ''}")



##====================================##
##   PositionalEncoding keras layer   ##
##====================================##
##
class PositionalEncoding(CustomLayer) :

    def __init__(self, num_freqs:int, slice_index:int=None, min_period:float=5, max_period:float=1e5, base:float=np.e, learnable:bool=False, **kwargs) :
        '''
        class PositionalEncoding

        A keras layer for calculating positional encodings for an input Tensor x.
        x has shape [B, S] or [B, S, N]
           B is batch size
           S is sequence length
           N is num features
           F is num frequencies
        If a third index exists then x must contain the token position index at feature index self.slice_index
        After applying both cos and sin encodings, output vector is of length 2*num_freqs 

        Inputs:

            >  num_freqs, int
               Number of frequencies to generate encodings for

            >  slice_index, int, default=None
               Index of the input feature vector corresponding to integer position index, if None then encode everything

            >  min_period, int, default=5
               Period of oscillation for the lowest frequency encoding

            >  max_period, int, default=1e5
               Period of oscillation for the highest frequency encoding

            >  base, float, default=np.e
               Base of the geometric series of num_freqs frequencies between 2pi/max_period and 2pi/min_period

            >  learnable, bool, default=False
               Whether to initialise the frequencies as trainable parameters
        '''

        ##  Base class contructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.slice_index = slice_index
        self.num_freqs   = num_freqs
        self.min_period  = min_period
        self.max_period  = max_period
        self.base        = base
        self.learnable   = learnable

        ##  Create keras sublayers
        self.initialise_frequencies()

    
    def call(self, x) :
        '''
        Calculate positional encodings for Tensor x.
        x has shape [B, S, N]
           B is batch size
           S is sequence length
           N is num features
           F is num frequencies
        x must contain the token position index at feature index self.slice_index
        '''
        return self.encode(x)


    def compute_mask(self, x, mask=None):
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask
        

    def encode(self, x) :
        '''
        Calculate positional encodings for Tensor x.
        x has shape [B, S, N]
           B is batch size
           S is sequence length
           N is num features
           F is num frequencies
        x must contain the token position index at feature index self.slice_index
        '''
        if type(self.slice_index) == type(None) : 
            x  = x[:, :, tf.newaxis]                    # Shape [B, S, 1]
        else : 
            x  = x[:, :, self.slice_index, tf.newaxis]  # Shape [B, S, 1]
        x  = tf.cast(x, self.dtype)
        cx = tf.math.cos(tf.matmul(x, self.freqs))     # [B, S, 1] * [1, F] --> Shape [B, S, F]
        sx = tf.math.sin(tf.matmul(x, self.freqs))     # [B, S, 1] * [1, F] --> Shape [B, S, F]
        #return tf.concat([tf.matmul(x, self.freqs), tf.matmul(x, self.freqs)], axis=-1)             # Shape [B, S, 2F]
        return tf.concat([cx, sx], axis=-1)             # Shape [B, S, 2F]
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain PositionalEncoding layers. 
        This dict stores all values we need to pass to __init__ to create a PositionalEncoding layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "slice_index" : self.slice_index, 
                "num_freqs"   : self.num_freqs, 
                "min_period"  : self.min_period,
                "max_period"  : self.max_period, 
                "base"        : self.base, 
                "learnable"   : self.learnable, 
            })
        return config


    def initialise_frequencies(self) :
        '''
        Create np array and keras Tensor storing the frequencies used to calculate positional encodings
        '''

        ##  Create constant array of frequencies following a log series between 2pi/max_period and 2pi/min_period
        ##  -  array has shape (1, self.num_freqs) to enable correct broadcasting through matrix multiplication
        ##  -  store copy as Tensor object
        ##  Set the scale factor
        freqs_np   = np.logspace(np.log(two_pi/self.max_period), np.log(two_pi/self.min_period), self.num_freqs, base=self.base, dtype=np.float64)
        freqs_np   = freqs_np.reshape((1, self.num_freqs))
        self.freqs = self.add_weight(
                        f"{self.name}_frequencies", 
                        initializer = tf.keras.initializers.Constant(value=tf.constant(freqs_np, dtype=self.dtype)),
                        shape       = freqs_np.shape,
                        trainable   = True,
                        dtype       = self.dtype,
                    )



##================================##
##   ReduceSequence keras layer   ##
##================================##
##
class ReduceSequence(CustomLayer) :
    
    def __init__(self, SUM:bool=True, MEAN:bool=True, STD:bool=True, **kwargs) :
        """
        class ReduceSequence

        A keras layer for reducing a sequence along the 1 axis, ignoring masked values.
        Input is expected to have shape [B, S, F] where B is batch size, S is sequence length, F are features.
        If multiple reductions are chosen, their results will be concatenated.
        Sequence is reduced along the S axis, giving output shape [B, N*F] where N is num reductions chosen.

        Inputs:
               
            >  SUM, bool, default=True
               Whether to implement reduction by sum
               
            >  MEAN, bool, default=True
               Whether to implement reduction by mean
               
            >  STD, bool, default=True
               Whether to implement reduction by standard deviation
        """
        
        ##  Base class constructor
        super().__init__(**kwargs)
        
        ##  Make sure at least one reduction has been requested
        if not (SUM or MEAN or STD) :
            raise ValueError("At least one reduction keyword must be True")
        
        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.SUM  = SUM
        self.MEAN = MEAN
        self.STD  = STD
        
        ##  Create keras layers
        base_name    = self.name
        self._concat = Concatenate(name=f"{base_name}_concat")
        
    
    def call(self, x, mask, training:bool=False) :
        """
        Reduce a tensor along axis a according to the reduction schemes chosen.
        If multiple reduction schemes then the results are concatenated.
        Input is expected to have shape [B, S, F] where B is batch size, S is sequence length, F are features.
        """

        ##  Make sure input has correct dimensions - should be [B, S, F]
        if len(x.shape) != 3 : raise ValueError(f"Expected x.shape to have length 3 but shape {x.shape} provided")

        ##  Make sure mask has correct dimensions - should be [B, S] where B and S match those of x
        if len(mask.shape) != 2 or mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1] :
            raise ValueError(f"Expected mask of shape {x.shape[:2]} but shape {mask.shape} provided")
        
        ##  Store list of reductions as they are calculated
        reductions = []
        
        ##  Create a copy of the mask with shape that allows broadcasting with x
        shaped_mask = tf.cast(mask, dtype=x.dtype)[:,:,tf.newaxis]     # Shapes [B, S] --> [B, S, 1]
        
        ##  Multiple x by the mask to force masked entries to zero
        masked_x = x * shaped_mask     # Shapes [B, S, F] * [B, S, 1] --> [B, S, F]
        
        ##  Sum along reduction axis - we have zeroed the masked entries so they do not contribute
        masked_sum = tf.keras.backend.sum(masked_x, axis=1)     # Shapes [B, S, F] --> [B, F]
        
        ##  If sum is a required reduction then append it to the list
        if self.SUM : reductions.append(masked_sum)
        
        ##  Calculate the number of un-masked entries in each sequence
        masked_num = tf.keras.backend.sum(shaped_mask, axis=1)     # Shapes [B, S, 1] --> [B, 1]
        
        ##  Calculate mean of un-masked entries
        masked_mean = masked_sum / masked_num     # Shapes [B, F] / [B, 1] --> [B, F]
        
        ##  If mean is a required reduction then append it to the list
        if self.MEAN : reductions.append(masked_mean)
            
        ##  Only calculate STD if it is a required reduction
        if self.STD :
            
            ##  Calculate residuals wrt masked mean
            res = x - masked_mean[:,tf.newaxis,:]     # Shapes [B, S, F] - [B, 1, F] --> [B, S, F]
            
            ##  Set residuals of masked entries to zero
            masked_res = res * shaped_mask     # Shapes [B, S, F] * [B, S, 1] --> [B, S, F]
        
            ##  Square the masked residuals
            masked_res_sq = tf.math.square(masked_res)     # Shapes [B, S, F] * [B, S, F] --> [B, S, F]
        
            ##  Calculate the variance by averaging the masked squared residuals
            masked_variance = tf.keras.backend.sum(masked_res_sq, axis=1) / masked_num     # Shapes [B, F] / [B, F] --> [B, F]
        
            ##  Calculate std as the sqrt of the variance
            masked_std = tf.math.sqrt(masked_variance)
            
            ##  Append std to the list
            reductions.append(masked_std)
        
        ##  Return a concatenated list of the reductions
        y = self._concat(reductions)     #  Shapes N x [B, F] --> [B, N*F]
        return y
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain ReduceSequence layers. 
        This dict stores all values we need to pass to __init__ to create a ReduceSequence layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "SUM"  : self.SUM, 
                "MEAN" : self.MEAN, 
                "STD"  : self.STD, 
            })
        return config



##============================##
##   RightSlice keras layer   ##
##============================##
##
class RightSlice(CustomLayer) :
    
    def __init__(self, slice_index:int, newaxis:bool=False, **kwargs) :
        """
        class RightSlice
        
        Pull the given slice_index from the right-most index of a tensor
        
        Inputs:
        
            >  slice_index, int
               Index to be extracted from right-most index
        
            >  newaxis, bool, default=False
               If True then use tf.newaxis to ensure output has the same numer of indices as input
        """
        
        ##  Base class constructor
        super().__init__(**kwargs)
        
        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.slice_index = slice_index
        self.newaxis     = newaxis
        

    def call(self, x, training=False, mask=None) :
        """
        Return the configured element of the right-most index of x
        Note that indexing with x[index] uses the tf.slice operation behind the scenes.
        Slicing with ... fills in all intermediate indices.
        """
        
        ##  If self.newaxis then use tf.newaxis to prevent the tensor shape from losing indices
        if self.newaxis :
            x[..., self.slice_index, tf.newaxis]
            
        ##  Otherwise just
        return x[..., self.slice_index]


    def compute_mask(self, x, mask=None) :
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask
    

    def get_config(self) :
        """
        Create the config dict needed to save models that contain RightSlice layers. 
        This dict stores all values we need to pass to __init__ to create a RightSlice layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "slice_index" : self.slice_index, 
                "newaxis"     : self.newaxis,
            })
        return config
    
        


##=========================##
##   Scaling keras layer   ##
##=========================##
##
class Scaling(CustomLayer) :
    
    def __init__(self, scale:float=1., trainable:bool=True, **kwargs) :
        '''
        class Scaling

        A keras layer for scaling a tensor by a trainable parameter

        Inputs:

            >  scale, float, default=1.
               Initial value for the learnable scale factor
        '''
        ##  Base class constructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.scale = scale

        ##  Set the scale factor
        self.s = self.add_weight(
                    f"{self.name}_scale", 
                    initializer = tf.keras.initializers.Constant(value=scale),
                    trainable   = True,
                    dtype       = self.dtype,
                )
    

    def call(self, x) :
        '''
        Scale Tensor x by a learnable scale factor
        '''
        return tf.math.scalar_mul(self.s, x)


    def compute_mask(self, x, mask=None):
        """
        Propagate a keras mask through the layer. Mask is a Tensor of bools with either the same shape as x, or one fewer indices.
        """
        return mask

    
    def get_config(self) :
        """
        Create the config dict needed to save models that contain Scaling layers. 
        This dict stores all values we need to pass to __init__ to create a Scaling layer with the same configuration.
        """
        config = super().get_config()
        config.update(
            {
                "scale" : self.scale, 
            })
        return config


