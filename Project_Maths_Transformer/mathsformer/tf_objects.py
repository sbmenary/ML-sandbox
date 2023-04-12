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

from collections import ChainMap

from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers    import (Add, Average, BatchNormalization, Concatenate, Dense, Dot, Dropout, Embedding, Input, Lambda, Layer, 
                                        LayerNormalization, Masking, MultiHeadAttention, Rescaling)
from tensorflow.keras.models    import Model, Sequential
from tensorflow.keras.losses    import Loss, SparseCategoricalCrossentropy



##=============##
##   Globals   ##
##=============##

##  Numerical constants
pi, two_pi = math.pi, 2*math.pi

##  An instance of sparse categorical cross-entropy loss
sparse_categorical_crossentropy_loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')




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



def get_nested_sublayers(layer, layers=[]) :
    """
    Return a list of all the nested sublayers found within layer.
    
    Searches for sublayers/submodels within the object attributes.
    Additionally searches for a list called layer.layers if one exists.
    Search is recursive to pick up all nested sublayers.
    
    Inputs:
    
        >  layer, any type but often keras.Model or keras.Layer
           An object containing nested sublayers / submodels to be searched for.
    """
        
    ##  Check for model.layers
    if hasattr(layer, "layers") and type(layer.layers) is list :
        for sublayer in layer.layers :
            if not (isinstance(sublayer, Layer) or isinstance(sublayer, Model)) : continue
            if sublayer.name in [l.name for l in layers] : continue
            layers.append(sublayer)
            get_nested_sublayers(sublayer, layers)
    
    ##  Check for other attributes that are instances of Layer or Model
    for attr_name, attr in layer.__dict__.items() :
        if not (isinstance(attr, Layer) or isinstance(attr, Model)) : continue
        if attr.name in [l.name for l in layers] : continue
        layers.append(attr)
        get_nested_sublayers(attr, layers)

    ##  Return container
    return layers


def masked_accuracy(y, y_pred, mask_value=0) :
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


def masked_sparse_categorical_crossentropy(y, y_pred, mask_value:int=0, weight_seq_by_length:bool=False) :
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
        
    ##  Calculate the loss for every token, including masked tokens
    loss = sparse_categorical_crossentropy_loss(y, y_pred)
    
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



##================================##
##   AttentionBlock keras layer   ##
##================================##
##
class AttentionBlock(CustomLayer) :

    
    def __init__(self, num_heads:int, ndim_hidden:int, ndim_out:int, dropout:float=0, self_attention:bool=False, use_causal_mask:bool=False, skip_connect:bool=True, layer_norm:bool=True, _mha:MultiHeadAttention=None, **kwargs) :
        '''
        class AttentionBlock

        A keras layer for applying multi-head attention.

        Inputs:

            >  num_heads, int
               Number of heads

            >  ndim_hidden, int
               Number of neurons in the query / key dimension being contracted over

            >  ndim_out, int
               Number of neurons in the output feature vector

            >  dropout, float, default=0
               Dropout rate

            >  self_attention, bool, default=False
               If True then only provide one sequence which will be used for both query and key

            >  use_causal_mask, bool, default=False
               If True then only attend to previous elements in the sequence

            >  skip_connect , bool, default=True
               Whether to use a skip connection

            >  layer_norm , bool, default=True
               Whether to use a layer normalisation

            >  _mha, keras MultiHeadAttention layer, default=None
               Pre-existing MultiHeadAttention layer, perhaps deserialised after loading from file
        '''

        ##  Base class constructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.num_heads       = num_heads
        self.ndim_hidden     = ndim_hidden
        self.ndim_out        = ndim_out
        self.dropout         = dropout
        self.self_attention  = self_attention
        self.use_causal_mask = use_causal_mask
        self.skip_connect    = skip_connect
        self.layer_norm      = layer_norm

        ##  Create keras layers
        base_name = self.name
        if _mha : self._mha = _mha
        else    : self._mha = MultiHeadAttention(name=f"{base_name}_mha", num_heads=self.num_heads, key_dim=self.ndim_hidden, value_dim=self.ndim_out, dropout=self.dropout, dtype=self.dtype)
        self._average   = Average           (name=f"{base_name}_average"   , dtype=self.dtype) if self.skip_connect else None
        self._layernorm = LayerNormalization(name=f"{base_name}_layer_norm", dtype=self.dtype) if self.layer_norm   else None
        

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
            q, v = x, x
        else : 
            if type(x) != list :
                raise TypeError(f"Not configured for self-attention, therefore expect input to be a list of two Tensors (query and reference)")
            q, v = x[0], x[1]

        ##  Execute attention, skip-connection and layer-normalisation
        y = self._mha(query=q, value=v, use_causal_mask=self.use_causal_mask, training=training)
        if self.skip_connect : y = self._average([q, y])
        if self.layer_norm   : y = self._layernorm(y, training=training)
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
                "num_heads"       : self.num_heads, 
                "ndim_hidden"     : self.ndim_hidden, 
                "ndim_out"        : self.ndim_out, 
                "dropout"         : self.dropout, 
                "self_attention"  : self.self_attention, 
                "use_causal_mask" : self.use_causal_mask, 
                "skip_connect"    : self.skip_connect, 
                "layer_norm"      : self.layer_norm, 
                "_mha"            : tf.keras.layers.serialize(self._mha),
            })
        return config


    @classmethod
    def get_custom_layer_types(cls) :
        """
        Return a set of custom layers used by this class.
        """
        return (MultiHeadAttention,)




##==============================##
##   DecoderBlock keras layer   ##
##==============================##
##
class DecoderBlock(CustomLayer) :


    def __init__(self, ndim_out:int, num_heads:int, ndim_hidden_mha:int, ndim_hidden_ff:int, num_hidden_layers_ff:int=1, dropout_mha:float=0, dropout_ff:float=0, skip_connect:bool=True, 
                 layer_norm:bool=True, use_causal_mask:bool=True, activation:str="relu", _self_att_block:AttentionBlock=None, _cross_att_block:AttentionBlock=None, **kwargs) :
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

            >  layer_norm, bool, default=True
               Whether to use layer normalisation in the attention and feed-forward blocks

            >  use_causal_mask, bool, default=True
               Whether to apply a causal mask in the multi-head attention block

            >  activation, str, default="relu"
               Activation function for non-linear layers

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
        self.layer_norm           = layer_norm
        self.use_causal_mask      = use_causal_mask
        self.activation           = activation
        
        ##  Create keras layers
        base_name = self.name
        if   _self_att_block  : self._self_att_block  = _self_att_block
        else                  : self._self_att_block  = AttentionBlock(name=f"{base_name}_self_attention_block" , ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=use_causal_mask, self_attention=True)
        if   _cross_att_block : self._cross_att_block = _cross_att_block
        else                  : self._cross_att_block = AttentionBlock(name=f"{base_name}_cross_attention_block", ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=False, self_attention=False)
        self._ff_block_1 = FeedForwardBlock(name=f"{base_name}_feedfwd_block_1", ndim_out=ndim_out, ndim_hidden=ndim_hidden_ff, dropout=dropout_ff, skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, batch_norm=False, activation=activation, num_hidden_layers=num_hidden_layers_ff)
        self._ff_block_2 = FeedForwardBlock(name=f"{base_name}_feedfwd_block_2", ndim_out=ndim_out, ndim_hidden=ndim_hidden_ff, dropout=dropout_ff, skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, batch_norm=False, activation=activation, num_hidden_layers=num_hidden_layers_ff)
                    
        
    def call(self, x, training=False, mask=None) :
        """
        Apply a multi-head self-attention and feed-forward block to a sequence.
        """
        seq_dec , seq_enc  = x[0]   , x[1]
        mask_dec, mask_enc = mask[0], mask[1]
        seq_dec = self._self_att_block (seq_dec, training=training)                  # N.B uses internal keras_mask so no mask argument needed
        seq_dec = self._ff_block_1     (seq_dec, mask=mask_dec, training=training)   
        seq_dec = self._cross_att_block([seq_dec, seq_enc], training=training)       # N.B uses internal keras_mask so no mask argument needed
        seq_dec = self._ff_block_2     (seq_dec, mask=mask_dec, training=training)   
        return seq_dec


    def compute_mask(self, *args, **kwargs) :
        """
        Compute the mask generated by the multi-head attention layer
        """
        return self._cross_att_block.compute_mask(*args, **kwargs)
    

    @classmethod
    def from_config(cls, config) :
        """
        Create a new EncoderBlock layer from a dictionary generated by the get_config() method.
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
                "layer_norm"           : self.layer_norm, 
                "use_causal_mask"      : self.use_causal_mask, 
                "activation"           : self.activation,
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


    def __init__(self, ndim_out:int, num_heads:int, ndim_hidden_mha:int, ndim_hidden_ff:int, num_hidden_layers_ff:int=1, dropout_mha:float=0, dropout_ff:float=0, skip_connect:bool=True, layer_norm:bool=True, use_causal_mask:bool=False, activation:str="relu", _att_block:AttentionBlock=None, **kwargs) :
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

            >  layer_norm, bool, default=True
               Whether to use layer normalisation in both the multi-head attention and feed-forward blocks

            >  use_causal_mask, bool, default=False
               Whether to apply a causal mask in the multi-head attention block

            >  activation, str, default="relu"
               Activation function for non-linear layers

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
        self.layer_norm           = layer_norm
        self.use_causal_mask      = use_causal_mask
        self.activation           = activation
        
        ##  Create keras layers
        base_name = self.name
        if _att_block : 
            self._att_block = _att_block
        else : 
            self._att_block = AttentionBlock  (name=f"{base_name}_attention_block", ndim_out=ndim_out, ndim_hidden=ndim_hidden_mha, dropout=dropout_mha, skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, num_heads=num_heads, use_causal_mask=use_causal_mask, self_attention=True)
        self._ff_block      = FeedForwardBlock(name=f"{base_name}_feedfwd_block"  , ndim_out=ndim_out, ndim_hidden=ndim_hidden_ff , dropout=dropout_ff , skip_connect=skip_connect, layer_norm=layer_norm, dtype=self.dtype, batch_norm=False, activation=activation, num_hidden_layers=num_hidden_layers_ff)
                    

        
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
                "layer_norm"           : self.layer_norm, 
                "use_causal_mask"      : self.use_causal_mask, 
                "activation"           : self.activation,
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
class Enumerate(Layer) :

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

    def __init__(self, ndim_out:int, ndim_hidden:int, num_hidden_layers:int=1, dropout:float=0, activation='relu', skip_connect:bool=False, layer_norm:bool=False, batch_norm:bool=True, **kwargs) :
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

            >  activation, str, default='relu'
               Activation function for the hidden layers

            >  skip_connect, bool, default=False
               Whether to apply skip connection between the input and output; only possible when ndim_out is equal to the input size, or
               when they can be broadcast to the same size (warning: check that output size is what you expect!)

            >  layer_norm, bool, default=False
               Whether to apply layer normalisation after the skip connection

            >  batch_norm, bool, default=True
               Whether to apply batch normalisation after the hidden afters
        '''

        ##  Base class contructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.ndim_out          = ndim_out
        self.ndim_hidden       = ndim_hidden
        self.num_hidden_layers = num_hidden_layers
        self.dropout           = dropout
        self.activation        = activation
        self.skip_connect      = skip_connect
        self.layer_norm        = layer_norm
        self.batch_norm        = batch_norm

        ##  Create keras sublayers
        self.initialise_layers()


    def call(self, x, training:bool=None, mask=None) :
        '''
        Operate this feed forward block on Tensor object x. The training flag is passed to all sublayers.
        '''
        y = x
        y = self._dense_block(y, training=training)
        y = self._dense_out  (y, training=training)
        if self.skip_connect : y = self._average   ([x, y], training=training)
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
                "ndim_out"          : self.ndim_out, 
                "ndim_hidden"       : self.ndim_hidden, 
                "num_hidden_layers" : self.num_hidden_layers,
                "dropout"           : self.dropout, 
                "activation"        : self.activation, 
                "skip_connect"      : self.skip_connect,
                "layer_norm"        : self.layer_norm,
                "batch_norm"        : self.batch_norm,
            })
        return config


    def initialise_layers(self) :
        '''
        Create new sublayer objects using their default initialisers
        '''
        dtype, base_name, dense_block_layers = self.dtype, self.name, []
        for l_idx in range(self.num_hidden_layers) :
            dense_block_layers.append(Dense(self.ndim_hidden, dtype=dtype, name=f"{base_name}_dense_hidden_{l_idx+1}", activation=self.activation))
            if self.batch_norm : 
                dense_block_layers.append(BatchNormalization(dtype=dtype, name=f"{base_name}_batch_norm_{l_idx+1}"))
            if self.layer_norm  :
                dense_block_layers.append(LayerNormalization(dtype=dtype, name=f"{base_name}_layer_norm_{l_idx+1}"))
            if self.dropout > 0 :
                dense_block_layers.append(Dropout(self.dropout, dtype=dtype, name=f"{base_name}_dropout_{l_idx+1}"))
        self._dense_block  = Sequential(dense_block_layers, name=f"{base_name}_dense_block")
        self._dense_out    = Dense     (self.ndim_out, dtype=dtype, name=f"{base_name}_dense_out", activation='linear')
        self._average      = Average   (dtype=dtype, name=f"{base_name}_average") if self.skip_connect else None



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
        self.batch_offset  = -1
        self.num_steps     = -1
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
        
        
    def on_batch_end(self, batch_idx:int, logs=None) :
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
    
    
    def on_epoch_begin(self, epoch_idx:int, logs=None) :
        """
        Processing to be run at the start of each epoch.
        Sets self.batch_offset equal to the number of batches already processed.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        
        ##  Set self.batch_offset to the number of batches already processed
        self.batch_offset = epoch_idx * self.num_steps
    
    
    def on_train_begin(self, logs=None) :
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
               Batch frequency with which to measure layer activations
               
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
        
        
    def on_batch_end(self, batch_idx:int, logs=None) :
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
            self.layer_means[layer.name].append(np.mean(weights))
            self.layer_stds [layer.name].append(np.std (weights))
    
    
    def on_epoch_begin(self, epoch_idx:int, logs=None) :
        """
        Processing to be run at the start of each epoch.
        Sets self.batch_offset equal to the number of batches already processed.
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        
        ##  Set self.batch_offset to the number of batches already processed
        self.batch_offset = epoch_idx * self.num_steps
    
    
    def on_train_begin(self, logs=None) :
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
            self.layer_means[layer.name] = []
            self.layer_stds [layer.name] = []


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

        ##  Get names of all layers for which weights have been recorded, an alphabetical order
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

            ##  Plot line tracking the layer mean weight, and shade region between std devs
            ax.plot(x, y, "-", lw=3, c='k')
            ax.fill_between(x, y-ey, y+ey, fc="darkblue", alpha=0.2, lw=0)

            ##  Draw text label on the first axis only
            if layer_idx == 0 :
                ax.text(0, 1.2, "Layer weights vs batch index", weight="bold", ha="left", 
                        va="bottom", fontsize=16, transform=ax.transAxes)


        ##  Save plot
        if savefig :
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)

        ##  Show plot
        if show :
            plt.show(fig)

        ##  Close plot
        if close :
            plt.close(fig)
        


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
        w = tf.keras.activations.softmax(self._weights)
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
    
    
    def on_epoch_end(self, epoch_idx:int, logs={}) :
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



##=================================##
##   MetricRecord keras callback   ##
##=================================##
##
class MetricRecord(Callback) :
    
    def __init__(self, batch_frequency:int, val_input, val_output, label:str="Partial\nval. loss", func=None, num_bootstrap:int=-1, 
                 plot_on_train_end:bool=False, plot_on_epoch_end:bool=False, plot_frequency:int=-1, yscale:str="log", 
                 logger=None, log_lvl:int=logging.DEBUG) :
        """
        class MetricRecord
        
        Tracks the value of a scale function over a validation dataset during training
        
        Inputs:
            
            >  batch_frequency, int
               Batch frequency with which to measure layer activations
               
            >  val_input, Tensor
               Validation datapoints
               
            >  val_output, Tensor
               True datapoint labels

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
        self.batch_frequency    = batch_frequency
        self.val_input          = val_input
        self.val_output         = val_output
        self.label              = label
        self.func               = func
        self.num_bootstrap      = num_bootstrap
        self.plot_on_train_end  = plot_on_train_end
        self.plot_on_epoch_end  = plot_on_epoch_end
        self.plot_frequency     = plot_frequency
        self.yscale             = yscale
        self.logger             = logger
        self.log_lvl            = log_lvl
        
        ##  Initialise containers and variables
        self.batch_indices = []
        self.epoch_starts  = []
        self.values        = []
        self.values_11pct  = []
        self.values_50pct  = []
        self.values_89pct  = []
        self.batch_offset  = 0

        ##  Initialise the bootstrap weights
        self.bootstrap_weights = None
        if num_bootstrap > 0 :
            num_data = len(val_output)
            self.bootstrap_indices = np.random.choice(num_data, size=(num_bootstrap, num_data))

        
    def on_batch_end(self, batch_idx:int, logs=None) :
        """
        Processing to be run at the end of each batch.
        With the given batch frequency, we pass self.val_input through the model and measure the loss.
        
        Inputs:
        
            >  batch_idx, int
               Index of the batch having just been processed
        """
        
        ##  Only proceed with the given batch frequency
        #if (batch_idx != 0) and ((batch_idx+1) % self.batch_frequency != 0) : return
        if (batch_idx == 0) or ((batch_idx+1) % self.batch_frequency != 0) : return
        
        ##  Store the batch index, using self.batch_offset to ensure continuation over epochs
        batch_idx += self.batch_offset
        self.batch_indices.append(batch_idx)
        
        ##  Calculate + store the loss
        x = self.val_input
        y, y_pred  = self.val_output, self.model(x, training=False)
        batch_vals = self.func(y=y, y_pred=y_pred).numpy()
        if len(batch_vals.shape) == 2 and batch_vals.shape[-1] == 1 : batch_vals = batch_vals[:,0]
        value = np.mean(batch_vals)
        self.values.append(value)

        ##  Find std dev on loss using bootstraps if configured
        v11, v89 = None, None
        if self.num_bootstrap > 0 : 

            ##  Do bootstraps
            bs_vals = []
            for bootstrap_indices in self.bootstrap_indices :
                x       = [tf.gather(xp, bootstrap_indices) for xp in self.val_input] if type(self.val_input) is list else tf.gather(self.val_input, bootstrap_indices)
                y       = tf.gather(self.val_output, bootstrap_indices)
                y_pred  = self.model(x, training=False)
                batch_vals = self.func(y=y, y_pred=y_pred).numpy()
                if len(batch_vals.shape) == 2 and batch_vals.shape[-1] == 1 : batch_vals = batch_vals[:,0]
                bs_vals.append(np.mean(batch_vals))

            ##  Store std dev
            v11, v50, v89 = np.percentile(bs_vals, [11, 50, 89])
            self.values_11pct.append(v11)
            self.values_50pct.append(v50)
            self.values_89pct.append(v89)

        ##  Log if configured
        if self.logger :
            flat_label = self.label.replace('\n',' ')
            self.logger.log(self.log_lvl, f"Metric {flat_label} after {batch_idx} batches is {value:.5}{f' [68% @ {v11:.5} - {v89:.5}]' if v11 else ''}")

        ##  Plot if configured
        if batch_idx > 0 and self.plot_frequency > 0 and ((batch_idx+1) % self.plot_frequency == 0) :
            self.plot(show=True, close=True)

    
    def on_epoch_begin(self, epoch_idx:int, logs=None) :
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
    
    
    def on_epoch_end(self, epoch_idx:int, logs={}) :
        """
        Processing to be run at the end of each epoch.
        Creates a plot of the loss curve if configured to do so
        
        Inputs:
        
            >  epoch_idx, int
               Index of the epoch about to be processed
        """
        if self.plot_on_epoch_end : 
            self.plot(show=True, close=True)
    
    
    def on_train_begin(self, logs=None) :
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
        self.batch_indices = []
        self.epoch_starts  = []
        self.values        = []
        self.values_11pct  = []
        self.values_89pct  = []
    
    
    def on_train_end(self, logs=None) :
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
        fig = plt.figure(figsize=(8, 4))
        fig.subplots_adjust(hspace=0.05, wspace=0.3)

        ##  Create and format upper axes for linear y-axis
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.tick_params(axis="both", which="both", top=True, right=True, direction="in")
        ax1.grid(which="both")
        #ax1.xaxis.set_ticklabels([])
        ax1.set_yscale(self.yscale)
        if self.yscale == "log" :
            ax1.set_ylabel(f"{self.label}\n[log]", ha="right", fontsize=14, labelpad=20, rotation=0)
        else :
            ax1.set_ylabel(f"{self.label}\n[linear]", ha="right", fontsize=14, labelpad=20, rotation=0)
        ax1.set_xlabel("Batch index", va="top", fontsize=14, labelpad=20)

        ##  Create and format lower axes for log y-axis
        #ax2 = fig.add_subplot(2, 1, 2)
        #ax2.tick_params(axis="both", which="both", top=True, right=True, direction="in")
        #ax2.set_yscale("log")
        #ax2.grid(which="both")

        #ax2.set_ylabel(f"{self.label}\n[log]", ha="right", fontsize=14, labelpad=20, rotation=0)
        #ax2.set_xlabel("Batch index", va="top", fontsize=14, labelpad=20)

        ##  Pull data as np arrays
        x, y, y_lo, y_mid, y_hi = self.batch_indices, self.values, self.values_11pct, self.values_50pct, self.values_89pct
        x, y, y_lo, y_mid, y_hi = np.array(x), np.array(y), np.array(y_lo), np.array(y_mid), np.array(y_hi)

        ##  Plot loss curves
        ax1.plot(x, y, "x-", lw=2, c="k")
        #ax2.plot(x, y, "x-", lw=2, c="k")

        ##  If we have calculated std then use to plot error band
        if len(y_lo) :
            ax1.fill_between(x, y_lo , y_mid, lw=0, fc="darkblue", alpha=0.3)
            ax1.fill_between(x, y_mid, y_hi , lw=0, fc="darkred" , alpha=0.3)
            #ax2.fill_between(x, y_lo , y_mid, lw=0, fc="darkblue", alpha=0.3)
            #ax2.fill_between(x, y_mid, y_hi , lw=0, fc="darkred" , alpha=0.3)
        
        ##  Plot vertical lines at epoch transitions
        for epoch_start in self.epoch_starts :
            ax1.axvline(epoch_start-0.5, ls="-", lw=2, c="k")
            #ax2.axvline(epoch_start-0.5, ls="-", lw=2, c="k")

        ##  Save plot
        if savefig :
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)

        ##  Show plot
        if show :
            plt.show(fig)

        ##  Close plot
        if close :
            plt.close(fig)



##====================================##
##   PositionalEncoding keras layer   ##
##====================================##
##
class PositionalEncoding(CustomLayer) :

    def __init__(self, num_freqs:int, slice_index:int=None, min_period:float=5, max_period:float=1e5, base:float=np.e, **kwargs) :
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
        '''

        ##  Base class contructor
        super().__init__(**kwargs)

        ##  Store all arguments provided to __init__, as these will be needed to implement model saving through the get_config() method
        self.slice_index = slice_index
        self.num_freqs   = num_freqs
        self.min_period  = min_period
        self.max_period  = max_period
        self.base        = base

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
        cx = tf.math.cos(tf.matmul(x, self._freqs))     # [B, S, 1] * [1, F] --> Shape [B, S, F]
        sx = tf.math.sin(tf.matmul(x, self._freqs))     # [B, S, 1] * [1, F] --> Shape [B, S, F]
        return tf.concat([cx, sx], axis=-1)             # Shape [B, S, 2F]
        

    def encode_np(self, x) :
        '''
        Calculate positional encodings for numpy array x.
        x has shape [B, S, N]
           B is batch size
           S is sequence length
           N is num features
           F is num frequencies
        x must contain the token position index at feature index self.slice_index
        '''
        if type(self.slice_index) == type(None) : 
            x  = x[:, :, np.newaxis]                    # Shape [B, S, 1]
        else : 
            x  = x[:, :, self.slice_index, np.newaxis]  # Shape [B, S, 1]
        cx = np.cos(np.matmul(x, self._freqs_np))       # [B, S, 1] * [1, F] --> Shape [B, S, F]
        sx = np.sin(np.matmul(x, self._freqs_np))       # [B, S, 1] * [1, F] --> Shape [B, S, F]
        return np.concat([cx, sx], axis=-1)             # Shape [B, S, 2F]
    

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
            })
        return config


    def initialise_frequencies(self) :
        '''
        Create np array and keras Tensor storing the frequencies used to calculate positional encodings
        '''

        ##  Create constant array of frequencies following a log series between 2pi/max_period and 2pi/min_period
        ##  -  array has shape (1, self.num_freqs) to enable correct broadcasting through matrix multiplication
        ##  -  store copy as Tensor object
        self._freqs_np = np.logspace(np.log(two_pi/self.max_period), np.log(two_pi/self.min_period), self.num_freqs, base=self.base, dtype=np.float32).reshape((1, self.num_freqs))
        self._freqs    = tf.constant(self._freqs_np, dtype=self.dtype)



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


