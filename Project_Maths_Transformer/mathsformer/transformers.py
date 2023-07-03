###
###  mathsformer.transformer.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of maths transformer objects.
"""

from __future__ import annotations

import logging, math

import numpy      as np
import tensorflow as tf

from collections.abc import Callable

from tensorflow.keras.layers     import Add, Average, Concatenate, Embedding, Input
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam

from .data import RandomDataGenerator_Addition
from .utils import trim_string
from .tf_objects import (DecoderBlock, EncoderBlock, Enumerate, FeedForwardBlock, LearnableMixture, MaskedCategoricalAccuracy,
                         MaskedSparseCategoricalCrossentropy, PositionalEncoding)
from .tf_objects import scalar_masked_sparse_categorical_crossentropy, scalar_masked_categorical_accuracy


##=================##
##==   Globals   ==##
##=================##

##  Module logger
logger  = logging.getLogger(__name__)


##=============##
##   Methods   ##
##=============##

def create_text_to_text_model(vocab_length:int, 
                              name:str, 
                              do_compile:bool       = True,
                              jit_compile:bool      = None,
                              use_old_loss:bool     = False,
                              dtype_in              = tf.int32, 
                              dtype                 = tf.float32, 
                              dropout:float         = 0.1, 
                              optimizer             = Adam,
                              optimizer_args:dict   = None,
                              idempotent_size:int   = -1,
                              pos_enc_num_freqs:int = 32, pos_enc_min_period:float = 4, pos_enc_max_period:float = 500 , pos_enc_learnable:bool = False, pos_enc_decoder:bool = True,
                              ndim_embedding:int          = 64, comb_type:str                  = "average",
                              num_preencoder_loops:int    = 1 , num_preencoder_blocks:int      = 5  , ndim_preencoder:int           = 64 , skip_connect_preencoder:bool = True, mixture_skip_connect_preencoder:bool = False,
                              num_encoder_loops:int       = 1 , num_encoder_blocks:int         = 5  , ndim_encoder:int              = 64 , skip_connect_encoder:bool    = True, mixture_skip_connect_encoder:bool    = False,
                              num_decoder_loops:int       = 1 , num_decoder_blocks:int         = 5  , ndim_decoder:int              = 64 , skip_connect_decoder:bool    = True, mixture_skip_connect_decoder:bool    = False,
                              num_heads_preencoder:int    = 8 , ndim_att_hidden_preencoder:int = 128, ndim_ff_hidden_preencoder:int = 128, 
                              num_heads_encoder:int       = 8 , ndim_att_hidden_encoder:int    = 128, ndim_ff_hidden_encoder:int    = 128, 
                              num_heads_decoder:int       = 8 , ndim_att_hidden_decoder:int    = 128, ndim_ff_hidden_decoder:int    = 128, 
                              num_post_layers_decoder:int = 3 , ndim_post_layers_decoder:int   = 512, 
                             ) :
    """
    Create a keras model for a seq-to-seq transformer

    Data input shape is [B, S] for batch size B and sequence length S, type integer
    -  inputs are token IDs where 0 entries are masked and propagated throughout model
    Output shape is [B, S, V] where V is the vocab length, type float
    -  outputs are logit probabilities over the token IDs at every step in the sequence

    Inputs:

        >  vocab_length, int: Number of tokens in the dictionary, enumerated as integers on [0, vocab_length)
        >  name        , str: Unique name for the model (also used to derive names of layers and sublayers)

        >  do_compile  , bool            , default=True      : Whether to compile the model
        >  jit_compile , bool            , default=None      : Whether to just-in-time compile the model for XLA acceleration
        >  use_old_loss, bool            , default=False     : Whether to use the old scalar loss function
        >  dtype_in    , dtype-compatible, default=tf.int32  : Data type of the input tensor
        >  dtype       , dtype-compatible, default=tf.float32: Data type of intermediate and output layers

        >  dropout       , float                , default=0.1                    : Dropout rate used by attention and feed-forward blocks
        >  optimizer     , keras optimizer class, default=Adam                   : Keras optimizer used to compile model
        >  optimizer_args, dict                 , defualt={'learning_rate':0.001}: Keyword arguments used to initialise optimizer class

        >  idempotent_size   , int  , default=-1   : Number of additional encoder loops to run, with outputs decoded in parallel for multiple loss contributions
        >  pos_enc_num_freqs , int  , default=32   : Number of frequencies used for positional encoding, which will be of length 2*pos_enc_num_freqs
        >  pos_enc_min_period, float, default=5    : Lower limit of the geometric series of wave-periods used for positional encoding
        >  pos_enc_max_period, float, default=10000: Upper limit of the geometric series of wave-periods used for positional encoding
        >  pos_enc_learnable , bool , default=False: Whether the positional encoding frequencies should be learnable parameters
        >  pos_enc_decoder   , bool , default=True : Whether to use positional encoding for the decoder

        >  ndim_embedding, int, default=64       : Size of the token embeddings
        >  comb_type     , str, default="average": Method for combining the token embeddings and position encodings
                                                   Options are: ["add", "sum", "average", "mean", "concat", "concatenate", "mixture"]

        >  num_preencoder_loops           , int , default=1    : How many times to loop through the pre-encoder blocks
        >  num_preencoder_blocks          , int , default=5    : Number of pre-encoder blocks
        >  ndim_preencoder                , int , default=64   : Length of pre-encoder outputs
        >  skip_connect_preencoder        , bool, default=True : Whether to apply a skip-connection across the pre-encoder's attention and feed-forward blocks 
        >  mixture_skip_connect_preencoder, bool, default=False: Whether to use a LearnableMixture skip connection in the pre-encoder
        >  num_preencoder_loops           , int , default=1    : How many times to pass through the pre-encoder blocks

        >  num_heads_encoder      , int, default=8  : Number of parallel heads in each attention-block
        >  ndim_att_hidden_encoder, int, default=128: Number of neurons in the hidden dimension contracted over to compute attention weights
        >  ndim_ff_hidden_encoder , int, default=128: Number of neurons in the hidden layer of the post-attention feed-forward block

        >  num_encoder_loops           , int , default=1    : How many times to loop through the encoder blocks
        >  num_encoder_blocks          , int , default=5    : Number of encoder blocks
        >  ndim_encoder                , int , default=64   : Length of encoder outputs
        >  skip_connect_encoder        , bool, default=True : Whether to apply a skip-connection across the encoder's attention and feed-forward blocks 
        >  mixture_skip_connect_encoder, bool, default=False: Whether to use a LearnableMixture skip connection in the encoder
        >  num_encoder_loops           , int , default=1    : How many times to pass through the encoder blocks

        >  num_heads_encoder      , int, default=8  : Number of parallel heads in each attention-block
        >  ndim_att_hidden_encoder, int, default=128: Number of neurons in the hidden dimension contracted over to compute attention weights
        >  ndim_ff_hidden_encoder , int, default=128: Number of neurons in the hidden layer of the post-attention feed-forward block

        >  num_decoder_loops           , int , default=1    : How many times to loop through the decoder blocks
        >  num_decoder_blocks          , int , default=5    : Number of decoder blocks
        >  ndim_decoder                , int , default=64   : Length of decoder outputs
        >  skip_connect_decoder        , bool, default=True : Whether to apply a skip-connection across the decoder's attention and feed-forward blocks 
        >  mixture_skip_connect_decoder, bool, default=False: Whether to use a LearnableMixture skip connection in the decoder
        >  num_decoder_loops           , int , default=1    : How many times to pass through the decoder blocks

        >  num_heads_decoder      , int, default=8  : Number of parallel heads in each attention-block
        >  ndim_att_hidden_decoder, int, default=128: Number of neurons in the hidden dimension contracted over to compute attention weights
        >  ndim_ff_hidden_decoder , int, default=128: Number of neurons in the hidden layer of the post-attention feed-forward block

        >  num_post_layers_decoder , int, default=3  : Number of layers in the post-decoder feed-forward block 
        >  ndim_post_layers_decoder, int, default=512: Number of neurons per-layer in the post-decoder feed-forward block 
    """
    ##  Resolve mutable default args
    if optimizer_args is None :
        optimizer_args = {'learning_rate': 1e-3}
    
    ##=============================================##
    ##===   Input layer - Output shape [B, S]   ===##
    ##=============================================##
    x_in_enc = Input((None,), dtype=dtype_in, name=f"{name}_encoder_input_layer")
    x_in_dec = Input((None,), dtype=dtype_in, name=f"{name}_decoder_input_layer")
            
    ##===========================================================================##
    ##===  Token embedding, masking 0s - Output shape [B, S, ndim_embedding]  ===##
    ##===========================================================================##
    x_embed_enc = Embedding(vocab_length, 
                            ndim_embedding, 
                            mask_zero=True, 
                            dtype=dtype, 
                            name=f"{name}_encoder_embedding")(x_in_enc)
    x_embed_dec = Embedding(vocab_length, 
                            ndim_embedding, 
                            mask_zero=True, 
                            dtype=dtype, 
                            name=f"{name}_decoder_embedding")(x_in_dec)
    
    ##=========================================================================##
    ##===  Enumerate indices for positional encoding - Output shape [B, S]  ===##
    ##=========================================================================##
    x_pos_enc = Enumerate(name=f"{name}_encoder_enumerate", dtype=dtype)(x_in_enc, minimal_dims=False)
    x_pos_dec = Enumerate(name=f"{name}_decoder_enumerate", dtype=dtype)(x_in_dec, minimal_dims=False)
    
    ##========================================================================##
    ##===  Positional encoding - Output shape [B, S, 2*pos_enc_num_freqs]  ===##
    ##========================================================================##
    x_pos_enc = PositionalEncoding(num_freqs  = pos_enc_num_freqs, 
                                   min_period = pos_enc_min_period, 
                                   max_period = pos_enc_max_period, 
                                   learnable  = pos_enc_learnable,
                                   dtype      = dtype, 
                                   name       = f"{name}_encoder_position_encoding")(x_pos_enc)
    x_pos_dec = PositionalEncoding(num_freqs  = pos_enc_num_freqs, 
                                   min_period = pos_enc_min_period, 
                                   max_period = pos_enc_max_period, 
                                   learnable  = pos_enc_learnable,
                                   dtype      = dtype, 
                                   name       = f"{name}_decoder_position_encoding")(x_pos_dec)

    ##==============================================================================================##
    ##===  Combine embeddings end pos enc - Output shape [B, S, N] where N depends on comb_type  ===##
    ##==============================================================================================##
    allowed_comb_types = ["add", "sum", "average", "mean", "concat", "concatenate", "mixture"]
    match comb_type.lower() :
        case "add" | "sum" :
            x_enc = Add(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            if pos_enc_decoder :
                x_dec = Add(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case "average" | "mean" :
            x_enc = Average(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            if pos_enc_decoder :
                x_dec = Average(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case "concat" | "concatenate" :
            x_enc = Concatenate(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            if pos_enc_decoder :
                x_dec = Concatenate(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case "mixture" :
            x_enc = LearnableMixture(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            if pos_enc_decoder :
                x_dec = LearnableMixture(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case _ :
            raise RuntimeError(f"comb_type '{comb_type}' not recognised, recognised keywords are {allowed_comb_types}")
    
    ##===================================================================##
    ##===  Pre-encoder blocks - Output shape [B, S, ndim_preencoder]  ===##
    ##===================================================================##
    preencoder_blocks = []
    for layer_idx in range(num_preencoder_blocks) :
        preencoder_blocks.append(EncoderBlock(
                                 ndim_preencoder, 
                                 num_heads_preencoder, 
                                 ndim_att_hidden_preencoder, 
                                 ndim_ff_hidden_preencoder, 
                                 dropout_mha          = dropout, 
                                 dtype                = dtype, 
                                 pre_layer_norm       = True, 
                                 post_layer_norm      = False, 
                                 skip_connect         = skip_connect_preencoder, 
                                 mixture_skip_connect = mixture_skip_connect_preencoder, 
                                 name                 = f"{name}_preencoder_block_{layer_idx+1}"))
    
    for loop_idx in range(num_preencoder_loops) :
        for preencoder_block in preencoder_blocks :
            x_enc = preencoder_block(x_enc)

    ##============================================================##
    ##===  Encoder blocks - Output shape [B, S, ndim_encoder]  ===##
    ##============================================================##
    encoder_blocks = []
    for layer_idx in range(num_encoder_blocks) :
        encoder_blocks.append(EncoderBlock(
                                 ndim_encoder, 
                                 num_heads_encoder, 
                                 ndim_att_hidden_encoder, 
                                 ndim_ff_hidden_encoder, 
                                 dropout_mha          = dropout, 
                                 dtype                = dtype, 
                                 pre_layer_norm       = True, 
                                 post_layer_norm      = False, 
                                 skip_connect         = skip_connect_encoder, 
                                 mixture_skip_connect = mixture_skip_connect_encoder, 
                                 name                 = f"{name}_encoder_block_{layer_idx+1}"))
        
    for loop_idx in range(num_encoder_loops) :
        for encoder_block in encoder_blocks :
            x_enc = encoder_block(x_enc)
    x_enc_list = [x_enc] 
    ## Previously [LayerNormalization(name=f"{name}_encoder_output_norm")(x_enc)] but post-LN now irrelevant
           
    for loop_idx in range(idempotent_size) :
        for encoder_block in encoder_blocks :
            x_enc = encoder_block(x_enc)
        x_enc_list.append(x_enc)
    
    ##===================================================================##
    ##===  Pre-decoder blocks - Output shape [B, S, ndim_predecoder]  ===##
    ##===================================================================##
    '''    predecoder_blocks = []
    for layer_idx in range(num_predecoder_blocks) :
        predecoder_blocks.append(DecoderBlock())
    
    for loop_idx in range(num_predecoder_loops) :
        for predecoder_block in predecoder_blocks :
            x_dec = predecoder_block(x_enc)'''
    
    ##============================================================##
    ##===  Decoder blocks - Output shape [B, S, ndim_decoder]  ===##
    ##============================================================##
    decoder_blocks = []
    for layer_idx in range(num_decoder_blocks) :
        decoder_blocks.append(DecoderBlock(
                                 ndim_decoder, 
                                 num_heads_decoder, 
                                 ndim_att_hidden_decoder, 
                                 ndim_ff_hidden_decoder, 
                                 dropout_mha          = dropout, 
                                 dtype                = dtype, 
                                 pre_layer_norm       = True, 
                                 post_layer_norm      = False, 
                                 skip_connect         = skip_connect_decoder, 
                                 mixture_skip_connect = mixture_skip_connect_decoder, 
                                 name                 = f"{name}_decoder_block_{layer_idx+1}"))
        
    x_dec_list = []
    for x_enc_this in x_enc_list :
        x_dec_this = x_dec
        for loop_idx in range(num_decoder_loops) :
            for decoder_block in decoder_blocks :
                x_dec_this = decoder_block([x_dec_this, x_enc_this])
        x_dec_list.append(x_dec_this)
        
    ##==================================================================================================##
    ##===  Predict logit probabilities using feed-forward block - Output shape [B, S, vocab_length]  ===##
    ##==================================================================================================##
    ##  - use layer_norm instead of batch_norm because elements in sequence are not independent
    ff_block = FeedForwardBlock(vocab_length, 
                         ndim_hidden       = ndim_post_layers_decoder, 
                         num_hidden_layers = num_post_layers_decoder, 
                         skip_connect      = False, 
                         pre_layer_norm    = True, 
                         post_layer_norm   = False, 
                         batch_norm        = False, 
                         dtype             = dtype, 
                         name              = f"{name}_output")
    x_out = [ff_block(x) for x in x_dec_list]
    
    ##  Create model
    model = Model([x_in_enc, x_in_dec], x_out if len(x_out)>1 else x_out[0], name=name)
    
    ##  Compile model with sparse categorical crossentropy loss and accuracy metric
    if do_compile :
        acc  = MaskedCategoricalAccuracy(scalar_output=True, equal_token_weight=True, use_keras_mask=False, mask_value=0)
        loss = MaskedSparseCategoricalCrossentropy(scalar_output=True, equal_token_weight=True, use_keras_mask=False, mask_value=0, from_logits=True)
        model.compile(loss        = loss, 
                      optimizer   = optimizer(**optimizer_args), 
                      metrics     = [acc],
                      jit_compile = jit_compile)
    
    ##  Return model
    return model



##====================================##
##   Transformer_Text_to_Text class   ##
##====================================##
##
class Transformer_Text_to_Text :
    
    def __init__(self, model:Model, token_transform) :
        """
        class Transformer_Text_to_Text
        
        Wrapper for text-to-text model that allows easy transform operations
        
        Input:
        
            >  model, Model
               Sequence-to-sequence keras model
            
            >  token_transform, data.TokenTransform
               Object giving access to (de)tokenising operations
        """
        self.model           = model
        self.token_transform = token_transform


    def masked_transform_from_data_tensor(self, X, Y_in, max_tokens:int=-1, device:str="CPU") :
        """
        Transform a tensor of input data into its predicted output string using masking. This means that all tokens attend to
        the correct representation of those that come before, regardless of whether the correct tokens were predicted before.
        
        Inputs:
        
            >  X, tf.Tensor of shape [N, S1]
               Tensor of tokens for the encoder input sequence 
        
            >  Y_in, tf.Tensor of shape [N, S1]
               Tensor of tokens for the decoder input sequence 
               
            >  max_tokens, int, default=-1
               Maximum tokens in sequence
               
            >  device, str, default="CPU"
               Device to run tensorflow on
        """
        ##  Recurse over tensor of inputs
        if len(X.shape) > 1 and X.shape[0] > 1 :
            return [self.masked_transform_from_data_tensor(Xp, Yp, max_tokens, device) for Xp, Yp in zip(X, Y_in)]
        
        ##  Check max tokens is long enough to contain a full sequence
        min_sequence_length = len(self.token_transform.seq_start_char) + len(self.token_transform.seq_end_char)
        if max_tokens > 0 and max_tokens < min_sequence_length :
            raise ValueError(f"max_tokens must have a minimum length of {min_sequence_length}, {max_tokens} provided")
        
        ##  If X is shape [S, F] then reshape it to [B, S, F] with B=1
        one_sequence_provided = len(X.shape) == 1
        if one_sequence_provided :
            X = tf.expand_dims(X, axis=0)

        ##  If Y_in is shape [S, F] then reshape it to [B, S, F] with B=1
        one_sequence_provided = len(Y_in.shape) == 1
        if one_sequence_provided :
            Y_in = tf.expand_dims(Y_in, axis=0)

        ##  Get output tokens
        with tf.device(device) :
            Y = self.model([X, Y_in])
            if type(Y) in [list, set, tuple] :
                Y = Y[0]
            Y = Y.numpy().argmax(axis=-1)
            

            if type(Y) in [list, set, tuple] :
                Y = Y[0]
                
        ##  Drop first dimension of Y (denoting batch size of length 1)
        Y = Y[0]

        ##  Trim Y to maximum length
        if max_tokens > 0 :
            Y = Y[:max_tokens]

        ##  Convert tensor-of-tokens into a string of detokenised characters, and strip start/end characters
        out_str = "".join([self.token_transform.detokeniser_dict[t] for t in Y])
        out_str.rstrip(self.token_transform.mask_char)
        if out_str[0]  == self.token_transform.seq_start_char : out_str = out_str[1:]
        if out_str[-1] == self.token_transform.seq_end_char   : out_str = out_str[:-1]
            
        ##  Return string with same format as input
        return out_str if one_sequence_provided else [out_str]
                

    def print_predictions_table(self, data_gen, num_print:int, print_fn:Callable[[str],None]=None, max_tokens:int=-1, 
                                min_col_length:int=12, max_col_length:int=30, negative_char:str='-') :
            """
            Print a table showing a number of generated datapoints alongside their predictions from the given transformer
            
            Inputs:
            
                >  data_gen, tf.tensors [[X, Y_in], Y_out] or data.RandomDataGenerator_Addition
                   Data sourve, either tensors or generator object
                
                >  num_print, int, default
                   Number of rows to print
                
                >  print_fn, callable with signature print_fn(str), default=logger.info
                   Function used to print rows
                
                >  max_tokens, int, default=-1
                   Maximum number of tokens allowed to be generated by the transformer
                
                >  min_col_length, int, default=10
                   Minimum column length
                
                >  max_col_length, int, default=30
                   Maximum column length
                
                >  negative_char, str, default='-'
                   Negative character representation
            """
            ##  Resolve print function
            if print_fn is None :
                print_fn = logger.info

            ##  Get data tensors
            if isinstance(data_gen, tf.keras.utils.Sequence) :
                X, Y_in, Y_out = data_gen.get_as_tensors(num_batches=math.ceil(num_print/data_gen.batch_size))
            else :
                (X, Y_in), Y_out = data_gen

            ##  Get str repr of X
            try :
                X_str = self.token_transform.detokenise_strings(X[:num_print,:].numpy())
            except : 
                X_str = [str(x) for x in X[:num_print,0].numpy()]

            if max_col_length > 0 :
                if max_tokens < 1 :
                    max_tokens = max_col_length
                max_tokens = min([max_tokens, max_col_length])

            ##  Get model predictions and log alongside true labels 
            col1, col2, col3, col4, col5, col6 = [], [], [], [], [], []
            for x, y_in, x_str, true_y_str in zip(X   [:num_print], 
                                                  Y_in[:num_print],
                                                  X_str,
                                                  self.token_transform.detokenise_strings(Y_out[:num_print  ].numpy())) :
                pred_y_str_mask = self.masked_transform_from_data_tensor(x, y_in, max_tokens=max_tokens)
                pred_y_str_gen  = self.transform_from_data_tensor(x, max_tokens=max_tokens)
                result = "X  " if pred_y_str_gen == true_y_str else ""
                try    : residual = str(int(pred_y_str_gen.replace(negative_char, "-")) - int(true_y_str.replace(negative_char, "-")))
                except : residual = "?   "
                col1.append(x_str)
                col2.append(true_y_str)
                col3.append(pred_y_str_mask)
                col4.append(pred_y_str_gen)
                col5.append(result)
                col6.append(residual)
            
            ##  Figure out lengths of each column
            l1 = max([min([max([len(x) for x in col1]), max_col_length]) + 1, min_col_length + 1])
            l2 = max([min([max([len(x) for x in col2]), max_col_length]) + 1, min_col_length + 1])
            l3 = max([min([max([len(x) for x in col3]), max_col_length]) + 1, min_col_length + 1])
            l4 = max([min([max([len(x) for x in col4]), max_col_length]) + 1, min_col_length + 1])
            l5 = max([min([max([len(x) for x in col5]), max_col_length]) + 1, min_col_length + 1])
            l6 = max([min([max([len(x) for x in col6]), max_col_length]) + 1, min_col_length + 1])
            lt  = l1 + l2 + l3 + l4 + l5 + l6

            ##  Log table header
            print_fn("-"*lt)
            print_fn("INPUT".rjust(l1) + "TRUE".rjust(l2) + "PRED(MASK)".rjust(l3) + "PRED(GEN)".rjust(l4) + "CORRECT".rjust(l5) + "RESIDUAL".rjust(l6))
            print_fn("-"*lt)
            for s1, s2, s3, s4, s5, s6 in zip(col1, col2, col3, col4, col5, col6) :
                print_fn(trim_string(s1, max_col_length, rjust=l1) + 
                         trim_string(s2, max_col_length, rjust=l2) + 
                         trim_string(s3, max_col_length, rjust=l3) + 
                         trim_string(s4, max_col_length, rjust=l4) + 
                         trim_string(s5, max_col_length, rjust=l5) + 
                         trim_string(s6, max_col_length, rjust=l6))
            

    def transform_from_data_tensor(self, X, max_tokens:int=-1, device:str="CPU", strategy:str="argmax") :
        """
        Transform a tensor of input data into its predicted output string using argmax to select tokens
        
        Inputs:
        
            >  X, tf.Tensor of shape [N, S1]
               Tensor of tokens for the encoder input sequence 
               
            >  max_tokens, int, default=-1
               Maximum tokens in sequence
               
            >  device, str, default="CPU"
               Device to run tensorflow on
               
            >  strategy, str, default="argmax"
               Strategy to use for selecting new tokens, either 'argmax' or 'sample'
        """
        ##  Recurse over tensor of inputs
        if len(X.shape) > 1 and X.shape[0] > 1 :
            return [self.transform_from_data_tensor(Xp, max_tokens, device, strategy) for Xp in X]
        
        ##  Check max tokens is long enough to contain a full sequence
        min_sequence_length = len(self.token_transform.seq_start_char) + len(self.token_transform.seq_end_char)
        if max_tokens > 0 and max_tokens < min_sequence_length :
            raise ValueError(f"max_tokens must have a minimum length of {min_sequence_length}, {max_tokens} provided")
            
        ##  Check that a valid token selection strategy was selected
        match strategy.lower() :
            case 'argmax' : do_argmax = True
            case 'sample' : do_argmax = False
            case _ : raise RuntimeError(f"Strategy '{strategy}' not recognised, choose 'argmax' or 'sample'")
        
        ##  If X is shape [S, F] then reshape it to [B, S, F] with B=1
        one_sequence_provided = len(X.shape) == 1
        if one_sequence_provided :
            X = X[tf.newaxis, :]
            
        #  Find tokens for start and end characters
        start_token = self.token_transform.tokeniser_dict[self.token_transform.seq_start_char]
        end_token   = self.token_transform.tokeniser_dict[self.token_transform.seq_end_char  ]

        ##  Select tf device
        with tf.device(device) :

            ##  Create initial sequence with shape [1, 1] and features [[seq_start_char]]
            Y = tf.cast([[start_token]], dtype=self.token_transform.dtype)

            ##  Keep generating tokens until we reach an end character, or the token limit
            best_token, num_tokens = start_token, 1
            while best_token != end_token and (max_tokens <= 0 or num_tokens < max_tokens) :
                
                ##  Generate logit predictions for all indices; slice logits for first sequence & final index
                token_logits = self.model([X, Y])
                if type(token_logits) in [list, set, tuple] :
                    token_logits = token_logits[0]
                token_logits = token_logits[0,-1].numpy()
                
                ##  Infer a token from the logits generated
                if do_argmax :
                    best_token = np.argmax(token_logits, axis=-1)
                else :
                    best_token = np.random.choice(len(token_logits), p=token_logits)
                
                ##  Append new token to the list
                Y = tf.concat([Y, tf.constant([[best_token]], dtype=self.token_transform.dtype)], axis=1)
                
                ##  Iterate forwards num_tokens
                num_tokens += 1
                
        ##  Drop first dimension of Y (denoting batch size of length 1)
        Y = Y[0].numpy()

        ##  Convert tensor-of-tokens into a string of detokenised characters, and strip start/end characters
        out_str = "".join([self.token_transform.detokeniser_dict[t] for t in Y])
        out_str = out_str[len(self.token_transform.seq_start_char):]
        if out_str[-1] == self.token_transform.seq_end_char : out_str = out_str[:-1]
            
        ##  Return string with same format as input
        return out_str if one_sequence_provided else [out_str]
    
    
    def transform_from_string(self, X, max_tokens:int=-1, device:str="CPU") :
        """
        Transform a list of input strings into their predicted output string using argmax to select tokens
        
        Inputs:
        
            >  X, string or list of strings
               String(s) to be transformed
               
            >  max_tokens, int, default=-1
               Maximum tokens in sequence
               
            >  device, str, default="CPU"ß
               Device to run tensorflow on
        """
        
        ##  Cast data to list format
        one_string_provided = type(X) == str
        if one_string_provided : X = [X]
                        
        ##  Create tensor of input data
        X = self.token_transform.strings_to_tensor(X)
        
        ##  Predict outputs from tensor input
        Y = self.transform_from_data_tensor(X, max_tokens=max_tokens, device=device)
        
        ##  Return strings with same format as input
        return Y[0] if one_string_provided else Y

