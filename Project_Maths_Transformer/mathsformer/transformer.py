###
###  mathsformer.transformer.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of maths transformer objects.
"""

import numpy      as np
import tensorflow as tf

from tensorflow.keras.layers     import Average, Concatenate, Dense, Embedding, Input, Masking
from tensorflow.keras.losses     import SparseCategoricalCrossentropy
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam

from .tf_objects import (masked_accuracy, masked_sparse_categorical_crossentropy, AttentionBlock, DecoderBlock, EncoderBlock, 
                         Enumerate, FeedForwardBlock, PositionalEncoding)


##=============##
##   Methods   ##
##=============##


def create_text_to_text_model(vocab_length:int, 
                              name:str, 
                              do_compile:bool     = True,
                              dtype_in            = tf.int32, 
                              dtype               = tf.float32, 
                              dropout:float       = 0.1, 
                              optimizer           = Adam,
                              optimizer_args:dict = {'learning_rate':0.001},
                              pos_enc_num_freqs:int       = 32, pos_enc_min_period:float     = 5, pos_enc_max_period:float = 10000,
                              ndim_embedding:int          = 64, comb_type:str                = "average",
                              num_pre_layers_encoder:int  = 0 , ndim_pre_layers_encoder:int  = 512, skip_connect_pre_encoder:bool = True,
                              num_pre_layers_decoder:int  = 0 , ndim_pre_layers_decoder:int  = 512, skip_connect_pre_decoder:bool = True,
                              num_encoder_blocks:int      = 5 , ndim_encoder:int             = 64 , skip_connect_encoder:bool     = True,
                              num_heads_encoder:int       = 8 , ndim_att_hidden_encoder:int  = 128, ndim_ff_hidden_encoder:int    = 128, 
                              num_decoder_blocks:int      = 5 , ndim_decoder:int             = 64 , skip_connect_decoder:bool     = True,
                              num_heads_decoder:int       = 8 , ndim_att_hidden_decoder:int  = 128, ndim_ff_hidden_decoder:int    = 128, 
                              num_post_layers_decoder:int = 3 , ndim_post_layers_decoder:int = 512, 
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

        >  do_compile, bool            , default=True      : Whether to compile the model
        >  dtype_in  , dtype-compatible, default=tf.int32  : Data type of the input tensor
        >  dtype     , dtype-compatible, default=tf.float32: Data type of intermediate and output layers

        >  dropout       , float                , default=0.1                    : Dropout rate used by attention and feed-forward blocks
        >  optimizer     , keras optimizer class, default=Adam                   : Keras optimizer used to compile model
        >  optimizer_args, dict                 , defualt={'learning_rate':0.001}: Keyword arguments used to initialise optimizer class

        >  pos_enc_num_freqs , int  , default=32   : Number of frequencies used for positional encoding, which will be of length 2*pos_enc_num_freqs
        >  pos_enc_min_period, float, default=5    : Lower limit of the geometric series of wave-periods used for positional encoding
        >  pos_enc_max_period, float, default=10000: Upper limit of the geometric series of wave-periods used for positional encoding

        >  ndim_embedding, int, default=64       : Size of the token embeddings
        >  comb_type     , str, default="average": Method for combining the token embeddings and position encodings, options are: ["add", "sum", "average", "mean", "concat", "concatenate"]

        >  num_pre_layers_encoder  , int , default=-1  : Number of layers in the pre-encoder feed-forward block (if <0 then skip entirely)
        >  ndim_pre_layers_encoder , int , default=512 : Number of neurons per-layer in the pre-encoder feed-forward block 
        >  skip_connect_pre_encoder, bool, default=True: Whether to apply a skip-connection across the pre-encoder feed-forward block 

        >  num_pre_layers_decoder  , int , default=-1  : Number of layers in the pre-decoder feed-forward block (if <0 then skip entirely)
        >  ndim_pre_layers_decoder , int , default=512 : Number of neurons per-layer in the pre-decoder feed-forward block 
        >  skip_connect_pre_decoder, bool, default=True: Whether to apply a skip-connection across the pre-decoder feed-forward block 

        >  num_encoder_blocks  , int , default=5   : Number of encoder blocks
        >  ndim_encoder        , int , default=64  : Length of encoder outputs
        >  skip_connect_encoder, bool, default=True: Whether to apply a skip-connection across the encoder's attention and feed-forward blocks 

        >  num_heads_encoder      , int, default=8  : Number of parallel heads in each attention-block
        >  ndim_att_hidden_encoder, int, default=128: Number of neurons in the hidden dimension contracted over to compute attention weights
        >  ndim_ff_hidden_encoder , int, default=128: Number of neurons in the hidden layer of the post-attention feed-forward block

        >  num_decoder_blocks  , int , default=5   : Number of decoder blocks
        >  ndim_decoder        , int , default=64  : Length of decoder outputs
        >  skip_connect_decoder, bool, default=True: Whether to apply a skip-connection across the decoder's attention and feed-forward blocks 

        >  num_heads_decoder      , int, default=8  : Number of parallel heads in each attention-block
        >  ndim_att_hidden_decoder, int, default=128: Number of neurons in the hidden dimension contracted over to compute attention weights
        >  ndim_ff_hidden_decoder , int, default=128: Number of neurons in the hidden layer of the post-attention feed-forward block

        >  num_post_layers_decoder , int, default=3  : Number of layers in the post-decoder feed-forward block 
        >  ndim_post_layers_decoder, int, default=512: Number of neurons per-layer in the post-decoder feed-forward block 
    """
    
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
    ##  -  if comb_type will lead to broadcasting with embeddings later on, then we don't need to repeat the enumerations
    ##     along the batch axis and can use minimal_dims=True for an ouput of [1, S] instead. This saves us memory here
    ##     and reduces the number of operations in the positional encoding step by a factor of B
    minimal_dims = comb_type.lower() in ["add", "sum", "average", "mean"]
    x_pos_enc    = Enumerate(name=f"{name}_encoder_enumerate", dtype=dtype)(x_in_enc, minimal_dims=minimal_dims)
    x_pos_dec    = Enumerate(name=f"{name}_decoder_enumerate", dtype=dtype)(x_in_dec, minimal_dims=minimal_dims)
    
    ##========================================================================##
    ##===  Positional encoding - Output shape [B, S, 2*pos_enc_num_freqs]  ===##
    ##========================================================================##
    x_pos_enc = PositionalEncoding(num_freqs  = pos_enc_num_freqs, 
                                   min_period = pos_enc_min_period, 
                                   max_period = pos_enc_max_period, 
                                   dtype      = dtype, 
                                   name       = f"{name}_encoder_position_encoding")(x_pos_enc)
    x_pos_dec = PositionalEncoding(num_freqs  = pos_enc_num_freqs, 
                                   min_period = pos_enc_min_period, 
                                   max_period = pos_enc_max_period, 
                                   dtype      = dtype, 
                                   name       = f"{name}_decoder_position_encoding")(x_pos_dec)

    ##==============================================================================================##
    ##===  Combine embeddings end pos enc - Output shape [B, S, N] where N depends on comb_type  ===##
    ##==============================================================================================##
    allowed_comb_types = ["add", "sum", "average", "mean", "concat", "concatenate"]
    match comb_type.lower() :
        case "add" | "sum" :
            x_enc = Add(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            x_dec = Add(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case "average" | "mean" :
            x_enc = Average(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            x_dec = Average(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case "concat" | "concatenate" :
            x_enc = Concatenate(name=f"{name}_encoder_emb_and_pos", dtype=dtype)([x_embed_enc, x_pos_enc])
            x_dec = Concatenate(name=f"{name}_decoder_emb_and_pos", dtype=dtype)([x_embed_dec, x_pos_dec])
        case _ :
            raise RuntimeError(f"comb_type '{comb_type}' not recognised, recognised keywords are {allowed_comb_types}")

    ##=========================================================================##
    ##===  Initial pre-processing - Output shape [B, S, ndim_(en/de)coder]  ===##
    ##=========================================================================##
    ##  - use layer_norm instead of batch_norm because elements in sequence are not independent
    if num_pre_layers_encoder >= 0 :
        x_enc = FeedForwardBlock(ndim_encoder, 
                                 ndim_hidden       = ndim_pre_layers_encoder, 
                                 num_hidden_layers = num_pre_layers_encoder, 
                                 dropout           = dropout, 
                                 layer_norm        = True, 
                                 batch_norm        = False,  
                                 skip_connect      = skip_connect_pre_encoder, 
                                 dtype             = dtype, 
                                 name              = f"{name}_encoder_feedfwd_block_pre_attention")(x_enc)
    if num_pre_layers_decoder >= 0 :
        x_dec = FeedForwardBlock(ndim_decoder, 
                                 ndim_hidden       = ndim_pre_layers_decoder, 
                                 num_hidden_layers = num_pre_layers_decoder, 
                                 dropout           = dropout, 
                                 layer_norm        = True, 
                                 batch_norm        = False,  
                                 skip_connect      = skip_connect_pre_decoder, 
                                 dtype             = dtype, 
                                 name              = f"{name}_decoder_feedfwd_block_pre_attention")(x_dec)
    
    ##============================================================##
    ##===  Encoder blocks - Output shape [B, S, ndim_encoder]  ===##
    ##============================================================##
    for layer_idx in range(num_encoder_blocks) :
        x_enc = EncoderBlock(ndim_encoder, 
                             num_heads_encoder, 
                             ndim_att_hidden_encoder, 
                             ndim_ff_hidden_encoder, 
                             dropout_mha  = dropout, 
                             dtype        = dtype, 
                             layer_norm   = True, 
                             skip_connect = skip_connect_encoder, 
                             name         = f"{name}_encoder_block_{layer_idx+1}")(x_enc)
    
    ##============================================================##
    ##===  Decoder blocks - Output shape [B, S, ndim_decoder]  ===##
    ##============================================================##
    for layer_idx in range(num_decoder_blocks) :
        x_dec = DecoderBlock(ndim_decoder, 
                             num_heads_decoder, 
                             ndim_att_hidden_decoder, 
                             ndim_ff_hidden_decoder, 
                             dropout_mha  = dropout, 
                             dtype        = dtype, 
                             layer_norm   = True, 
                             skip_connect = skip_connect_decoder, 
                             name         = f"{name}_decoder_block_{layer_idx+1}")([x_dec, x_enc])
        
    ##==================================================================================================##
    ##===  Predict logit probabilities using feed-forward block - Output shape [B, S, vocab_length]  ===##
    ##==================================================================================================##
    ##  - use layer_norm instead of batch_norm because elements in sequence are not independent
    x = FeedForwardBlock(vocab_length, 
                         ndim_hidden       = ndim_post_layers_decoder, 
                         num_hidden_layers = num_post_layers_decoder, 
                         skip_connect      = False, 
                         layer_norm        = True, 
                         batch_norm        = False, 
                         dtype             = dtype, 
                         name              = f"{name}_feedfwd_block_post_attention")(x_dec)
    
    ##  Create model
    model = Model([x_in_enc, x_in_dec], x, name=name)
    
    ##  Compile model with sparse categorical crossentropy loss and accuracy metric
    if do_compile :
        model.compile(loss      = masked_sparse_categorical_crossentropy, 
                      optimizer = optimizer(**optimizer_args), 
                      metrics   = [masked_accuracy])
    
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
                

    def transform_from_data_tensor(self, X, max_tokens:int=-1, device:str="CPU", strategy:str="argmax") :
        """
        Transform a tensor of input data into its predicted output string using argmax to select tokens
        
        Inputs:
        
            >  X, Tensor with final dimensions [S, 2]
               Tensor of (token, index) pairs for the input sequence of length S
               
            >  max_tokens, int, default=-1
               Maximum tokens in sequence
               
            >  device, str, default="CPU"
               Device to run tensorflow on
               
            >  strategy, str, default="argmax"
               Strategy to use for selecting new tokens, either 'argmax' or 'sample'
        """
        ##  Recurse over tensor of inputs
        if len(X.shape) > 1 and X.shape[0] > 1 :
            return [self.transform_from_data_tensor(Xp, max_tokens, device) for Xp in X]
        
        ##  Check max tokens is long enough to contain a full sequence
        min_sequence_length = len(self.token_transform.seq_start_char) + len(self.token_transform.seq_end_char)
        if max_tokens > 0 and max_tokens < min_sequence_length :
            raise ArgumentError(f"max_tokens must have a minimum length of {min_sequence_length}, {max_tokens} provided")
            
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
                token_logits = self.model([X, Y])[0,-1].numpy()
                
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
               
            >  device, str, default="CPU"
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

