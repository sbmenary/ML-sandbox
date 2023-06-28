###
###  mathsformer.selfsupervised_learning_addition_model_backend.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of methods for running self-supervised learning attion model experiments.
"""

import logging, time

import numpy      as np
import tensorflow as tf

from collections.abc import Callable

from matplotlib                         import pyplot as plt
from tensorflow.keras.callbacks         import Callback, EarlyStopping, LambdaCallback, ModelCheckpoint
from tensorflow.keras.models            import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers        import AdamW

from .config       import Config
from .data         import RandomDataGenerator_Addition, TokenTransform
from .tf_objects   import (create_custom_objects_dict, 
                           scalar_masked_categorical_accuracy, scalar_masked_sparse_categorical_crossentropy,
                           MaskedCategoricalAccuracy, MaskedSparseCategoricalCrossentropy, 
                           DecoderBlock, EncoderBlock, Enumerate, FeedForwardBlock, LearnableMixture, PositionalEncoding,
                           AdaptiveLearningRate, LayerActivationRecord, LayerWeightsRecord, LoggerCallback, MetricRecord,
                          )
from .transformers import create_text_to_text_model, Transformer_Text_to_Text


##=================##
##==   Globals   ==##
##=================##

##  Module logger
logger  = logging.getLogger(__name__)

##  Full specification of default config values 
##  -  Do not nodify this object. To override config values, load DEFAULT_CONFIG and then a custom_config into a Config
##     where custom_config contains the override values
DEFAULT_CONFIG = {
    "global" : {
        "base_seed"        : -1,
        "working_dir"      : "SSL_loopy_enc_dec_notebook_[global>problem_tag]_embed[model>ndim_embedding]_enc_[model>encoder>num_blocks]blocks_[model>encoder>num_loops]loops_width[model>encoder>ndim_ff_hidden]_dec_[model>decoder>num_blocks]blocks_[model>decoder>num_loops]loops_width[model>decoder>ndim_ff_hidden]_post[model>post_decoder>num_layers]_width[model>post_decoder>ndim]_idem[model>idempotent_size]_[date]",
        "problem_tag"      : "int1234_num1245",
        "log_lvl_iostream" : logging.INFO,
        "log_lvl_fstream"  : logging.DEBUG,
    },
    "data" : {
        "train_data" : {
            "int_lengths"      : [1, 2, 3, 4],
            "num_ints"         : [1, 2, 4, 5],
            "batch_size"       : 32,
            "num_batches"      : 2000,
            "gen_base_seed"    : 104,
            "gen_reproducible" : False, 
        },
        "val_data" : {
            "int_lengths"      : [1, 2, 3, 4],
            "num_ints"         : [3],
            "batch_size"       : 32,
            "num_batches"      : 50,
            "gen_base_seed"    : 105,
            "gen_reproducible" : True,
        },
        "test_data" : {
            "int_lengths"      : [1, 2, 3, 4],
            "num_ints"         : [6],
            "batch_size"       : 32,
            "num_batches"      : 100,
            "gen_base_seed"    : 106,
            "gen_reproducible" : True,
        },
        "characters"              : ['M', 'B', 'E', 'N', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-'],
        "mask_char"               : 'M',
        "seq_start_char"          : 'B',
        "seq_end_char"            : 'E',
        "negative_char"           : 'N',
        "dtype"                   : "int32",
    },
    "model" : {
        "load_pretrained_model" : None,
        "name"                  : "mathsformer_LLM",
        "dtype"                 : "float32",
        "dropout"               : 0.1,
        "jit_compile"           : False,
        "use_old_loss"          : True,
        "optimizer"             : AdamW,
        "optimizer_args"        : {"learning_rate":1e-4, "weight_decay":2e-2},
        "idempotent_size"       : 2,
        "positional_encoding" : {
            "num_freqs"         : 64,
            "min_period"        : 4,
            "max_period"        : 1000,
            "learnable"         : True,
        },
        "ndim_embedding"        : 128,
        "comb_type"             : 'mixture',
        "pre_encoder" : {
            "num_blocks"           : -1,
            "num_loops"            : 1,
            "num_heads"            : 8,
            "ndim"                 : 128,
            "ndim_att_hidden"      : 128,
            "ndim_ff_hidden"       : 512,
            "skip_connect"         : True,
            "mixture_skip_connect" : True,
        },
        "encoder" : {
            "num_blocks"           : 2,
            "num_loops"            : 4,
            "num_heads"            : 8,
            "ndim"                 : 128,
            "ndim_att_hidden"      : 128,
            "ndim_ff_hidden"       : 512,
            "skip_connect"         : True,
            "mixture_skip_connect" : True,
        },
        "decoder" : {
            "num_blocks"           : 3,
            "num_loops"            : 1,
            "num_heads"            : 8,
            "ndim"                 : 128,
            "ndim_att_hidden"      : 128,
            "ndim_ff_hidden"       : 512,
            "skip_connect"         : True,
            "mixture_skip_connect" : True,
        },
        "post_decoder" : {
            "num_layers"        : 3,
            "ndim"              : 512,
        },
    },
    "training" : {
        "train"          : True,
        "max_epochs"     : 100000,
        "log_after_epoch" : {
            "do"          : True,
            "log_lvl"     : logging.DEBUG,
        },
        "early_stopping" : {
            "do"                   : True,
            "patience"             : 6,
            "monitor"              : "loss",
            "mode"                 : "min",
            "restore_best_weights" : True,
        },
        "model_checkpoint" : {
            "do"       : True,
            "filename" : "model_checkpoint_epoch{epoch}_val_loss_{val_loss:.5}.keras",
        },
        "layer_weights_record" : {
            "do"               : False,
            "batch_frequency"  : 2000,
            "recursive"        : True,
        },
        "adaptive_learning_rate" : {
            "do"                 : True,
            "decay_factor"       : 0.2,
            "monitor"            : "loss",
            "mode"               : "min",
            "patience"           : 2,
            "log_lvl"            : logging.DEBUG,
        },
        "print_tables_during_training" : {
            "do"        : True,
            "num_print" : 10,
        },
    },
    "evaluate" : {
        "num_print"            : 50,
        "save_model"           : True,
        "plot_weights"         : False,
        "plot_training_curves" : True,
    },
}


##=================##
##==   Methods   ==##
##=================##


def create_tensor_dataset(token_transform, max_int:int, negative_char:str='-', include_neg:bool=True, shuffle:bool=True) :
    '''
    Create dataset of tensors X, Y_in, Y_out
    
    X is a tokenised representation of a str "A+B" or "A-B"
    Y is a tokenised representation of the answer, sliced s.t. Y_in=Y[:-1] and Y_out=Y[1:]
    A, B are integers with maximum amplitude given
    
    Inputs:
    
        >  token_transform, data.TokenTransform
           Tokeniser object
           
        >  max_int, int
           Maximum integer amplitude (inclusive)
           
        >  negative_char, str, default='-'
           Character used to represent a negative number
           
        >  include_neg, bool, default=True
           If True then A,B include both +ve and -ve integers, otherwise only +ve
           
        >  shuffle, bool, default=True
           If True then shuffle dataset
           
    Returns:
    
        >  X    , tf.Tensor of shape [N, S1], tensor of N tokenised input sums
        >  Y_in , tf.Tensor of shape [N, S2], tensor of N tokenised answer inputs
        >  Y_out, tf.Tensor of shape [N, S2], tensor of N tokenised answer labels
    '''
    ##  Log start
    logger.info(f"Creating data tensors with max_int={max_int:,}, negative_char={negative_char}, include_neg={include_neg}, shuffle={shuffle}")
    
    ##  Get start time
    start_time = time.time()
    
    ##  Get array of individual numbers
    if include_neg : singles = np.arange(-max_int, max_int+1, dtype=np.int32) 
    else           : singles = np.arange(0       , max_int+1, dtype=np.int32)
        
    ##  Get array of pairs of numbers
    pairs   = np.array([[(x,y) for x in singles] for y in singles])
    pairs   = np.concatenate(pairs)
    
    ##  Get array of A+B and A-B for all pairs (A,B)
    summed  = pairs[:,0] + pairs[:,1]
    minus   = pairs[:,0] - pairs[:,1]
    
    ##  Create dataset of str representations of sums
    dataset = []
    for (i1, i2), s, m in zip(pairs, summed, minus) :
        i1, i2 = f"{i1}".replace("-", negative_char), f"{i2}".replace("-", negative_char)
        s , m  = f"{s }".replace("-", negative_char), f"{m }".replace("-", negative_char)
        dataset.append((f"{i1}+{i2}", f"{s}"))
        dataset.append((f"{i1}-{i2}", f"{m}"))
        
    ##  Shuffle strings
    np.random.shuffle(dataset)
    
    ##  Log strings creation including execution time
    logger.info(f"Created list of strings with length {len(dataset):,} in {time.time()-start_time:.1f}s")
    
    ##  Reset start time for logging
    start_time = time.time()
    
    ##  Convert strings to data tensors
    data_X = token_transform.strings_to_tensor([x[0] for x in dataset])
    data_Y = token_transform.strings_to_tensor([x[1] for x in dataset])
    
    ##  Delete strings from memory
    del dataset
    
    ##  Slice labels into model input and output
    data_Y_in, data_Y_out = data_Y[:,:-1], data_Y[:,1:]
    
    ##  Log final result
    logger.info(f"Strings converted to tensors with shape [{data_X.shape}, {data_Y_in.shape}], {data_Y_out.shape} in {time.time()-start_time:.1f}s")

    ##  Return sliced data tensors
    return data_X, data_Y_in, data_Y_out


def create_text_to_text_model_from_config(cfg_model, token_transform) :
    """
    Create a text-to-text transformer model
    
    Inputs:
    
        >  cfg_model, Config
           Model configuration
           
        >  token_transform, TokenTransform
           Tokeniser
    """
    return create_text_to_text_model(
                          vocab_length                    = token_transform.vocab_length, 
                          name                            = cfg_model["name"],
                          do_compile                      = True,
                          use_old_loss                    = cfg_model["use_old_loss"],
                          dtype_in                        = token_transform.dtype,
                          dtype                           = cfg_model["dtype"],
                          dropout                         = cfg_model["dropout"],
                          jit_compile                     = cfg_model["jit_compile"],
                          optimizer                       = cfg_model.get("optimizer", Adam),
                          optimizer_args                  = cfg_model.get("optimizer_args", {}),
                          idempotent_size                 = cfg_model["idempotent_size"],
                          pos_enc_num_freqs               = cfg_model["positional_encoding"]["num_freqs"],
                          pos_enc_min_period              = cfg_model["positional_encoding"]["min_period"],
                          pos_enc_max_period              = cfg_model["positional_encoding"]["max_period"],
                          pos_enc_learnable               = cfg_model["positional_encoding"]["learnable"],
                          ndim_embedding                  = cfg_model["ndim_embedding"],
                          num_preencoder_blocks           = cfg_model["pre_encoder"]["num_blocks"],
                          num_preencoder_loops            = cfg_model["pre_encoder"]["num_loops"],
                          ndim_preencoder                 = cfg_model["pre_encoder"]["ndim"],
                          num_heads_preencoder            = cfg_model["pre_encoder"]["num_heads"],
                          ndim_att_hidden_preencoder      = cfg_model["pre_encoder"]["ndim_att_hidden"],
                          ndim_ff_hidden_preencoder       = cfg_model["pre_encoder"]["ndim_ff_hidden"],
                          skip_connect_preencoder         = cfg_model["pre_encoder"]["skip_connect"],
                          mixture_skip_connect_preencoder = cfg_model["pre_encoder"]["mixture_skip_connect"],
                          num_encoder_blocks              = cfg_model["encoder"]["num_blocks"],
                          num_encoder_loops               = cfg_model["encoder"]["num_loops"],
                          ndim_encoder                    = cfg_model["encoder"]["ndim"],
                          num_heads_encoder               = cfg_model["encoder"]["num_heads"],
                          ndim_att_hidden_encoder         = cfg_model["encoder"]["ndim_att_hidden"],
                          ndim_ff_hidden_encoder          = cfg_model["encoder"]["ndim_ff_hidden"],
                          skip_connect_encoder            = cfg_model["encoder"]["skip_connect"],
                          mixture_skip_connect_encoder    = cfg_model["encoder"]["mixture_skip_connect"],
                          num_decoder_blocks              = cfg_model["decoder"]["num_blocks"],
                          num_decoder_loops               = cfg_model["decoder"]["num_loops"],
                          ndim_decoder                    = cfg_model["decoder"]["ndim"],
                          num_heads_decoder               = cfg_model["decoder"]["num_heads"],
                          ndim_att_hidden_decoder         = cfg_model["decoder"]["ndim_att_hidden"],
                          ndim_ff_hidden_decoder          = cfg_model["decoder"]["ndim_ff_hidden"],
                          skip_connect_decoder            = cfg_model["decoder"]["skip_connect"],
                          mixture_skip_connect_decoder    = cfg_model["decoder"]["mixture_skip_connect"],
                          num_post_layers_decoder         = cfg_model["post_decoder"]["num_layers"],
                          ndim_post_layers_decoder        = cfg_model["post_decoder"]["ndim"],)




def get_callbacks(cfg_training:Config, working_dir:str="", transformer:Transformer_Text_to_Text=None, 
                  train_gen:RandomDataGenerator_Addition=None, val_gen:RandomDataGenerator_Addition=None,
                  negative_char:str="-") -> list[Callback] :
    """
    Inputs:
    
        >  cfg, mathsformer.Config
           Training configuration
           
        >  working_dir, str, default=""
           Working directory to insert into file paths if needed
    """
    ##  Create list of training callbacks
    callbacks = []
    
    ##  Add logger callback
    logger_callback_config = cfg_training.get("log_after_epoch", {})
    if logger_callback_config.get("do", True) :
        log_lvl         = logger_callback_config.get("log_lvl", logging.DEBUG)
        logger_callback = LoggerCallback(logger, loglvl=log_lvl)
        callbacks.append(logger_callback)
        logger.info(f"Registered training callback: LoggerCallback with loglvl={log_lvl}")

    ##  Add callback for early stopping
    early_stopping_config = cfg_training.get("early_stopping", {})
    if early_stopping_config.get("do", False) :
        monitor              = early_stopping_config.get("monitor"                , "val_loss")
        mode                 = early_stopping_config.get("mode"                   , 'min')
        restore_best_weights = early_stopping_config.get("restore_best_weights"   , True      )
        patience             = early_stopping_config.get("patience"               , 1         )
        early_stopping       = EarlyStopping(monitor              = monitor, 
                                             mode                 = mode,
                                             patience             = patience, 
                                             restore_best_weights = restore_best_weights)
        callbacks.append(early_stopping)
        logger.info(f"Registered training callback: EarlyStopping with monitor={monitor}, mode={mode}, patience={patience}, restore_best_weights={restore_best_weights}")

    ## Adaptive learning rate
    adaptive_learning_rate_config = cfg_training.get("adaptive_learning_rate", {})
    if adaptive_learning_rate_config.get("do", False) :
        decay_factor = adaptive_learning_rate_config.get("decay_factor", 0.5)
        patience     = adaptive_learning_rate_config.get("patience"    , 1)
        monitor      = adaptive_learning_rate_config.get("monitor"     , None)
        mode         = adaptive_learning_rate_config.get("mode"        , 'min')
        log_lvl      = adaptive_learning_rate_config.get("log_lvl"     , logging.DEBUG)
        adaptive_lr  = AdaptiveLearningRate(decay_factor = decay_factor,
                                            patience     = patience,
                                            monitor      = monitor,
                                            mode         = mode,
                                            logger       = logger,
                                            log_lvl      = log_lvl,)
        callbacks.append(adaptive_lr)
        logger.info(f"Registered training callback: AdaptiveLearningRate with decay_factor={decay_factor}, patience={patience}, monitor={monitor}, mode={mode}, log_lvl={log_lvl}")

    ## Add callback for model checkpointing
    model_checkpoint_config = cfg_training.get("model_checkpoint", {})
    if model_checkpoint_config.get("do", False) :
        filename         = model_checkpoint_config.get("filename", "model_checkpoint_epoch{epoch}_val_loss_{val_loss:.5}.keras")
        filepath         = f"{working_dir}/{filename}"
        model_checkpoint = ModelCheckpoint(filepath=filepath)
        callbacks.append(model_checkpoint)
        logger.info(f"Registered training callback: ModelCheckpoint with filepath={filepath}")

    ##  Add callback to record layer activations
    layer_activations_record_config = cfg_training.get("layer_activations_record", {})
    if layer_activations_record_config.get("do", False) :
        batch_frequency = layer_activations_record_config.get("batch_frequency", 1000)
        max_datapoints  = layer_activations_record_config.get("max_datapoints" , 128)
        (val_X, val_Y_in), val_Y_out = val_gen
        layer_activation_record = LayerActivationRecord(
            batch_frequency = batch_frequency, 
            val_input       = [val_X[:max_datapoints], val_Y_in[:max_datapoints]], 
        )
        logger.info(f"Registered training callback: LayerActivationRecord with batch_frequency={batch_frequency}, max_datapoints={max_datapoints}")
        callbacks.append(layer_activation_record)

    ##  Add callback to record layer weights - use recursive=True to monitor all sublayers
    layer_weights_record_config = cfg_training.get("layer_weights_record", {})
    if layer_weights_record_config.get("do", False) :
        batch_frequency      = layer_weights_record_config.get("batch_frequency", 1000)
        recursive            = layer_weights_record_config.get("recursive"      , True)
        layer_weights_record = LayerWeightsRecord(batch_frequency = batch_frequency, 
                                                  recursive       = recursive      )
        callbacks.append(layer_weights_record)
        logger.info(f"Registered training callback: LayerWeightsRecord with batch_frequency={batch_frequency}, recursive={recursive}")

    ##  Add callback for printing table
    print_table_callback_config = cfg_training.get("print_tables_during_training", {})
    if print_table_callback_config.get("do", False) :
        num_print = print_table_callback_config.get("num_print", 5)
        callback  = LambdaCallback(on_epoch_end=lambda batch, logs : test_transformer(
            transformer, train_gen, val_gen, num_print=num_print, print_fn=logger.debug, negative_char=negative_char))
        callbacks.append(callback)
        logger.info(f"Registered training callback: LambdaCallback for test_transformer with num_print={num_print}, negative_char='{negative_char}'")

    ##  Add callback to intermittently record loss & accuracy over small subset of validation data
    loss_record_config = cfg_training.get("loss_record", {})
    if loss_record_config.get("do", False) :
        batch_frequency  = loss_record_config.get("batch_frequency", 1000)
        max_datapoints   = loss_record_config.get("max_datapoints" , 2048)
        num_bootstrap    = loss_record_config.get("num_bootstrap"  , 10)
        plot_frequency   = loss_record_config.get("plot_frequency" , -1)
        plot_after_epoch = loss_record_config.get("plot_after_epoch", False)
        log_lvl          = loss_record_config.get("log_lvl"         , logging.DEBUG)
        (train_X, train_Y_in), train_Y_out = train_gen
        if val_gen is not None :
            (val_X, val_Y_in), val_Y_out = val_gen
        loss_record = MetricRecord(
            batch_frequency   = batch_frequency, 
            data_input        = [train_X[:max_datapoints], train_Y_in[:max_datapoints]], 
            data_output       = train_Y_out[:max_datapoints],
            validation_data   = None if val_gen is None else ([val_X[:max_datapoints], val_Y_in[:max_datapoints]], val_Y_out[:max_datapoints]),
            func              = scalar_masked_sparse_categorical_crossentropy,
            label             = "Partial loss.",
            yscale            = "log",
            num_bootstrap     = num_bootstrap,
            plot_on_train_end = True,
            plot_on_epoch_end = plot_after_epoch,
            plot_frequency    = plot_frequency,
            logger            = logger,
            log_lvl           = log_lvl,
        )
        callbacks.append(loss_record)
        logger.info(f"Registered training callback: MetricRecord with batch_frequency={batch_frequency}, max_datapoints={max_datapoints}, num_bootstrap={num_bootstrap}, plot_frequency={plot_frequency}, plot_after_epoch={plot_after_epoch}, log_lvl=log_lvl")
        acc_record = MetricRecord(
            batch_frequency   = batch_frequency, 
            data_input        = [train_X[:max_datapoints], train_Y_in[:max_datapoints]], 
            data_output       = train_Y_out[:max_datapoints],
            validation_data   = None if val_gen is None else ([val_X[:max_datapoints], val_Y_in[:max_datapoints]], val_Y_out[:max_datapoints]),
            func              = scalar_masked_categorical_accuracy,
            label             = "Partial acc.",
            yscale            = "linear",
            num_bootstrap     = num_bootstrap,
            plot_on_train_end = True,
            plot_on_epoch_end = plot_after_epoch,
            plot_frequency    = plot_frequency,
            logger            = logger,
            log_lvl           = log_lvl,
        )
        callbacks.append(acc_record)
        logger.info(f"Registered training callback: MetricRecord (masked_accuracy) with batch_frequency={batch_frequency}, max_datapoints={max_datapoints}, num_bootstrap={num_bootstrap}")

    ##  Return list of callbacks
    return callbacks



def get_data_generators(cfg_data:Config, token_transform:TokenTransform) -> tuple[RandomDataGenerator_Addition, RandomDataGenerator_Addition, RandomDataGenerator_Addition, RandomDataGenerator_Addition] :
    """
    Create train/val/test data generators

    Inputs:

        >  cfg, Config
           Data configuration

        >  token_transform, TokenTransform
           Tokeniser object
    """
    ##  Create training data generator
    train_gen = RandomDataGenerator_Addition(
                                    token_transform = token_transform, 
                                    int_lengths     = cfg_data["train_data"]["int_lengths"],
                                    num_ints        = cfg_data["train_data"]["num_ints"],
                                    batch_size      = cfg_data["train_data"]["batch_size"],
                                    num_batches     = cfg_data["train_data"]["num_batches"],
                                    base_seed       = cfg_data["train_data"]["gen_base_seed"],
                                    reproducible    = cfg_data["train_data"]["gen_reproducible"],
                                    negative_char   = cfg_data["negative_char"],)
    
    ##  Create training data generator that has forced reproducible=True
    train_gen_reproducible = RandomDataGenerator_Addition(
                                    token_transform = token_transform, 
                                    int_lengths     = cfg_data["train_data"]["int_lengths"],
                                    num_ints        = cfg_data["train_data"]["num_ints"],
                                    batch_size      = cfg_data["train_data"]["batch_size"],
                                    num_batches     = cfg_data["train_data"]["num_batches"],
                                    base_seed       = cfg_data["train_data"]["gen_base_seed"],
                                    reproducible    = True,
                                    negative_char   = cfg_data["negative_char"],)
    
    ##  Log a sample training batch
    logger.info(f"Training data generator created with the following config: {train_gen}")
    (X, Y_in), Y_out = train_gen[0]
    logger.info(f"Output shapes for a test batch are ({X.shape}, {Y_in.shape}), {Y_out.shape}")

    ##  Create validation data generator
    val_gen = RandomDataGenerator_Addition(
                                    token_transform = token_transform, 
                                    int_lengths     = cfg_data["val_data"]["int_lengths"],
                                    num_ints        = cfg_data["val_data"]["num_ints"],
                                    batch_size      = cfg_data["val_data"]["batch_size"],
                                    num_batches     = cfg_data["val_data"]["num_batches"],
                                    base_seed       = cfg_data["val_data"]["gen_base_seed"],
                                    reproducible    = cfg_data["val_data"]["gen_reproducible"],
                                    negative_char   = cfg_data["negative_char"],)
    
    ##  Log a sample validation batch
    logger.info(f"Validation data generator created with the following config: {val_gen}")
    (X, Y_in), Y_out = val_gen[0]
    logger.info(f"Output shapes for a test batch are ({X.shape}, {Y_in.shape}), {Y_out.shape}")

    ##  Create test data generator
    test_gen = RandomDataGenerator_Addition(
                                    token_transform = token_transform, 
                                    int_lengths     = cfg_data["test_data"]["int_lengths"],
                                    num_ints        = cfg_data["test_data"]["num_ints"],
                                    batch_size      = cfg_data["test_data"]["batch_size"],
                                    num_batches     = cfg_data["test_data"]["num_batches"],
                                    base_seed       = cfg_data["test_data"]["gen_base_seed"],
                                    reproducible    = cfg_data["test_data"]["gen_reproducible"],
                                    negative_char   = cfg_data["negative_char"],)
    
    ##  Log a sample test batch
    logger.info(f"Test data generator created with the following config: {test_gen}")
    (X, Y_in), Y_out = test_gen[0]
    logger.info(f"Output shapes for a test batch are ({X.shape}, {Y_in.shape}), {Y_out.shape}")
    
    ##  Return all three generators
    return train_gen, train_gen_reproducible, val_gen, test_gen



def load_text_to_text_model(fname:str) -> Model :
    """
    Load a text-to-text transformer model from file
    
    Inputs:
    
        >  fname, str
           Filename
    """
    ##  Create custom objects dictionary from model layers
    custom_objects = create_custom_objects_dict(Enumerate, PositionalEncoding, LearnableMixture, 
                                                DecoderBlock, EncoderBlock, FeedForwardBlock)
    
    ##  Add custom loss and metrics to custom_objects
    custom_objects["MaskedCategoricalAccuracy"] = MaskedCategoricalAccuracy
    custom_objects["MaskedSparseCategoricalCrossentropy"] = MaskedSparseCategoricalCrossentropy
    custom_objects["scalar_masked_sparse_categorical_crossentropy"] = scalar_masked_sparse_categorical_crossentropy
    custom_objects["scalar_masked_categorical_accuracy"] = scalar_masked_categorical_accuracy
    
    ##  Load model and return
    return tf.keras.models.load_model(fname, custom_objects=custom_objects)



def plot_token_distribution(data_gen, num_batches:int=-1, savefig:str=None, show:bool=True, 
                            close:bool=True, dpi:int=150) :
    """
    Plot the distribution of tokens output by the generator provided.
    
    Inputs:
    
        >  data_gen, RandomDataGenerator_Addition
           Data generator
    
        >  num_batches, int, default=-1
           Number of batches to generate, if <1 then fall back to generator length
           
        >  savefig, str, default=None
           File to save the figure to
           
        >  show, bool, default=True
           Whether to call plt.show(fig)
           
        >  close, bool, default=True
           Whether to call plt.close(fig)
           
        >  dpi, int, default=150
           Pixel density to be passed on to fig.savefig (only relevant for particular file formats)
    """
    ##  Resolve the number of batches
    if num_batches < 1 :
        num_batches = len(data_gen)
    
    ##  Get sample of train data labels
    data_gen_sample = np.concatenate([data_gen[i][1].numpy().flatten() for i in range(num_batches)])

    ##  Ignore masked tokens
    data_gen_sample = data_gen_sample[data_gen_sample != 0]

    ##  Count number for each token
    chars, freqs = [], []
    for token, char in data_gen.token_transform.detokeniser_dict.items() :
        chars.append(char)
        freqs.append(len(data_gen_sample[data_gen_sample==token]))

    ##  Normalise counts to frequency
    freqs     = np.array(freqs).astype(np.float32)
    freqs_err = np.sqrt(freqs)
    freqs_tot = np.sum(freqs)
    freqs     /= freqs_tot
    freqs_err /= freqs_tot

    ##  Log token frequencies
    if logger is not None :
        for char, freq, freq_err in zip(chars, freqs, freqs_err) :
            logger.debug(f"Token '{char}' in data with frequency {100.*freq:.1f} +- {100.*freq_err:.1f} % (masked)")

    ##  Create plot
    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_subplot(1, 1, 1)
    ax.tick_params(which="both", axis="both", right=True, top=True, labelsize=10, direction="in")
    ax.grid()
    ax.set_xlabel("Character", fontsize=14, va="top"  , labelpad=15)
    ax.set_ylabel("Frequency\nin data", fontsize=14, ha="right", rotation=0, labelpad=20)

    ##  Plot bar chart of frequencies
    ax.bar(chars, freqs, yerr=freqs_err)

    ##  Save figure
    if savefig is not None :
        if logger is not None :
            logger.info(f"Saving distribution of token frequencies to file {savefig}")
        fig.savefig(savefig, bbox_inches="tight", dpi=dpi)
        
    ##  Show figure
    if show :
        plt.show(fig)
        
    ##  Close figure
    if close :
        plt.close(fig)



def plot_training_curves(history:dict, show:bool=False, close:bool=False, savefig:str=None, savefig_args:dict=None) :
    """
    Plot the training curves

    Inputs:

        >  history, dict
           Dictionary of metric name : array[values]

        >  show, bool, default=False
           Whether to call plt.show(fig)

        >  close, bool, default=False
           Whether to call plt.close(fig)

        >  savefig, str, default=None
           Whether to call fig.savefig(savefig)

        >  savefig_args, dict, default=None
           Optional named arguments for savefig
    """
    ##  Find and group keys of different metrics to plot
    keys = []
    for key in history :
        if len(key) > 4 and key[:4] == "val_" : continue
        val_key = f"val_{key}"
        val_key = val_key if val_key in history else None
        keys.append((key, val_key))
        
    ##  Create plot figure
    num_rows = len(keys)
    fig      = plt.figure(figsize=(8, 4*num_rows))
    fig.subplots_adjust(hspace=0.08, wspace=0.4)
    
    ##  Loop over key pairs and plot in separate row
    axes = []
    for row_idx, (key1, key2) in enumerate(keys) :
        
        ##  Create and format LHS axis, which has a linear yscale
        ax1 = fig.add_subplot(2*num_rows, 2, 2*row_idx + 1)
        ax1.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
        if row_idx != num_rows - 1 : ax1.xaxis.set_ticklabels([])
        ax1.grid(which="both")
        ax1.set_ylabel("Metric\nvalue", ha="right", va="center", fontsize=12, labelpad=15, rotation=0)
        
        ##  Create and format LHS axis, which has a log yscale
        ax2 = fig.add_subplot(2*num_rows, 2, 2*row_idx + 2)
        ax2.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
        if row_idx != num_rows - 1 : ax2.xaxis.set_ticklabels([])
        ax2.set_yscale("log")
        ax2.grid(which="both")
        
        ##  Plot metric curve for first key
        ax1.plot(history[key1], "x-", alpha=0.5, lw=2, c="darkred", label=key1)
        ax2.plot(history[key1], "x-", alpha=0.5, lw=2, c="darkred", label=key1)
        
        ##  Plot metric curve for second key
        if key2 in history :
            ax1.plot(history[key2], "x-", alpha=0.5, lw=2, c="darkblue", label=key2)
            ax2.plot(history[key2], "x-", alpha=0.5, lw=2, c="darkblue", label=key2)
            
        ##  Draw legend on RHS
        ax2.legend(bbox_to_anchor=(1.05, 1), frameon=False, fontsize=9)
          
        ##  Append axes to list
        axes.append((ax1, ax2))
        
    ##  Set xlabels only for bottom axes
    axes[-1][0].set_xlabel("Epoch index", ha="center", va="top", fontsize=12, labelpad=10)
    axes[-1][1].set_xlabel("Epoch index", ha="center", va="top", fontsize=12, labelpad=10)
        
    ##  Set titles only for top axes
    axes[0][0].set_title("[Linear axis]", fontsize=10, pad=12)
    axes[0][1].set_title("[Log axis]"   , fontsize=10, pad=12)
        
    ##  Show fig
    if show :
        plt.show(fig)
        
    ##  Save fig
    if savefig :
        if savefig_args is None : 
            savefig_args = {}
        savefig_args['bbox_inches'] = savefig_args.get('bbox_inches', "tight")
        savefig_args['dpi'        ] = savefig_args.get('dpi'        , 150)
        fig.savefig(savefig, **savefig_args)
        
    ##  Close fig
    if close :
        plt.close(fig)
        
    ##  Return fig
    return fig



def plot_weights(callbacks:list[Callback], show:bool=True, savefig:str=None) -> None :
    """
    Plot the LayerWeightsRecord callback
    
    Inputs:
    
        >  callbacks, list[Callback]
           List of callbacks provided to model.fit
           
        >  show, bool, default=True
           Whether to call plt.show(fig)
           
        >  savefig, str, default=None
           Filename to save figure to
    """
    
    ##  Find LayerWeightsRecord
    layer_weights_records = [c for c in callbacks if isinstance(c, LayerWeightsRecord)]
    
    ##  Make sure one-and-only-one LayerWeightsRecord exists
    match len(layer_weights_records) :
        case 0 :
            logger.error(f"Cannot plot layer weights - no LayerWeightsRecord found in callbacks")
            return
        case 1 :
            pass
        case _ :
            logger.warning(f"Cannot plot layer weights - multiple LayerWeightsRecord found in callbacks")
            return
    
    ##  Check that some data exists
    layer_weights_record = layer_weights_records[0]
    if len(layer_weights_record.batch_indices) == 0 :
        logger.warning("Cannot plot layer weights - no data found (perhaps we skipped the training run?)")
        return
    
    ##  If we made it to here then plot :)
    logger.info("Plotting layer weights")
    layer_weights_record.plot(num_col=7, show=show, savefig=savefig)
    


def test_transformer(transformer   :Transformer_Text_to_Text,
                     train_gen     :RandomDataGenerator_Addition=None, 
                     val_gen       :RandomDataGenerator_Addition=None, 
                     test_gen      :RandomDataGenerator_Addition=None,
                     num_print     :int=10,
                     max_tokens    :int=15,
                     max_col_length:int=30,
                     negative_char :str="-",
                     print_fn      :Callable[[str],None]=None) -> None :
    """
    Print a short table summarising a few data entries from each generator
    """
    ##  Resolve print_fn
    if print_fn is None :
        print_fn = logger.info

    ##  Test transformer with training generator
    if train_gen is not None :
        print_fn("Running text --> text mathsformer inference on some training data:")
        transformer.print_predictions_table(train_gen, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length, negative_char=negative_char, print_fn=print_fn)
    
    ##  Test transformer with validation generator
    if val_gen is not None :
        print_fn("Running text --> text mathsformer inference on some validation data:")
        transformer.print_predictions_table(val_gen, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length, negative_char=negative_char, print_fn=print_fn)
    
    ##  Test transformer with test generator
    if test_gen is not None :
        print_fn("Running text --> text mathsformer inference on some test data:")
        transformer.print_predictions_table(test_gen, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length, negative_char=negative_char, print_fn=print_fn)



def validate_config(cfg:Config) -> bool :
    """
    Raise exceptions in the case of obvious program misconfigurations
    Otherwise returns True
    """
    ##  Make sure top-level sections are all present
    for req_section in ["global", "data", "model", "training", "evaluate"] :
        if req_section in cfg : continue
        raise KeyError(f"Expected section {req_section} in config, sections are {cfg.keys()}")

    ##  Pull data variables from cfg
    mask_char      = cfg["data"]["mask_char"]
    seq_start_char = cfg["data"]["seq_start_char"]
    seq_end_char   = cfg["data"]["seq_end_char"]
    negative_char  = cfg["data"]["negative_char"]
    char_tokens    = cfg["data"]["characters"]
    
    ##  Check that only single character tokens are provided
    for char_token in char_tokens :
        if len(char_token) == 1 : continue
        raise ValueError(f"All character tokens must be single characters but '{char_tokens}' found")
        
    ##  Check mask character is provided
    if len(mask_char) != 1 :
        raise ValueError(f"Mask character must be a single character but '{mask_char}' provided")
        
    ##  Check mask character in character list
    if mask_char not in char_tokens :
        raise ValueError(f"Mask character '{mask_char}' not found in character list: {char_tokens}")
    
    ##  Check that mask character is first in char_tokens list (ensures it's assigned a token of 0)
    if char_tokens[0] != mask_char :
        raise ValueError(f"Mask character '{mask_char}' must be the first in the char_tokens list provided, "
                        +f"instead found list: {char_tokens}")
        
    ##  Check seq_start_char character is provided
    if len(seq_start_char) != 1 :
        raise ValueError(f"Sequence start character must be a single character but '{seq_start_char}' provided")
        
    ##  Check seq_start_char character in character list
    if seq_start_char not in char_tokens :
        raise ValueError(f"Sequence start character '{seq_start_char}' not found in character list: {char_tokens}")
        
    ##  Check seq_end_char character is provided
    if len(seq_end_char) != 1 :
        raise ValueError(f"Sequence end character must be a single character but '{seq_end_char}' provided")
        
    ##  Check seq_start_char character in character list
    if seq_end_char not in char_tokens :
        raise ValueError(f"Sequence end character '{seq_end_char}' not found in character list: {char_tokens}")
        
    ##  Check negative_char character is provided
    if len(negative_char) != 1 :
        raise ValueError(f"Negative symbol character must be a single character but '{negative_char}' provided")
        
    ##  Check negative_char character in character list
    if negative_char not in char_tokens :
        raise ValueError(f"Negative symbol character '{negative_char}' not found in character list: {char_tokens}")
        
    ##  If here then config validated correctly
    return True
