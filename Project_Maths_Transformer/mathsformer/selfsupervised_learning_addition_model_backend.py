###
###  mathsformer.selfsupervised_learning_addition_model_backend.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of methods for running self-supervised learning attion model experiments.
"""

import logging

import tensorflow as tf

from matplotlib                 import pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models    import Model

from .config       import Config
from .data         import RandomDataGenerator_Addition, TokenTransform
from .tf_objects   import (create_custom_objects_dict, 
                           masked_accuracy, masked_sparse_categorical_crossentropy, 
                           DecoderBlock, EncoderBlock, Enumerate, FeedForwardBlock, LearnableMixture, PositionalEncoding,
                           AdaptiveLearningRate, LayerWeightsRecord, LoggerCallback,
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
        "working_dir"      : "working_dir_[problem_tag]_[model_tag]_[date]",
        "problem_tag"      : "baseline",
        "model_tag"        : "baseline",
        "log_lvl_iostream" : logging.INFO,
        "log_lvl_fstream"  : logging.DEBUG,
    },
    "data" : {
        "train_data" : {
            "int_lengths"      : [1, 2, 3],
            "num_ints"         : [1, 2, 4],
            "batch_size"       : 32,
            "num_batches"      : 4000,
            "gen_base_seed"    : 100,
            "gen_reproducible" : False, 
        },
        "val_data" : {
            "int_lengths"      : [1, 2, 3],
            "num_ints"         : [3],
            "batch_size"       : 32,
            "num_batches"      : 500,
            "gen_base_seed"    : 101,
            "gen_reproducible" : True,
        },
        "test_data" : {
            "int_lengths"      : [4],
            "num_ints"         : [3],
            "batch_size"       : 32,
            "num_batches"      : 1000,
            "gen_base_seed"    : 102,
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
        "learning_rate"         : 1e-3,
        "jit_compile"           : False,
        "positional_encoding" : {
            "num_freqs"         : 16,
            "min_period"        : 4,
            "max_period"        : 250,
        },
        "ndim_embedding"        : 32,
        "comb_type"             : 'average',
        "pre_encoder"           : {
            "num_layers"        : -1,
            "ndim"              : 128,
            "skip_connect"      : True,
        },
        "pre_decoder" : {
            "num_layers"        : -1,
            "ndim"              : 128,
            "skip_connect"      : True,
        },
        "encoder" : {
            "num_blocks"        : 5,
            "num_heads"         : 8,
            "ndim"              : 32,
            "ndim_att_hidden"   : 32,
            "ndim_ff_hidden"    : 128,
            "skip_connect"      : True,
        },
        "decoder" : {
            "num_blocks"        : 5,
            "num_heads"         : 8,
            "ndim"              : 32,
            "ndim_att_hidden"   : 32,
            "ndim_ff_hidden"    : 128,
            "skip_connect"      : True,
        },
        "post_decoder" : {
            "num_layers"        : 3,
            "ndim"              : 128,
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
            "monitor"              : "val_masked_accuracy",
            "mode"                 : "max",
            "restore_best_weights" : True,
        },
        "model_checkpoint" : {
            "do"       : True,
            "filename" : "model_checkpoint_epoch{epoch}_val_loss_{val_loss:.5}.h5",
        },
        "layer_weights_record" : {
            "do"               : True,
            "batch_frequency"  : 4000,
            "recursive"        : True,
        },
        "adaptive_learning_rate" : {
            "do"                 : True,
            "decay_factor"       : 0.3,
            "monitor"            : "loss",
            "mode"               : "min",
            "patience"           : 2,
            "log_lvl"            : logging.DEBUG,
        },
    },
    "evaluate" : {
        "num_print"            : 20,
        "save_model"           : True,
        "plot_weights"         : False,
        "plot_training_curves" : True,
    },
}


##=================##
##==   Methods   ==##
##=================##


def create_text_to_text_model_from_config(cfg_model:Config, token_transform:TokenTransform) :
    """
    Create a text-to-text transformer model
    
    Inputs:
    
        >  cfg_model, Config
           Model configuration
           
        >  token_transform, TokenTransform
           Tokeniser
    """
    return create_text_to_text_model(
                          vocab_length             = token_transform.vocab_length, 
                          name                     = cfg_model["name"],
                          do_compile               = True,
                          dtype_in                 = token_transform.dtype,
                          dtype                    = cfg_model["dtype"],
                          dropout                  = cfg_model["dropout"],
                          jit_compile              = cfg_model["jit_compile"],
                          optimizer_args           = {"learning_rate": cfg_model["learning_rate"]},
                          pos_enc_num_freqs        = cfg_model["positional_encoding"]["num_freqs"],
                          pos_enc_min_period       = cfg_model["positional_encoding"]["min_period"],
                          pos_enc_max_period       = cfg_model["positional_encoding"]["max_period"],
                          ndim_embedding           = cfg_model["ndim_embedding"],
                          comb_type                = cfg_model["comb_type"],
                          num_pre_layers_encoder   = cfg_model["pre_encoder"]["num_layers"],
                          ndim_pre_layers_encoder  = cfg_model["pre_encoder"]["ndim"],
                          skip_connect_pre_encoder = cfg_model["pre_encoder"]["skip_connect"],
                          num_pre_layers_decoder   = cfg_model["pre_decoder"]["num_layers"],
                          ndim_pre_layers_decoder  = cfg_model["pre_decoder"]["ndim"],
                          skip_connect_pre_decoder = cfg_model["pre_decoder"]["skip_connect"],
                          num_encoder_blocks       = cfg_model["encoder"]["num_blocks"],
                          ndim_encoder             = cfg_model["encoder"]["ndim"],
                          skip_connect_encoder     = cfg_model["encoder"]["skip_connect"],
                          num_heads_encoder        = cfg_model["encoder"]["num_heads"],
                          ndim_att_hidden_encoder  = cfg_model["encoder"]["ndim_att_hidden"],
                          ndim_ff_hidden_encoder   = cfg_model["encoder"]["ndim_ff_hidden"],
                          num_decoder_blocks       = cfg_model["decoder"]["num_blocks"],
                          ndim_decoder             = cfg_model["decoder"]["ndim"],
                          skip_connect_decoder     = cfg_model["decoder"]["skip_connect"],
                          num_heads_decoder        = cfg_model["decoder"]["num_heads"],
                          ndim_att_hidden_decoder  = cfg_model["decoder"]["ndim_att_hidden"],
                          ndim_ff_hidden_decoder   = cfg_model["decoder"]["ndim_ff_hidden"],
                          num_post_layers_decoder  = cfg_model["post_decoder"]["num_layers"],
                          ndim_post_layers_decoder = cfg_model["post_decoder"]["ndim"])



def get_callbacks(cfg_training:Config, working_dir:str="") -> list[Callback] :
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
        logger.info(f"Registeried training callback: AdaptiveLearningRate with decay_factor={decay_factor}, patience={patience}, monitor={monitor}, mode={mode}, log_lvl={log_lvl}")

    ## Add callback for model checkpointing
    model_checkpoint_config = cfg_training.get("model_checkpoint", {})
    if model_checkpoint_config.get("do", False) :
        filename         = model_checkpoint_config.get("filename", "model_checkpoint_epoch{epoch}_val_loss_{val_loss:.5}.h5")
        filepath         = f"{working_dir}/{filename}"
        model_checkpoint = ModelCheckpoint(filepath=filepath)
        callbacks.append(model_checkpoint)
        logger.info(f"Registeried training callback: ModelCheckpoint with filepath={filepath}")

    ##  Add callback to record layer weights - use recursive=True to monitor all sublayers
    layer_weights_record_config = cfg_training.get("layer_weights_record", {})
    if layer_weights_record_config.get("do", False) :
        batch_frequency      = layer_weights_record_config.get("batch_frequency", 1000)
        recursive            = layer_weights_record_config.get("recursive"      , True)
        layer_weights_record = LayerWeightsRecord(batch_frequency = batch_frequency, 
                                                  recursive       = recursive      )
        callbacks.append(layer_weights_record)
        logger.info(f"Registered training callback: LayerWeightsRecord with batch_frequency={batch_frequency}, recursive={recursive}")

    ##  Return list of callbacks
    return callbacks



def get_data_generators(cfg_data:Config, token_transform:TokenTransform) -> tuple[RandomDataGenerator_Addition, RandomDataGenerator_Addition, RandomDataGenerator_Addition] :
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
    return train_gen, val_gen, test_gen



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
    custom_objects["masked_sparse_categorical_crossentropy"] = masked_sparse_categorical_crossentropy
    custom_objects["masked_accuracy"] = masked_accuracy
    
    ##  Load model and return
    return tf.keras.models.load_model(fname, custom_objects=custom_objects)



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
        ax1.xaxis.set_ticklabels([])
        ax1.grid(which="both")
        ax1.set_ylabel("Metric\nvalue", ha="right", va="center", fontsize=12, labelpad=15, rotation=0)
        
        ##  Create and format LHS axis, which has a log yscale
        ax2 = fig.add_subplot(2*num_rows, 2, 2*row_idx + 2)
        ax2.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
        ax2.xaxis.set_ticklabels([])
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
                     train_gen     :RandomDataGenerator_Addition, 
                     val_gen       :RandomDataGenerator_Addition, 
                     test_gen      :RandomDataGenerator_Addition,
                     num_print     :int=10,
                     max_tokens    :int=30,
                     max_col_length:int=30) -> None :
    """
    Print a short table summarising a few data entries from each generator
    """
    ##  Test transformer with training generator
    logger.info("Running text --> text mathsformer inference on some training data:")
    train_gen.print_predictions_table(transformer, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length)
    
    ##  Test transformer with validation generator
    logger.info("Running text --> text mathsformer inference on some validation data:")
    val_gen.print_predictions_table(transformer, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length)
    
    ##  Test transformer with test generator
    logger.info("Running text --> text mathsformer inference on some test data:")
    test_gen.print_predictions_table(transformer, num_print=num_print, max_tokens=max_tokens, max_col_length=max_col_length)



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
