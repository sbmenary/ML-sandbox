###
###  mathsformer.utils.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of methods for data generation and manipulation.
"""

import numpy as np
import tensorflow as tf



##=============##
##   Methods   ##
##=============##


def add_positions_to_sequences(dataset, zero_indexed:bool=True) :
    """
    Take dataset of tokens and add a new feature containing the sequence position indices
    
    Inputs:
    
        >  dataset, array of shape (M,N) or (M,N,1) for M sequences of N tokens
           Tokenised dataset
           
        >  zero_indexed, bool, default=True
           Whether to begin indexing at 0, otherwise begin at 1
    """
    
    ##  Make sure shape is [M,N,1]
    if len(dataset.shape) == 2 :
        dataset = dataset.reshape(dataset.shape+(1,))
        
    ##  Create array of sequence indices with shape [M,N,1]
    pos_idcs = np.arange(dataset.shape[1])
    if not zero_indexed : pos_idcs += 1
    pos_idcs = np.array([pos_idcs for i in range(dataset.shape[0])])
    pos_idcs = pos_idcs.reshape(pos_idcs.shape+(1,))
    
    ##  Concatenate arrays along final axis and return
    return np.concatenate([dataset, pos_idcs], axis=-1)



def detokenise_string(x:list, detokeniser_dict:dict, mask_char:str='', seq_start_char:str='', seq_end_char:str='') :
    """
    Convert sequence of tokens x into its string representation
    
    Inputs:
    
        >  x, 1D-iterable of integers such as list or array of shape (N,)
           Sequence to be decoded
           
        >  detokeniser_dict, dict
           Dictionary of token:character pairs
           
        >  mask_char, str, default=''
           Character to be stripped from the end of the string (masked characters inside the string are preserved)
           
        >  seq_start_char, str, default=''
           Character to be stripped from the start of a string
           
        >  seq_end_char, str, default=''
           Character to be stripped from the end of a string
    """
    s = ''.join([detokeniser_dict[c] for c in x])
    s = s.rstrip(mask_char)
    if s[0 ] == seq_start_char : s = s[1:]
    if s[-1] == seq_end_char   : s = s[:-1]
    return s



def detokenise_strings(dataset:list, detokeniser_dict:dict, mask_char:str='', seq_start_char:str='', seq_end_char:str='') :
    """
    Convert a list of token sequences into their string representations
    
    Inputs:
    
        >  dataset, 2D-iterable of integers such as list or array of shape (M,N)
           List of sequences to be decoded
           
        >  detokeniser_dict, dict
           Dictionary of token:character pairs
           
        >  mask_char, str, default=''
           Character to be stripped from the end of the string (masked characters inside the string are preserved)
           
        >  seq_start_char, str, default=''
           Character to be stripped from the start of a string
           
        >  seq_end_char, str, default=''
           Character to be stripped from the end of a string
    """
    return np.array([detokenise_string(x, 
                                       detokeniser_dict, 
                                       mask_char=mask_char, 
                                       seq_start_char=seq_start_char, 
                                       seq_end_char=seq_end_char) for x in dataset])



def strings_to_tensor(strings, tokeniser_dict:dict, fix_output_length:int=-1, mask_char:str='', seq_start_char:str='', seq_end_char:str='', 
					  zero_indexed:bool=True, add_position_indices:bool=True, logger=None, log_elements:int=8, dtype=tf.int32) :
    """
    Convert a list of strings into a square tensor of tokenised datapoints, optionally with positional indices

    Inputs:

    	>  strings, list
    	   List of strings to be converted into a square tensor of tokenised datapoints, optionally with positional indices

    	>  tokeniser_dict, dict
    	   Dictionary of character:integer pairs used to tokenise the string

    	>  fix_output_length, int, default=-1
    	   Fixed length of the output sequence, including seq_start_char/seq_end_char (strings will be trimmed & padded if necessary)
    	   If -1 then we will not trim & pad, and rely on you having provided a list of constant-length strings

    	>  mask_char, single character str, default=''
    	   Mask character - must be provided if masking is to be performed

    	>  seq_start_char, single character str, default=''
    	   Special character to append to the start of the string

    	>  seq_end_char, single character str, default=''
    	   Special character to append to the end of the string

    	>  zero_indexed, bool, default=True
    	   If True then we begin the positional indices at 0, otherwise begin at 1

    	>  add_position_indices, bool, default=True
    	   Whether to include positional indices in the dataset created

    	>  logger, Logger, default=None
    	   If logger provided then use it to print log messages as we work

    	>  log_elements, int, default=8
    	   Number of data rows to print when debugging tensor values

    	>  dtype, tf.dtype compatible, default=tf.int32
    	   Dtype of the output tensor

    """
    ##  Record initial strings for debug
    if logger : logger.debug(f"Input strings:\n{strings[:log_elements]}")
    
    ##  Tokenise strings
    dataset = tokenise_strings(strings, tokeniser_dict, fix_output_length=fix_output_length,
                               mask_char=mask_char, seq_start_char=seq_start_char, seq_end_char=seq_end_char)
    if logger : logger.debug(f"Tokenised dataset:\n{dataset[:log_elements]}")
    
    ##  Add feature corresponding to token index in sequence
    if add_position_indices : 
    	dataset = add_positions_to_sequences(dataset, zero_indexed=zero_indexed)
    	if logger : logger.debug(f"Enumerated dataset:\n{dataset[:log_elements]}")
    
    ##  Convert dataset to tensor of type int32
    dataset = tf.constant(dataset, dtype=dtype)
    if logger : logger.debug(f"Tensor dataset:\n{dataset[:log_elements]}")
    
    ##  Return new dataset
    return dataset



def tokenise_string(s:str, tokeniser_dict:dict, fix_output_length:int=-1, mask_char:str='', seq_start_char:str='', seq_end_char:str='') :
    """
    Convert string s into sequence of tokens
    
    Inputs:
    
        >  s, str
           String to be encoded
           
        >  tokeniser_dict, dict
           Dictionary of character:token pairs
           
        >  fix_output_length, int, default=-1
           If >0 then fix encoding to this length, trimming and padding as appropriate 
           
        >  mask_char, str, default=''
           Character used to pad string to target length
           
        >  seq_start_char, str, default=''
           Character to be appended to the start of a string
           
        >  seq_end_char, str, default=''
           Character to be appended to the end of a string
    """
    
    ##  Validate inputs
    len_mask, len_start, len_end = len(mask_char), len(seq_start_char), len(seq_end_char)
    if len_mask  > 1 : raise RuntimeError(f"mask_char must be a single character but '{mask_char}' provided")
    if len_start > 1 : raise RuntimeError(f"seq_start_char must be a single character but '{seq_start_char}' provided")
    if len_end   > 1 : raise RuntimeError(f"seq_end_char must be a single character but '{seq_end_char}' provided")
    
    ##  Compute effective string length, accounting for start and end characters
    str_length = fix_output_length - len_start - len_end
    if fix_output_length > 0 and str_length < 1 : raise RuntimeError(f"fix_output_length={fix_output_length} cannot be satisfied with seq_start_char={seq_start_char} and seq_end_char={seq_end_char}")
    
    ##  Create new str with trimming and extra characters
    s = f"{seq_start_char}{s[:str_length]}{seq_end_char}"

    ##  Only call ljust if needed, since mask_char may not have been provided
    if len(s) < fix_output_length : s = s.ljust(fix_output_length, mask_char)
        
    ##  Tokenise new str
    return [tokeniser_dict[c] for c in s]



def tokenise_strings(dataset:list, tokeniser_dict:dict, fix_output_length:int=-1, mask_char:str='', seq_start_char:str='', seq_end_char:str='') :
    """
    Convert list of strings into sequence of tokens
    
    Inputs:
    
        >  dataset, list
           List of strings to be encoded
           
        >  tokeniser_dict, dict
           Dictionary of character:token pairs
           
        >  fix_output_length, int, default=-1
           If >0 then fix encoding to this length, trimming and padding as appropriate 
           
        >  mask_char, str, default=''
           Character used to pad string to target length
           
        >  seq_start_char, str, default=''
           Character to be appended to the start of a string
           
        >  seq_end_char, str, default=''
           Character to be appended to the end of a string
    """
    return np.array([tokenise_string(s, 
                                     tokeniser_dict,
                                     fix_output_length=fix_output_length,
                                     mask_char=mask_char,
                                     seq_start_char=seq_start_char,
                                     seq_end_char=seq_end_char) for s in dataset])

