###
###  mathsformer.data.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of methods for data generation and manipulation.
"""

from __future__ import annotations

import logging, math, time

import numpy as np
import tensorflow as tf

from collections.abc import Callable

from .utils import CustomLogLevel


##=================##
##==   Globals   ==##
##=================##

##  Module logger
logger  = logging.getLogger(__name__)



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
        dataset = dataset[..., np.newaxis]
        
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
           If -1 then we will use the smallest length that covers all data without trimming

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
    if logger : logger.log(CustomLogLevel.DEBUG, f"Input strings:\n{strings[:log_elements]}")
    
    ##  Tokenise strings
    dataset = tokenise_strings(strings, tokeniser_dict, fix_output_length=fix_output_length,
                               mask_char=mask_char, seq_start_char=seq_start_char, seq_end_char=seq_end_char)
    if logger : logger.log(CustomLogLevel.DEBUG, f"Tokenised dataset:\n{dataset[:log_elements]}")
    
    ##  Add feature corresponding to token index in sequence
    if add_position_indices : 
        dataset = add_positions_to_sequences(dataset, zero_indexed=zero_indexed)
        if logger : logger.log(CustomLogLevel.DEBUG, f"Enumerated dataset:\n{dataset[:log_elements]}")
    
    ##  Convert dataset to tensor of type int32
    dataset = tf.constant(dataset, dtype=dtype)
    if logger : logger.log(CustomLogLevel.DEBUG, f"Tensor dataset:\n{dataset[:log_elements]}")
    
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
           Fixed length of token sequences obtained by trimming and padding as appropriate
           If < 0 then use smallest length that covers all sequences without trimming
           
        >  mask_char, str, default=''
           Character used to pad string to target length
           
        >  seq_start_char, str, default=''
           Character to be appended to the start of a string
           
        >  seq_end_char, str, default=''
           Character to be appended to the end of a string
    """

    ##  Derive fix_output_length
    if fix_output_length < 0 : 
        max_string_length = max([len(s) for s in dataset])
        fix_output_length = max_string_length + len(seq_start_char) + len(seq_end_char)

    ##  Tokenise every string in dataset
    return np.array([tokenise_string(s, 
                                     tokeniser_dict,
                                     fix_output_length=fix_output_length,
                                     mask_char=mask_char,
                                     seq_start_char=seq_start_char,
                                     seq_end_char=seq_end_char) for s in dataset])



def train_val_test_split(X, split_idx1:int, split_idx2:int) :
    """
    Split tensor X into three along first axis.
    if split_idx1, split_idx2 are in the range [0, 1) then interpret them as near-as-possible fractions of X

    Inputs:

        >  X, tensor
           Tensor to be sliced along first axis

        >  split_idx1, int or float
           If int then index of first split, otherwise approximate fraction of X

        >  split_idx2, int or float
           If int then index of second split, otherwise approximate fraction of X
    """
    ##  Resolve split indices
    num_X = len(X)
    if split_idx1 < 1 : split_idx1 = int(split_idx1*num_X)
    if split_idx2 < 1 : split_idx2 = int(split_idx2*num_X)

    ##  Return three split tensors
    return X[:split_idx1], X[split_idx1:split_idx2], X[split_idx2:]

   

##========================================##
##   RandomDataGenerator_Addition class   ##
##========================================##
##
class RandomDataGenerator_Addition(tf.keras.utils.Sequence) :
    
    def __init__(self, token_transform:TokenTransform, int_lengths:list, num_ints:list, batch_size:int, num_batches:int, 
                 base_seed:int=-1, reproducible:bool=False, negative_char:str='-') :
        """
        class RandomDataGenerator
        
        Data generator used to create individual batches of input/output data on-the-fly for a keras model.
        If reproducible=True then self[i] will always generate the same result, otherwise it will sample new
        data every time it is called.
        WARNING: data are not guaranteed to be unique - different batches may contain the same datapoints!
        Text sequences are of the the form "A +- B +- C..."
        Data are formatted for sequence-to-sequence text model, e.g. for the sum "12 + 36 = 48" we might have:
           >  X  =  ["B12+36EMMM", "B48EM"]
           >  Y  =  "48EMM"
        
        Inputs:
        
            >  token_transform, TokenTransform
               Method for transforming strings to/from tokenised tensors
               
            >  int_lengths, list
               List of allowed integer-lengths (N.B. will be sampled uniformly, so 1-digit numbers occur with the
               same frequency as N-digit numbers!)
               
            >  num_ints, list
               List of allowed integer-multiplicities (N.B. will be sampled uniformly, so sequences of 1 number
               occur with the same frequency as sequences of N numbers!)
               
            >  batch_size, int
               Number of sequences per batch
               
            >  num_batches, int
               Number of batches to constitute a full epoch
               
            >  base_seed, int, default=-1
               Random seed used to initialise random number generator, if -1 then fall back to system time
        
            >  reproducible, bool, default=False
               If True then re-initialise the rng seed to base_seed + i when calling self[i] for reproducible results
               Otherwise do not re-initialise rng, allowing it to continue generating potentially new datapoints

            >  negative_char, str, default='N'
               Character used to represent a negative number
        """
        if base_seed < 0 :
            base_seed = int(time.time())
        
        self.token_transform = token_transform
        self.int_lengths     = int_lengths
        self.num_ints        = num_ints
        self.batch_size      = batch_size
        self.num_batches     = num_batches
        self.base_seed       = base_seed
        self.reproducible    = reproducible
        self.negative_char   = negative_char
        self.reset_rng()

    
    def __getitem__(self, index:int) :
        """
        Returns a new set of tensors ([X, Y_in], Y_out) with length self.batch_size
        
        Inputs:
        
            >  index, int
               Index of the call, only meaningful if we have self.reproducible = True
        """
        if self.reproducible :
            self.reset_rng(self.base_seed + index)
        x = self.rng.choice(self.num_ints, size=(self.batch_size,))
        y = [self._generate_string(xp) for xp in x]
        X, Y = [yp[0] for yp in y], [yp[1] for yp in y]
        X = self.token_transform.strings_to_tensor(X)
        Y = self.token_transform.strings_to_tensor(Y)
        return [X, Y[:,:-1]], Y[:,1:]
    
    
    def __len__(self) :
        """
        Following generator convention: returns number of batches
        """
        return self.num_batches
    
    
    def __str__(self) :
        """
        Returns a string summarising the generator configuration
        """
        return f"Generator of {self.num_ints} integers of length {self.int_lengths} in {self.num_batches} batches of size {self.batch_size} (base_seed={self.base_seed}, reproducible={self.reproducible})"
    
    
    def _generate_int_string(self, length:int) :
        """
        Returns a string representation of a random integer with the length provided
        
        Inputs:
        
            >  length, int
               Number of digits in the integer
        """
        sign        = self.rng.choice(["", "N"])
        lead_char   = str(self.rng.randint(1, 10))
        other_chars = "".join([str(self.rng.randint(0, 10)) for i in range(length-1)])
        return sign + lead_char + other_chars
    
    
    def _generate_int_strings(self, lengths:list) :
        """
        Returns an array of strings representating random integers with the lengths provided
        
        Inputs:
        
            >  length, list-of-int
               Number of digits for each integer
        """
        return np.array([self._generate_int_string(l) for l in lengths])
    
    
    def _generate_string(self, num:int) :
        """
        Returns a pair of strings (X, Y) where X is a sum and Y is the result
        
        Inputs:
        
            >  num, int
               Number of integers in the sum
        """
        lengths = self.rng.choice(self.int_lengths, size=(num,))
        ints    = self._generate_int_strings(lengths)
        out_s, out_i = ints[0], int(ints[0].replace(self.negative_char,"-"))
        for si in ints[1:] :
            f = self.rng.uniform(0, 1)
            i = int(si.replace(self.negative_char,"-"))
            if f < 0.5 :
                out_s += "+" + si
                out_i += i
            else :
                out_s += "-" + si
                out_i -= i
        return out_s, str(out_i).replace("-",self.negative_char)
    
    
    def get_as_tensors(self, num_batches:int=-1) :
        """
        Create a number of batches and combine their outputs into a single set of tensors
        
        Inputs:
        
            >  num_batches, int, default=-1
               Number of batches to generate, if < 1 then fall back to self.num_batches
        """
        ##  If num batches not set then return all of them
        if num_batches < 1 :
            num_batches = self.num_batches
        
        ##  Containers to stores batches
        X, Y_in, Y_out = [], [] ,[]

        ##  Fill containers with batch results
        for i in range(num_batches) :
            [x, yi], yo = self[i]
            X, Y_in, Y_out = X + [x], Y_in + [yi], Y_out + [yo]

        ##  Find max widths of tensors, which currently have ragged shapes
        len_x, len_yi, len_yo = max([xp.shape[1] for xp in X]), max([xp.shape[1] for xp in Y_in]), max([xp.shape[1] for xp in Y_out])

        ##  Pad all tensors to the same width
        for i in range(len(X)) :
            X    [i] = tf.pad(X    [i], [[0, 0], (0, len_x -X    [i].shape[1])])
            Y_in [i] = tf.pad(Y_in [i], [[0, 0], (0, len_yi-Y_in [i].shape[1])])
            Y_out[i] = tf.pad(Y_out[i], [[0, 0], (0, len_yo-Y_out[i].shape[1])])

        ##  Concatenate batch results into single tensor
        X, Y_in, Y_out = tf.concat(X, axis=0), tf.concat(Y_in, axis=0), tf.concat(Y_out, axis=0)
        
        ##  Return
        return X, Y_in, Y_out
    
    
    def print_predictions_table(self, transformer, num_print:int, print_fn:Callable[[str],None]=None, max_tokens:int=-1, 
                                min_col_length:int=10, max_col_length:int=30) :
        """
        Print a table showing a number of generated datapoints alongside their predictions from the given transformer
        
        Inputs:
        
            >  transformer
               Transformer object used to generate predictions
               
            >  num_print, int
               Number of rows to print
               
            >  print_fn, callable with signature print_fn(str), default=logger.info
               Function used to print rows
               
            >  max_tokens, int, default=-1
               Maximum number of tokens allowed to be generated by the transformer
               
            >  min_col_length, int, default=10
               Minimum column length
               
            >  max_col_length, int, default=30
               Maximum column length
        """
        if print_fn is None :
            print_fn = logger.info

        X, Y_in, Y_out = self.get_as_tensors(num_batches=math.ceil(num_print/self.batch_size))

        ##  Get model predictions and log alongside true labels 
        col1, col2, col3, col4, col5 = [], [], [], [], []
        for x, x_str, true_y_str in zip(X[:num_print], 
                                        transformer.token_transform.detokenise_strings(X    [:num_print,:].numpy()),
                                        transformer.token_transform.detokenise_strings(Y_out[:num_print  ].numpy())) :
            pred_y_str = transformer.transform_from_data_tensor(x, max_tokens=max_tokens)
            result     = "X  " if pred_y_str == true_y_str else ""
            try    : residual = str(int(pred_y_str.replace(self.negative_char,"-")) - int(true_y_str.replace(self.negative_char,"-")))
            except : residual = "?   "
            col1.append(x_str)
            col2.append(true_y_str)
            col3.append(pred_y_str)
            col4.append(result)
            col5.append(residual)
          
        ##  Figure out lengths of each column
        l1 = max([min([max([len(x) for x in col1]), max_col_length]) + 1, min_col_length + 1])
        l2 = max([min([max([len(x) for x in col2]), max_col_length]) + 1, min_col_length + 1])
        l3 = max([min([max([len(x) for x in col3]), max_col_length]) + 1, min_col_length + 1])
        l4 = max([min([max([len(x) for x in col4]), max_col_length]) + 1, min_col_length + 1])
        l5 = max([min([max([len(x) for x in col5]), max_col_length]) + 1, min_col_length + 1])
        lt  = l1 + l2 + l3 + l4 + l5

        ##  Log table header
        print_fn("-"*lt)
        print_fn("INPUT".rjust(l1) + "TRUE".rjust(l2) + "PRED".rjust(l3) + "CORRECT".rjust(l4) + "RESIDUAL".rjust(l5))
        print_fn("-"*lt)
        for s1, s2, s3, s4, s5 in zip(col1, col2, col3, col4, col5) :
            print_fn(s1[:max_col_length].rjust(l1) + s2[:max_col_length].rjust(l2) + s3[:max_col_length].rjust(l3) + s4[:max_col_length].rjust(l4) + s5[:max_col_length].rjust(l5))
            
            
    def reset_rng(self, seed:int=-1) :
        """
        Set the internal rng with the seed provided
        
        Inputs:
        
            >  seed, int, default=-1
               Random seed, if < 0 then fall back to self.base_seed
        """
        if seed < 0 :
            seed = self.base_seed
        self.rng = np.random.RandomState(seed)
        
        
    def summary(self, print_fn:Callable[[str],None]=None) :
        """
        Print a summary of the generator
        
        Inputs:
        
            >  print_fn, callable with signature print_fn(str), default=print
               Function used to print strings
        """
        if print_fn is None :
            print_fn = print
        print_fn(str(self))

        

##==========================##
##   TokenTransform class   ##
##==========================##
##
class TokenTransform :
    
    def __init__(self, characters:list=None, seq_start_char:str='', seq_end_char:str='', mask_char:str='', dtype:str="int32") :
        """
        class TokenTransform
        
        Wrapper for methods used to perform (de)tokenisation, storing the special characters and
        tokenisation dictionaries
        
        Inputs:
        
            >  characters, list, list of characters provided as length-1 strings
               Characters used to define the dictionary - must already contain special characters
               
            >  seq_start_char, str, default=''
               Character to denote the beginning of a sequence
               
            >  seq_end_char, str, default=''
               Character to denote the end of a sequence
               
            >  mask_char, str, default=''
               Character to denote masked values
               
            >  dtype, dtype-compatible object, default='int32'
               Data type for any data tensors created
        """
        ##  Resolve muable defaults
        if characters is None : characters = []

        ##  Set internal variables
        self.characters     = characters
        self.seq_start_char = seq_start_char
        self.seq_end_char   = seq_end_char
        self.mask_char      = mask_char
        self.dtype          = dtype
        self.vocab_length   = len(self.characters)
        
        ##  Check configuration and set dictionaries
        self.initialise()
    
    
    def __len__(self) :
        """
        Return number of characters registered
        """
        return len(self.char_tokens)
    
    
    def __str__(self) :
        """
        Return a string summary of this class using the self.summary method
        """
        summary = []
        self.summary(lambda s : summary.append(s))
        return "\n".join(summary)
        
        
    def detokenise_string(self, x:list) :
        """
        Convert list-of-tokens into a string of characters
        
        Inputs:
        
            >  x, list
               List-of-tokens to be detokenised
        """
        return detokenise_string(x, 
                                 detokeniser_dict = self.detokeniser_dict, 
                                 mask_char        = self.mask_char, 
                                 seq_start_char   = self.seq_start_char, 
                                 seq_end_char     = self.seq_end_char)
        
        
    def detokenise_strings(self, dataset:list) :
        """
        Convert list-of-list-of-tokens into list-of-strings
        
        Inputs:
        
            >  dataset, list
               List-of-lists-of-tokens to be detokenised
        """
        return detokenise_strings(dataset, 
                                  detokeniser_dict = self.detokeniser_dict, 
                                  mask_char        = self.mask_char, 
                                  seq_start_char   = self.seq_start_char, 
                                  seq_end_char     = self.seq_end_char)
    
    @classmethod
    def from_dictionary(cls, d:dict) :
        """
        Create a new TokenTransform object by pulling constructor keyword arguments from the dictionary provided
        
        Inputs:
        
            >  d, dict
               Dictionary of keyword arguments including those needed by the class constructor 
        """
        return cls(characters     = d.get("characters"    , []), 
                   seq_start_char = d.get("seq_start_char", ''),
                   seq_end_char   = d.get("seq_end_char"  , ''),
                   mask_char      = d.get("mask_char"     , ''),
                   dtype          = d.get("dtype"         , ''))
        
        
    def initialise(self) :
        """
        Check valid configuration, then construct internal dictionaries for future (de)tokenisation
        """
        ##  Validate seq_start_char
        match len(self.seq_start_char) :
            case 0 : pass
            case 1 : 
                if self.seq_start_char not in self.characters :
                    raise RuntimeError(f"seq_start_char '{self.seq_start_char}' is not in characters list {self.characters}")
            case _ : 
                raise RuntimeError(f"seq_start_char must have length of 0 or 1, but '{self.seq_start_char}' provided")
        
        ##  Validate seq_end_char
        match len(self.seq_end_char) :
            case 0 : pass
            case 1 : 
                if self.seq_end_char not in self.characters :
                    raise RuntimeError(f"seq_end_char '{self.seq_end_char}' is not in characters list {self.characters}")
            case _ : 
                raise RuntimeError(f"seq_end_char must have length of 0 or 1, but '{self.seq_end_char}' provided")
        
        ##  Validate mask_char
        match len(self.mask_char) :
            case 0 : pass
            case 1 : 
                if self.mask_char not in self.characters :
                    raise RuntimeError(f"mask_char '{self.mask_char}' is not in characters list {self.characters}")
            case _ : 
                raise RuntimeError(f"mask_char must have length of 0 or 1, but '{self.mask_char}' provided")
                
        ##  Create dictionaries
        self.tokeniser_dict   = dict([(t,i) for i,t in enumerate(self.characters)])
        self.detokeniser_dict = dict([(i,t) for i,t in enumerate(self.characters)])
        
        ##  Make sure mask token is 0
        mask_token = self.tokeniser_dict[self.mask_char]
        if mask_token != 0 :
            raise RuntimeError(f"Mask character {self.mask_char} with a token value {mask_token}, expected 0. Fix: make sure the mask character is first in the list provided!")
    
    
    def strings_to_tensor(self, strings:list, add_position_indices:bool=False, logger=None) :
        """
        Convert a list-of-strings to a fixed-size tensor of tokens
        
        Inputs:
        
            >  strings, list
               List of strings to be tokenised
               
            >  add_position_indices, bool, default=False
               If True then output tensor includes an additional axis with enumerated sequences indices
               
            >  logger, default=None
               Optional logger to pass to backend function for debugging
        """
        return strings_to_tensor(strings,
                                 tokeniser_dict       = self.tokeniser_dict, 
                                 mask_char            = self.mask_char, 
                                 seq_start_char       = self.seq_start_char, 
                                 seq_end_char         = self.seq_end_char, 
                                 add_position_indices = add_position_indices, 
                                 logger               = logger,
                                 dtype                = self.dtype)
    
    
    def summary(self, print_fn:Callable[[str],None]=None) :
        """
        Print a summary of the special characters and tokenising dictionaries contained.
        
        Inputs:
        
            >  print_fn, callable function with signature print_fn(str), default=print
               Function used to print the summary
        """
        if print_fn is None :
            print_fn = print
        print_fn(f"TokenTransform of dtype {self.dtype} with {self.vocab_length} characters: {self.characters}")
        print_fn(f"Special characters are seq_start_char ({self.seq_start_char}), seq_end_char ({self.seq_end_char}), mask_char ({self.mask_char})")
        print_fn(f"Tokeniser dictionary is {self.tokeniser_dict}")
        print_fn(f"Detokeniser dictionary is {self.detokeniser_dict}")
        
        
    def tokenise_string(self, s:str, fix_output_length:int=-1) :
        """
        Convert a character string into a list of tokens
        
        Inputs:
    
        >  s, str
           String to be encoded
           
        >  fix_output_length, int, default=-1
           Length of tokenised sequences, if < 0 then no trimming or padding applied
        """
        return tokenise_string(s, 
                               tokeniser_dict    = self.tokeniser_dict, 
                               fix_output_length = fix_output_length, 
                               mask_char         = self.mask_char, 
                               seq_start_char    = self.seq_start_char, 
                               seq_end_char      = self.seq_end_char)
        
        
    def tokenise_strings(self, dataset:list, fix_output_length:int=-1) :
        """
        Create a square numpy array containing the tokenised strings
        
        Inputs:
    
        >  dataset, list
           List of strings to be encoded
           
        >  fix_output_length, int, default=-1
           Length of tokenised sequences, if < 0 then use smallest length that covers all sequences without trimming
        """
        return tokenise_strings(dataset, 
                                tokeniser_dict    = self.tokeniser_dict, 
                                fix_output_length = fix_output_length, 
                                mask_char         = self.mask_char, 
                                seq_start_char    = self.seq_start_char, 
                                seq_end_char      = self.seq_end_char)

