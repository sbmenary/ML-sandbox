###
###  mathsformer.config.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of methods for program configuration.
"""

import ast, configparser

from collections.abc import Callable

from .utils import summarise_dict



##==================##
##   Config class   ##
##==================##
##
class Config(dict) :
    
    def __init__(self, *argv, **kwargs) :
        """
        class Config
        
        An instance of dictionary class with additional methods to help it act as a config store
        
        Can load configparser.ConfigParser objects, but provides additional functionality in particular regarding 
        nested arguments, printing, and the sketchy use of eval() to cast arguments to Python types instead 
        of storing everything as strings
        
        Inputs:
        
            >  [optional]
               Filename, dictionary or configparser.ConfigParser object objects to load
               In the case of a clash, later arguments are used to override earlier ones
        
            >  [optional]
               Keyword arguments to be loaded as if they were a dict of config values
               
            >  lvl_separator, str, default='>'
               Special character used to split keys into a hierarchy of nested dictionaries
        """
        ##  Initialise as empty dict
        super().__init__()
        
        ##  Load from positional arguments
        lvl_separator = kwargs.pop('lvl_separator', '>')
        for arg in argv :
            self.load(arg, lvl_separator=lvl_separator)
            
        ##  Load keyword arguments
        self.load_dict(kwargs)
        
        
    def __str__(self, lvl_separator:str='>') -> str :
        """
        Return a string representation of dictionary using the self.summary() method
        
        Inputs:
        
            >  lvl_separator, str, default='>'
               Special character used to represent traversal through a hierarchy of nested dictionaries
        """
        line_strs = []
        self.summary(print_fn=lambda x : line_strs.append(x), lvl_separator=lvl_separator)
        return "\n".join(line_strs)
        
    
    def load(self, cfg:str|dict|configparser.ConfigParser, lvl_separator:str='>') -> None :
        """
        Load config values from an object of type str/dict/configparser.ConfigParser
        If str then load as configparser.ConfigParser object first
        
        Inputs:
        
            >  cfg, str/dict/configparser.ConfigParser
               Object to load
               
            >  lvl_separator, str, default='>'
               Special character used to split keys into a hierarchy of nested dictionaries
        """
        ##  Load from string
        if type(cfg) is str :
            return self.load_string(cfg)
            
        ##  Load from dict
        if type(cfg) is dict :
            return self.load_dict(cfg)
            
        ##  Load from Config object
        if type(cfg) is configparser.ConfigParser :
            return self.load_config(cfg)
        
        ##  Otherwise raise TypeError
        raise TypeError(f"Cannot load object of type {type(cfg)}, expected a string, dict or ConfigParser object")
    
    
    def load_config(self, cfg:configparser.ConfigParser, lvl_separator:str='>') -> None :
        """
        Load config values from an object of type configparser.ConfigParser

        WARNING: config values are interpreted using ast.literal_eval to allow appropriate type casting. For safety this does not allow 
        references to packages such as 'tf.float32', and it is still vulnerable to crashes / service attacks using long/malformed strings.
        
        Inputs:
        
            >  cfg, configparser.ConfigParser
               Object to load
               
            >  lvl_separator, str, default='>'
               Special character used to split keys into a hierarchy of nested dictionaries
        """
        ##  Loop over section names
        ##  -  config file is assumed to have no subsections
        for sec_name in cfg.sections() :

     		##  Get corresponding section values
            sec_vals = cfg[sec_name]

            ##  Pull the corresponding section from self dict
            if sec_name not in self :
                self[sec_name] = {}
            section = self[sec_name]

            ##  Loop over keys and values provided
            for key, val in sec_vals.items() :

        		##  Traverse the tree of nested subdictionaries, inserting new key:dict pairs as necessary
                keys = key.split(lvl_separator)
                tree = section
                for key in keys[:-1] :
                    key = key.replace(' ','')
                    if key not in tree :
                        tree[key] = {}
                    tree = tree[key]

                ##  Interpret the final key as the variable name and insert the config value into self dict
                final_key = keys[-1].replace(' ','')
                tree[final_key] = ast.literal_eval(val)
                
                
    def load_dict(self, d:dict, **kwargs) -> None :
        """
        Load config values from another dictionary object
        
        WARNING: d is automatically unrolled - if your config _value_ is itself a dict, beware that it will
        be treated as part of the config hierarchy, meaning that any pre-existing dictionary keys will only 
        be overwritten and not erased
        
        Inputs:
        
            >  d, dict
               Object to load
        
            >  **kwargs
               Named arguments to load by treating them as a dict
        """
        Config._merge_into_dict(self, d     )
        Config._merge_into_dict(self, kwargs)
                
    
    def load_file(self, fname:str, lvl_separator:str='>') -> None :
        """
        Load config values from a config file by loading into a configparser.ConfigParser object
        
        Inputs:
        
            >  fname, str
               Name of file to load
               
            >  lvl_separator, str, default='>'
               Special character used to split keys into a hierarchy of nested dictionaries
        """
        cfg = self._open_config_file(fname)
        return self.load_config(cfg, lvl_separator=lvl_separator)
                
    
    def load_string(self, s:str, lvl_separator:str='>') -> None :
        """
        Load config values from a string by loading into a configparser.ConfigParser object
        
        Inputs:
        
            >  s, str
               String to load
               
            >  lvl_separator, str, default='>'
               Special character used to split keys into a hierarchy of nested dictionaries
        """
        cfg = self._read_config_from_string(s)
        return self.load_config(cfg, lvl_separator=lvl_separator)
    
    
    def summary(self, print_fn:Callable[[str],None]=None, lvl_separator:str=">") -> None :
        """
        Print a summary of the config
        
        Inputs:
        
            >  print_fn, callable with signature print_fn(s), default=print
               Print function
               
            >  lvl_separator, str, default='>'
               Special character used to represent traversal through a hierarchy of nested dictionaries
        """
        ##  Resolve print function
        if print_fn is None :
            print_fn = print
            
        ##  Loop over strings that summarise the dictionary and print
        for line_str in summarise_dict(self, lvl_separator=lvl_separator) :
            print_fn(line_str)
        
    
    @staticmethod
    def _merge_into_dict(self_d:dict, d:dict) -> None :
        """
        Merge values from nested dictionary d into self_d
        
        Inputs:
        
            >  self_d, dict
               Dictionary to be merged into
        
            >  d, dict
               Dictionary to be merged from
        """
        for key, val in d.items() :
            if type(val) != dict :
                self_d[key] = val
                continue
            if key not in self_d :
                self_d[key] = {}
            Config._merge_into_dict(self_d[key], val)
    
    
    def _open_config_file(self, fname:str) :
        """
        Open and return a config file at the filename specified
        
        Inputs:
        
            >  fname, str
               Name of the file to open
        """
        config = configparser.ConfigParser()
        config.read_file(fname)
        return config
    
    
    def _read_config_from_string(self, s:str) :
        """
        Create a config object from the string provided
        
        Inputs:
        
            >  s, str
               String to be read by the config object
        """
        config = configparser.ConfigParser()
        config.read_string(s)
        return config
        