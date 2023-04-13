###
###  mathsformer.utils.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of utility methods.
"""

import __main__
import datetime, enum, logging, os, random, sys, time

import numpy      as np
import tensorflow as tf



##===============##
##   Constants   ##
##===============##

class CustomLogLevel(enum.IntEnum) :
	"""
	Enumeration for cutom log-levels that live between logging.DEBUG=10 and logging.INFO=20.
	This allows us to enumerate package-specific debug-levels without falling all the way back to the system DEBUG level.
	"""
	CRITICAL          = 50
	ERROR             = 40
	WARNING           = 30
	INFO              = 20
	DEBUG_VERY_LOW    = 19
	DEBUG_LOW         = 18
	DEBUG_MEDIUM_LOW  = 16
	DEBUG_MEDIUM      = 15
	DEBUG_MEDIUM_HIGH = 14
	DEBUG_HIGH        = 12
	DEBUG_VERY_HIGH   = 11
	DEBUG             = 10



##=============##
##   Methods   ##
##=============##

def create_working_directory(working_dir:str, increment:bool=True, tags:dict=None) :
    """
    Create a new working directory at the location working_dir, resolving keyword arguments.
    
    Allowed keywords are:
        >  [date]:  the current date with the format %Y_%m_%d
        >  [time]:  the current time with the format %H%M%S
        >  [A>B>C]:  a value from the dict provided, with key structure tags[A][B][C]
    
    Inputs:
    
        >  working_dir, str
           Name of the directory to be created, including keyword arguments
           
        >  increment, bool, default=True
           If True then we create a directory with an incremented version number if working_dir already exists
           
        >  tags, dict, default=None
           Dictionary containing additional keyword args
    """

    ##  Insert global tags
    working_dir = working_dir.replace("[date]", datetime.datetime.today().strftime('%Y_%m_%d'))
    working_dir = working_dir.replace("[time]", datetime.datetime.today().strftime('%H%M%S'))

    ##  Insert tags from dictionary
    replace_tags = extract_brace_substrings(working_dir)
    for tag in replace_tags :
        if tag == "date" : continue
        if tag == "time" : continue
        if not tags : raise KeyError(f"Tag '{tag}' requested but no dictionary of tags provided")
        val = tags
        for key in tag.split(">") :
            val = val[key]
        working_dir = working_dir.replace(f"[{tag}]", str(val).replace(".","p").replace("-","m"))

    ##  Iterate until we find a working directory name that doesn't already exist
    trial_working_dir, version = working_dir, 1
    while os.path.exists(trial_working_dir) :
        if not increment : 
            raise RuntimeError(f"Directory {trial_working_dir} already exists and using increment=False")
        version += 1
        trial_working_dir = f"{working_dir}_v{version}"
        
    ##  Create dir and return its name
    os.mkdir(trial_working_dir)
    return trial_working_dir



def extract_brace_substrings(s:str, open_brace_char:str="[", close_brace_char:str="]") :
    """
    Find all substrings in s enclosed by the characters open_brace_char and close_brace_char.
    Warning: assumes that substrings follow a valid structure of nested braces.
    
    Inputs:
    
        >  s, str
           String to be parsed
           
        >  open_brace_char, str, default='['
           Character to be interpreted as an opening brace
           
        >  close_brace_char, str, default=']'
           Character to be interpreted as a closing brace
    """
    
    ##  Check arguments
    if len(open_brace_char ) != 1 : raise RuntimeError(f"open_brace_char='{open_brace_char }' is not a single character")
    if len(close_brace_char) != 1 : raise RuntimeError(f"open_brace_char='{close_brace_char}' is not a single character")
    if open_brace_char == close_brace_char : raise RuntimeError(f"open_brace_char cannot be the same as close_brace_char ('{open_brace_char}' provided)")
    
    ##  Initialise variables before looping over characters in s 
    substrings   = []    #  List of results
    nesting_lvl  = 0     #  Keep track of how deep we have gone within nesting structure
    substr_start = 0     #  Keep track of the index when the current substr started
    
    ##  Loop through characters and search for substr,
    for char_idx, c in enumerate(s):
        
        ##  If open-brace char then record start of substr if outer brace, and increment nesting level
        if c == open_brace_char :
            if not nesting_lvl : substr_start = char_idx + 1
            nesting_lvl += 1
            continue
            
        ##  If close-brace char then decrement nesting level, and record substr if we closed an outer brace
        if c == close_brace_char :
            if not nesting_lvl : raise RuntimeError(f"String '{s}' does not contain a valid structure of braces with format '{open_brace_char}...{close_brace_char}'")
            nesting_lvl -= 1
            if not nesting_lvl: substrings.append(s[substr_start:char_idx])
            continue
    
    ##  Check that we closed the final braces correctly
    if nesting_lvl :
        raise RuntimeError(f"String '{s}' does not contain a valid structure of braces with format '{open_brace_char}...{close_brace_char}'")
    
    ##  Return substrings
    return substrings



def fancy_message(message:str) :
    """
    Return the given message with the format
    
    ==================
    ===   message  ===
    ==================
    
    Inputs:
    
        >  message, str
           Message to be placed inside fancy block
    """
    middle_str        = f"===   {message}   ==="
    middle_str_length = len(middle_str)
    enclosing_str     = "="*middle_str_length
    return f"{enclosing_str}\n{middle_str}\n{enclosing_str}"



def initialise_logging(name:str=None, iostream=sys.stdout, fname:str=None, log_lvl_iostream:int=logging.INFO, log_lvl_fstream:int=logging.DEBUG) :
    """
    Create new logger object without output streams to stdout and to file.
    
    Inputs:

    	>  name, str, default=None
    	   Name of the logger to be configured
    
        >  iostream, streamable object, default=sys.stdout
           Output stream, if None then do not create an iostream
    
        >  fname, str, default=None
           Name of log file, if None then do not create a filestream
    
        >  log_lvl_iostream, int, default=logging.INFO
           Log-level for output stream to stdout
    
        >  log_lvl_fstream, int, default=logging.DEBUG
           Log-level for output stream to file
           
    Returns:
    
        >  logger, Logger: global logger with our output streams
    """
    
    ##  Get root logger
    logger = logging.getLogger(name)

    ##  Create stream to output log messages, default is sys.stdout
    if type(iostream) != type(None) :
        io_handler = logging.StreamHandler(sys.stdout)
        io_handler.setFormatter(logging.Formatter("%(levelname)7s %(funcName)s: %(message)s"))
        io_handler.setLevel(log_lvl_iostream)
        logger.addHandler(io_handler)

    ##  Create stream to output log messages to file
    if fname :
        f_handler = logging.FileHandler(fname)
        f_handler.setFormatter(logging.Formatter("%(levelname)7s %(asctime)s %(filename)s %(funcName)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        f_handler.setLevel(log_lvl_fstream)
        logger.addHandler(f_handler)

    ##  Set root logger level
    handler_lvls = [h.level for h in logger.handlers]
    if len(handler_lvls) > 0 :
        logger.setLevel(min(handler_lvls))
    
    ##  Log start time 
    logger.info(f"Begin logging on {datetime.datetime.today().strftime('%Y-%m-%d')} at {datetime.datetime.today().strftime('%H:%M:%S')}")
    
    ##  Return global logger
    return logger



def initialise_program(program_description:str, working_dir:str, global_config:dict, logger_name:str="mathsformer", log_lvl_iostream:int=logging.INFO, log_lvl_fstream:int=logging.DEBUG) :
    """
    Groups many repeated program initialisation routines together. Steps are:
    1. Create a new working directory at working_dir, using global_config to resolve any name tags
    2. Create a new logger object without one output stream to stdout, and another to a file in working_dir
    3. Log versions of all compatible modules stored in sys.modules
    4. Log config values stored in global_config
    5. Initialise random seeds of python random, numpy and tensorflow for reproducibility
    
    Inputs:

    	>  program_description, str
    	   Description of the program being run
    
        >  working_dir, str
           Name of working directory to create
    
        >  global_config, dict
           Dictionary of config values

        >  logger_name, str, default='mathsformer'
           Name of the logger to be configured
    
        >  log_lvl_iostream, int, default=logging.INFO
           Log-level for output stream to stdout
    
        >  log_lvl_fstream, int, default=logging.DEBUG
           Log-level for output stream to file
           
    Returns:
    
        >  working_dir, str   : name of the working directory we created
        >  logger     , Logger: global logger with our output streams
        >  base_seed  , int   : set base python seed
        >  np_seed    , int   : set numpy seed
        >  tf_seed    , int   : set tensorflow seed
    """
    
    ##  Create working directory
    working_dir = create_working_directory(working_dir, tags=global_config)
    print(fancy_message(f"Working directory created at {working_dir}"))
    
    ##  Start logging to stdout and to a file stream in working directory
    logger = initialise_logging(name=logger_name, fname=f"{working_dir}/log.txt", log_lvl_iostream=log_lvl_iostream, log_lvl_fstream=log_lvl_fstream)

    ##  Log program being run
    if hasattr(__main__, "__file__") : logger.info(f"Running program: {__main__.__file__}")
    logger.info(f"Program description: {program_description}")
    logger.info(f"Working directory: {working_dir}")
    
    ##  Log package versions for reproducibility
    log_versions(logger, pull_from_sys=True, pull_submodules=True)
    
    ##  Log config values
    for config_str in summarise_dict(global_config) :
        logger.info(f"Registered global config value {config_str}")
        
    ##  Initialise random seeds for reproducibility
    base_seed = global_config.get("base_seed", -1)
    base_seed, np_seed, tf_seed = initialise_random_seeds(base_seed) 
    logger.info(f"Python random seed set: {base_seed}")
    logger.info(f"Numpy random seed set: {np_seed}")
    logger.info(f"TensorFlow random seed set: {tf_seed}")
    
    ##  Return the things we created!
    return working_dir, logger, base_seed, np_seed, tf_seed
        
        

def initialise_random_seeds(base_seed:int=-1) :
    """
    Set the random seeds for base python, numpy and tensorflow.
    In principle this should be enough to create reproducible results, assuming equal environments.
    If base_seed < 1 then we use the current system time as the base seed
    To avoid any nasty overlap, we define np_seed = base_seed + 1 and tf_seed = base_seed + 2
    
    Inputs:
    
        >  base_seed, int, default=-1
           Seed to use for base python, if < 1 then default to system time
           
    Returns:
    
        >  base_seed, int: set base python seed
        >  np_seed  , int: set numpy seed
        >  tf_seed  , int: set tensorflow seed
    """

    ##  If base_seed <= 0 then replace with current system time
    if base_seed < 1 :
        base_seed = int(time.time())

    ##  Set seed for python stdlib random
    random.seed(base_seed)

    ##  Set seed for numpy random
    np_seed = base_seed + 1
    np.random.seed(np_seed)

    ##  Set seed for tf random
    tf_seed = base_seed + 2
    tf.random.set_seed(tf_seed)
    
    ##  Return seeds
    return base_seed, np_seed, tf_seed



def log_versions(logger, packages:list=[], pull_from_sys:bool=False, pull_submodules:bool=False) :
    """
    Log the module versions
    
    Inputs:
    
        >  logger, python Logger object
           Logger to log to
           
        >  packages, list, default=[]
           List of packages, assumed to have a __version__ attribute we can query

        >  pull_from_sys, bool, default=False
           If True then pull all parent modules from sys.modules

        >  pull_submodules, bool, default=False
           If True then also pull submodules from sys.modules
    """
    ##  If pull_from_sys then add modules from sys.modules
    if pull_from_sys :
    	packages = [(k,v) for k,v in sys.modules.items() if hasattr(v, "__version__") and (pull_submodules or "." not in lstrip_multiple(k, [".","/"]))]
    	packages = sorted(packages, key=lambda p:p[0])
    	packages = [v for k,v in packages]
    	return log_versions(logger, packages)

    ##  Create nicely-formatted strings
    pkg_names    = ["Python"   ] + [str(pkg.__name__   ) if hasattr(pkg, "__name__"   ) else "unknown" for pkg in packages]
    pkg_versions = [sys.version] + [str(pkg.__version__) if hasattr(pkg, "__version__") else "unknown" for pkg in packages]
    len_names, len_versions = max([7, max([len(s) for s in pkg_names])]), max([7, max([len(s) for s in pkg_versions])])
    split_str    = "-"*(len_names+2) + "+" + "-"*(len_versions+2)
    
    ##  Print strings to logger
    logger.info(split_str)
    logger.info(   "PACKAGE".rjust(len_names) +  "  |  VERSION")
    logger.info(split_str)
    for name, version in zip(pkg_names, pkg_versions) :
        logger.info(f"{name.rjust(len_names)}  |  {version}"   )
    logger.info(split_str)



def lstrip_multiple(s:str, chars:list) :
	"""
	Strip many characters from the LHS of string s

	Inputs:

		>  s, str
		   String to be stripped

		>  chars, list
		   Characters to be stripped
	"""
	while len(s) > 0 and s[0] in chars :
		s = s[1:]
	return s



def rstrip_multiple(s:str, chars:list) :
	"""
	Strip many characters from the RHS of string s

	Inputs:

		>  s, str
		   String to be stripped

		>  chars, list
		   Characters to be stripped
	"""
	return lstrip_multiple(s[::-1], chars=chars)[::-1]



def summarise_dict(dict_or_obj, base_str:str="", global_summary_list:list=[], lvl_separator:str=">") :
    """Return a list of strings summarising objects in a nested dictionary

    Inputs:

    	>  dict_or_obj, dict or printable object
    	   Dictionary to be recursively iterated, or object to be printed

    	>  base_str, str, default=''
		   String summarising the higher-level namespaces in which the current object is nested

		>  global_summary_list, list, default=[]
		   List containing strings of all objects in the dictionary that have been created so far, to be appended with current object

		>  lvl_separator, str, default=">"
		   String use to represent passing to a lower level of the dictionary
    """

    ##  If we were given a dictionary then iterate and recursively call this function on all elements
    ##  Otherwise append a new summary string to the global list for this object, using the base string to resolve all higher-level namespaces
    if type(dict_or_obj) is dict :
        for k, v in dict_or_obj.items() :
            new_base_str = f"{base_str} {lvl_separator} {k}" if len(base_str) > 0 else str(k)
            summarise_dict(v, new_base_str, global_summary_list, lvl_separator)
    else :
        global_summary_list.append(f"{base_str}: {dict_or_obj}")

    ##  Return list of all object summarise created so far - at top-level this will be all objects in the dictionary
    return global_summary_list
