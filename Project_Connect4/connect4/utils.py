###
###  connect4.enums.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Various utils to make life easier.
"""

from __future__ import annotations
from enum       import IntEnum
import logging, sys


##  Global logger for this module
logger = logging.getLogger(__name__)

##  Global root logger and handlers
root_logger       = logging.getLogger()
log_stdout_stream = None
log_file_handlers = []



###=================================###
###   DebugLevel class definition   ###
###=================================###

class DebugLevel(IntEnum):
    """
    An enumeration for the verbosity level of debug statements. 
    Usual logging values are: [CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, NOTSET=0]
    This doesn't provide a huge amount of debug granularity, so we enumerate a few extra levels around this.
    Options are: MUTE=100, LOWEST=14, VERY_LOW=13, LOW=12, MED_LOW=11, MEDIUM=10, MED_HIGH=9, HIGH=8, VERY_HIGH=7, HIGHEST=6, ALL=1
    MEDIUM is equivalent to logging.DEBUG
    Use of IntEnum allows labelled int operations like "if debug_level >= DebugLevel.MEDIUM :"
    """
    MUTE      = 100
    LOWEST    = 14
    VERY_LOW  = 13
    LOW       = 12
    MED_LOW   = 11
    MEDIUM    = 10
    MED_HIGH  = 9
    HIGH      = 8
    VERY_HIGH = 7
    HIGHEST   = 6
    ALL       = 1
    
    def message(self, max_lvl:DebugLevel, message:str) -> bool :
        """
        Print message only if self <= max_lvl.
        Returns bool indicating whether the message was printed.
        """

        ##  If debug_lvl <= max_lvl then print message and return True
        if self <= max_lvl :
            print(message)
            return True

        ##  Otherwise no message printed and return False
        return False



###======================###
###   Method defitions   ###
###======================###
  
    
def add_logfile(fname:str, loglevel:int, mode:str="w+") :
    """
    Add a new logfile at the path and with the loglevel provided
    """
    ##  We want to interact with global variables
    global root_logger, log_stdout_stream, log_file_handlers

    ##  Create logfile using the path and mode provided
    fh = logging.FileHandler(fname, mode=mode)
    
    ##  Set format and loglevel
    fh.setFormatter(logging.Formatter('%(asctime)s  %(name)s  %(levelname)s  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
    fh.setLevel(loglevel)
    
    ##  Add handler to root logger
    root_logger.addHandler(fh)
    
    ##  Store handler for future reference
    log_file_handlers.append(fh)

    ##  Make sure logger level equal to the maximum level of the streams
    root_logger.setLevel(min([log_stdout_stream.level if log_stdout_stream else logging.INFO] + [fh.level for fh in log_file_handlers]))


def set_loglevel(loglevel:int) :
    """
    Set the loglevel of the stdout stream
    Create the stream if not done already
    """
    ##  We want to interact with global variables
    global root_logger, log_stdout_stream
    
    ##  If stdout stream doesn't exist yet, then create and add to root logger
    if not log_stdout_stream :
        log_stdout_stream = logging.StreamHandler(sys.stdout)
        log_stdout_stream.setFormatter(logging.Formatter('%(name)s  %(message)s'))
        root_logger.addHandler(log_stdout_stream)
    
    ##  Set loglevel
    log_stdout_stream.setLevel(loglevel)

    ##  Make sure logger level equal to the maximum level of the streams
    root_logger.setLevel(min([log_stdout_stream.level if log_stdout_stream else logging.INFO] + [fh.level for fh in log_file_handlers]))
