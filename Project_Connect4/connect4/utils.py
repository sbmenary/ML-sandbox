###
###  connect4.enums.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Various utils to make life easier.
"""

from __future__ import annotations
from enum       import IntEnum



###=================================###
###   DebugLevel class definition   ###
###=================================###

class DebugLevel(IntEnum):
    """
    An enumeration for the verbosity level of debug statements
    Options are: MUTE=0, LOW=1, MEDIUM=2, HIGH=3, ALL=4
    Use of IntEnum allows labelled int operations like "if lvl >= DebugLevel.MEDIUM :"
    """
    MUTE   = 0
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3
    ALL    = 4
    
    def message(self, min_lvl:DebugLevel, message:str) -> bool :
        """
        Print message only if self >= min_lvl.
        Returns bool indicating whether the message was printed.
        """

        ##  If debug_lvl >= min_lvl then print message and return True
        if self >= min_lvl :
            print(message)
            return True

        ##  Otherwise no message printed and return False
        return False
